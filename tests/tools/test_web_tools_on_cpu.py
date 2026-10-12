# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the web_search / open_page tools (HTTP calls mocked)."""

import json

import pytest
import requests

from verl.tools import web_tools as wt
from verl.utils.reward_score import _tool_execution_bonus

LISBOA = {"title": "Lisboa", "url": "https://pt.wikipedia.org/wiki/Lisboa"}
PORTO = {"title": "Porto", "url": "https://pt.wikipedia.org/wiki/Porto"}


def _result(result_json):
    return json.loads(result_json)["result"]


class _Response:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


def _serve(monkeypatch, *responses):
    """Make requests.post return the given responses in order, and record the payloads."""
    queue, calls = list(responses), []

    def post(url, json=None, timeout=None):
        calls.append(json)
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(wt.requests, "post", post)
    monkeypatch.setattr(wt.time, "sleep", lambda s: None)
    return calls


def test_search_results_list_titles_and_urls_per_query():
    text, total = wt.format_search_results(["capital", "norte"], [[LISBOA, PORTO], [PORTO]])
    assert total == 3
    assert text == (
        'Query: "capital"\n1. Lisboa\n   https://pt.wikipedia.org/wiki/Lisboa\n'
        "2. Porto\n   https://pt.wikipedia.org/wiki/Porto\n"
        '---\nQuery: "norte"\n1. Porto\n   https://pt.wikipedia.org/wiki/Porto'
    )


def test_web_search_sends_limits_and_formats(monkeypatch):
    calls = _serve(monkeypatch, _Response(200, {"result": [[LISBOA]]}))
    out, meta = wt.perform_web_search("http://x/web_search", ["capital"], 10, 5, 5)
    assert calls == [{"queries": ["capital"], "topk": 10, "max_results": 5}]
    assert "1. Lisboa\n   https://pt.wikipedia.org/wiki/Lisboa" in _result(out)
    assert meta["status"] == "success" and meta["total_results"] == 1


def test_web_search_no_results_and_errors(monkeypatch):
    _serve(monkeypatch, _Response(200, {"result": [[]]}))
    out, meta = wt.perform_web_search("http://x/web_search", ["q"], 10, 10, 5)
    assert _result(out) == "No search results found." and meta["status"] == "no_results"

    _serve(monkeypatch, *[requests.exceptions.ConnectionError("refused")] * wt.MAX_RETRIES)
    out, meta = wt.perform_web_search("http://x/web_search", ["q"], 10, 10, 5)
    assert _result(out).startswith("Search error:") and meta["status"] == "api_error"


@pytest.mark.parametrize(
    "value,expected",
    [("Rio Tejo\n\nfoz do Tejo\n", ["Rio Tejo", "foz do Tejo"]), (["a", " "], ["a"]), (None, []), (7, ["7"])],
)
def test_query_list_coercion(value, expected):
    assert wt._as_query_list(value) == expected


def test_open_page_found(monkeypatch):
    page = {**LISBOA, "text": "Lisboa é a capital de Portugal."}
    _serve(monkeypatch, _Response(200, {"result": [page]}))
    out, meta = wt.perform_open_page("http://x/page", "https://pt.wikipedia.org/wiki/Lisboa", 5)
    assert _result(out) == "Title: Lisboa\nURL: https://pt.wikipedia.org/wiki/Lisboa\n\nLisboa é a capital de Portugal."
    assert meta["status"] == "success"


def test_open_page_not_found(monkeypatch):
    _serve(monkeypatch, _Response(200, {"result": [None]}))
    out, meta = wt.perform_open_page("http://x/page", "https://example.com/x", 5)
    assert _result(out) == 'Error: page not found: "https://example.com/x". Use a URL returned by web_search.'
    assert meta["status"] == "not_found"


def test_open_page_retries_server_errors_then_succeeds(monkeypatch):
    page = {**PORTO, "text": "t"}
    calls = _serve(monkeypatch, _Response(503), _Response(200, {"result": [page]}))
    out, meta = wt.perform_open_page("http://x/page", PORTO["url"], 5)
    assert len(calls) == 2 and meta["status"] == "success"


def test_open_page_client_error_is_not_retried(monkeypatch):
    calls = _serve(monkeypatch, _Response(404))
    out, meta = wt.perform_open_page("http://x/page", PORTO["url"], 5)
    assert len(calls) == 1 and meta["status"] == "api_error"
    assert _result(out).startswith("Error: could not open the page")


def _trajectory(tool_output):
    return (
        "<tool_call>\n<function=open_page>\n<parameter=url>\nhttps://pt.wikipedia.org/wiki/Lisboa\n"
        "</parameter>\n</function>\n</tool_call>"
        f"user\n<tool_response>\n{tool_output}\n</tool_response>\nassistant\n<answer>Lisboa</answer>"
    )


def test_tool_bonus_rewards_opened_pages_only():
    found = wt._result("Title: Lisboa\nURL: https://pt.wikipedia.org/wiki/Lisboa\n\nLisboa é a capital.")
    not_found = wt._result('Error: page not found: "x". Use a URL returned by web_search.')
    failed = wt._result("Error: could not open the page: refused")
    assert _tool_execution_bonus(_trajectory(found), bonus=0.1) == 0.1
    assert _tool_execution_bonus(_trajectory(not_found), bonus=0.1) == 0.0
    assert _tool_execution_bonus(_trajectory(failed), bonus=0.1) == 0.0
