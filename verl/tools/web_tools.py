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

"""Simulated web browsing over the local corpus: search the web, then open a page.

``web_search`` lists the pages (title and URL) that best match each query, like a
search engine results page without snippets. ``open_page`` takes one of those URLs
and returns the page's text. Both are served by the retrieval server
(``/web_search`` and ``/page``). Results are ``{"result": ...}`` JSON like
``SearchTool``, and failure messages start with ``Error:`` or ``Search error:``, so
the tool-execution bonus does not reward failed calls.
"""

import json
import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

import requests

from verl.tools.search_tool import PoolMode, init_search_execution_pool
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MAX_RETRIES = 5


def _result(text: str) -> str:
    return json.dumps({"result": text}, ensure_ascii=False)


def _as_query_list(value: Any) -> list[str]:
    """XML tool-call formats deliver parameter values as text: one query per non-empty line."""
    if isinstance(value, str):
        return [q.strip() for q in value.splitlines() if q.strip()]
    if isinstance(value, list):
        return [str(q).strip() for q in value if str(q).strip()]
    if value is None:
        return []
    return [str(value).strip()]


def _post_json(url: str, payload: dict, timeout: int) -> tuple[Optional[dict], Optional[str]]:
    """POST JSON, retrying server errors, connection errors and timeouts.

    Returns (response JSON, None) on success and (None, error message) on failure.
    """
    error = "no response"
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code in (500, 502, 503, 504):
                error = f"Server Error ({response.status_code})"
            else:
                response.raise_for_status()
                return response.json(), None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            error = str(e)
        except Exception as e:
            return None, str(e)
        if attempt < MAX_RETRIES - 1:
            time.sleep(attempt + 1)
    return None, error


def format_search_results(query_list: list[str], results: list[list[dict]]) -> tuple[str, int]:
    """Format /web_search results: per query, a numbered list of page titles with their URLs.

    Returns the formatted text and the total number of pages listed.
    """
    blocks, total = [], 0
    for query, pages in zip(query_list, results, strict=False):
        total += len(pages)
        lines = [f"{i + 1}. {p['title']}\n   {p['url']}" for i, p in enumerate(pages)] or ["(no results)"]
        blocks.append(f'Query: "{query}"\n' + "\n".join(lines))
    return "\n---\n".join(blocks), total


def perform_web_search(
    web_search_url: str, query_list: list[str], topk: int, max_results: int, timeout: int
) -> tuple[str, dict[str, Any]]:
    """Run a simulated web search and return (result_json, metadata)."""
    response, error = _post_json(
        web_search_url, {"queries": query_list, "topk": topk, "max_results": max_results}, timeout
    )
    metadata = {"query_count": len(query_list), "status": "unknown", "total_results": 0, "api_request_error": error}
    if error:
        metadata["status"] = "api_error"
        return _result(f"Search error: {error}"), metadata
    text, total = format_search_results(query_list, response.get("result", []))
    if total == 0:
        metadata["status"] = "no_results"
        return _result("No search results found."), metadata
    metadata.update(status="success", total_results=total)
    return _result(text), metadata


def perform_open_page(page_url: str, url: str, timeout: int) -> tuple[str, dict[str, Any]]:
    """Open one page by URL and return (result_json, metadata)."""
    response, error = _post_json(page_url, {"urls": [url]}, timeout)
    metadata = {"status": "unknown", "found": False, "api_request_error": error}
    if error:
        metadata["status"] = "api_error"
        return _result(f"Error: could not open the page: {error}"), metadata
    page = response["result"][0]
    if page is None:
        metadata["status"] = "not_found"
        return _result(f'Error: page not found: "{url}". Use a URL returned by web_search.'), metadata
    metadata.update(status="success", found=True)
    return _result(f"Title: {page['title']}\nURL: {page['url']}\n\n{page['text']}"), metadata


class _WebToolBase(BaseTool):
    """Shared setup: timeout and the job-wide Ray execution pool used by SearchTool."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.timeout = config.get("timeout", 30)
        self.execution_pool = init_search_execution_pool(
            num_workers=config.get("num_workers", 120),
            enable_global_rate_limit=config.get("enable_global_rate_limit", True),
            rate_limit=config.get("rate_limit", 120),
            mode=PoolMode.ThreadMode,
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        return instance_id or str(uuid4()), ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        pass


class WebSearchTool(_WebToolBase):
    """List the pages (title and URL) that best match each query.

    Config: ``web_search_url`` (the server's ``/web_search`` endpoint), ``topk`` passages
    retrieved per query (default 10), ``max_results`` pages listed per query (default 10).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.web_search_url = config.get("web_search_url")
        assert self.web_search_url, "Configuration must include 'web_search_url'"
        self.topk = config.get("topk", 10)
        self.max_results = config.get("max_results", 10)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query_list = _as_query_list(parameters.get("query_list"))
        if not query_list:
            return ToolResponse(text=_result("Error: 'query_list' is missing or empty.")), 0.0, {}
        try:
            text, metadata = await self.execution_pool.execute.remote(
                perform_web_search, self.web_search_url, query_list, self.topk, self.max_results, self.timeout
            )
        except Exception as e:
            logger.error(f"[WebSearchTool] Execution failed: {e}")
            return ToolResponse(text=_result(f"Search execution failed: {e}")), 0.0, {"error": str(e)}
        return ToolResponse(text=text), 0.0, metadata


class OpenPageTool(_WebToolBase):
    """Open one page by its URL (as listed by ``web_search``) and return its text.

    Config: ``page_url`` (the server's ``/page`` endpoint).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.page_url = config.get("page_url")
        assert self.page_url, "Configuration must include 'page_url'"

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        url = parameters.get("url")
        if isinstance(url, list) and len(url) == 1:
            url = url[0]
        if not isinstance(url, str) or not url.strip():
            return ToolResponse(text=_result("Error: 'url' must be one page URL.")), 0.0, {}
        try:
            text, metadata = await self.execution_pool.execute.remote(
                perform_open_page, self.page_url, url.strip(), self.timeout
            )
        except Exception as e:
            logger.error(f"[OpenPageTool] Execution failed: {e}")
            return ToolResponse(text=_result(f"Error: could not open the page: {e}")), 0.0, {"error": str(e)}
        return ToolResponse(text=text), 0.0, metadata
