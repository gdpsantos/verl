# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py

import argparse
import json
import unicodedata
import urllib.parse
import warnings
from collections import defaultdict
from typing import Optional

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def _page_name(url_or_title: str) -> str:
    """Page name from a URL, tolerant of how the URL was written.

    Takes the part after "/wiki/" (so http/https, host and trailing "/" don't matter),
    undoes percent-encoding and treats "_" as a space: ".../wiki/S%C3%A3o%20Tom%C3%A9"
    and ".../wiki/São_Tomé" both give "São Tomé".
    """
    name = url_or_title.strip().split("#")[0].rstrip("/")
    if "/wiki/" in name:
        name = name.split("/wiki/", 1)[1]
    return " ".join(urllib.parse.unquote(name).replace("_", " ").split())


def _fold(name: str) -> str:
    """Case- and accent-insensitive form of a page name."""
    text = unicodedata.normalize("NFD", name.casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def find_page(url: str) -> Optional[str]:
    """Title of the page a URL points to, or None.

    Tries an exact match first, then ignores case, then also accents, so a sloppy URL
    still opens its page but never shadows a page that matches more exactly
    ("Baço" and "Baco" are different pages).
    """
    name = _page_name(url)
    if name in pages:
        return name
    return page_keys["case"].get(name.casefold()) or page_keys["fold"].get(_fold(name))


def build_page_index(corpus_path: str):
    """Group corpus passages into whole web pages (articles) for /web_search and /page.

    Passage ids look like "<article>-<chunk>"; passages are ordered by chunk number.
    Each page gets the readable form of its corpus URL ("https://pt.wikipedia.org/wiki/São_Tomé").
    Returns {title: {"url", "text"}} and the lookup tables used by find_page.
    """
    passages, urls = defaultdict(list), {}
    with open(corpus_path) as f:
        for line_no, line in enumerate(f):
            doc = json.loads(line)
            lines = doc["contents"].split("\n")
            # The title line of "contents" is what BM25 hits report, so pages are keyed by it.
            title = lines[0].strip('"')
            chunk = doc.get("id", "").rsplit("-", 1)[-1]
            order = int(chunk) if chunk.isdigit() else line_no
            passages[title].append((order, "\n".join(lines[1:]).strip()))
            if title not in urls and doc.get("url"):
                urls[title] = urllib.parse.unquote(doc["url"]).replace(" ", "_")
    pages = {
        title: {"url": urls.get(title, "https://pt.wikipedia.org/wiki/" + title.replace(" ", "_")),
                "text": "\n".join(text for _, text in sorted(parts))}
        for title, parts in passages.items()
    }
    keys = {"case": {}, "fold": {}}
    for title in pages:
        keys["case"].setdefault(title.casefold(), title)
        keys["fold"].setdefault(_fold(title), title)
    return pages, keys


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}" for query in query_list
                ]

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        torch.cuda.empty_cache()

        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: list[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        # The query analyzer must match the one the index was built with
        # (pyserini --language). LuceneSearcher defaults to English, so a
        # Portuguese index queried without this returns near-random passages.
        language = getattr(config, "language", None)
        if language:
            self.searcher.set_language(language)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!", stacklevel=2)
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [
                {
                    "title": content.split("\n")[0].strip('"'),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content,
                }
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: "):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()

        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        language: str = None,
    ):
        self.retrieval_method = retrieval_method
        self.language = language
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.

    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }

    Output format (when return_scores=True，similarity scores are returned):
    {
        "result": [
            [   # Results for each query
                {
                    {"document": doc, "score": score}
                },
                # ... more documents
            ],
            # ... results for other queries
        ]
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries, num=request.topk, return_score=request.return_scores
    )

    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i], strict=True):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


class WebSearchRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    max_results: int = 10


@app.post("/web_search")
def web_search_endpoint(request: WebSearchRequest):
    """
    Simulated web search: the pages (title and URL) of the best-matching passages.

    Retrieves `topk` passages per query and lists their pages in rank order, without
    duplicates (a page can have several passages), at most `max_results` per query.

    Input format:
    {"queries": ["capital de Portugal"], "topk": 10, "max_results": 10}

    Output format:
    {"result": [[{"title": "Lisboa", "url": "https://pt.wikipedia.org/wiki/Lisboa"}, ...]]}
    """
    results = retriever.batch_search(query_list=request.queries, num=request.topk or config.retrieval_topk)
    resp = []
    for hits in results:
        titles = []
        for hit in hits:
            title = hit.get("title") or hit["contents"].split("\n")[0].strip('"')
            if title in pages and title not in titles:
                titles.append(title)
        resp.append([{"title": t, "url": pages[t]["url"]} for t in titles[: request.max_results]])
    return {"result": resp}


class PageRequest(BaseModel):
    urls: list[str]


@app.post("/page")
def page_endpoint(request: PageRequest):
    """
    Open web pages by URL: the whole article (all of its passages, in order).

    The URL may be written readably or percent-encoded, with http or https and a
    trailing "/". If no page matches exactly, case and then accents are ignored.

    Input format:
    {"urls": ["https://pt.wikipedia.org/wiki/Lisboa", "https://example.com/x"]}

    Output format (null when no page has that URL):
    {"result": [{"title": "Lisboa", "url": "https://pt.wikipedia.org/wiki/Lisboa", "text": "..."}, null]}
    """
    resp = []
    for url in request.urls:
        title = find_page(url)
        resp.append({"title": title, **pages[title]} if title else None)
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument(
        "--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl",
        help="Local corpus file.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument(
        "--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model."
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Query-analyzer language for BM25; must match the --language used at index time (e.g. pt).",
    )

    args = parser.parse_args()

    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method=args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        language=args.language,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)

    # Whole articles as web pages, for the /web_search and /page endpoints.
    pages, page_keys = build_page_index(args.corpus_path)
    print(f"Page index: {len(pages)} pages")

    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host=args.host, port=args.port)
