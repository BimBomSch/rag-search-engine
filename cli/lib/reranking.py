import os
import json
from time import sleep

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key and not base_url:
    raise RuntimeError("OPENAI_API_KEY, BASE_URL environment variables not set")

client = OpenAI(base_url=base_url, api_key=api_key,)
model = "google/gemma-3-27b"


def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        response = client.responses.create(model=model, input=prompt,)
        score_text = (response.output_text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(0.1)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]

def llm_rerank_batch(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    if not documents:
        return []

    doc_map = {}
    doc_list = []
    for doc in documents:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON array of numbers, nothing else.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""
    
    response = client.responses.create(model=model, input=prompt,)
    ranking_text = (response.output_text or "").strip()

    parsed_ids = json.loads(ranking_text)

    reranked = []
    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            reranked.append({**doc_map[doc_id], "batch_rank": i + 1})

    return reranked[:limit]

def rerank(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    if method == "batch":
        return llm_rerank_batch(query, documents, limit)
    else:
        return documents[:limit]
