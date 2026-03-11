from config import (
    COLLECTION_DISPLAY_TO_KEY,
    DCD_N_RESULTS,
    DCD_RERANK_TOP_N,
    DOMAIN_DISPLAY_TO_KEY,
    collection_chroma,
    reranker_model,
)
from prompts import system_prompt_clf_collection, system_prompt_clf_domain
from utils import classification_by_so


def _display_to_key(display: str, mapping: dict, kind: str) -> str:
    """Convert display name to key; raise clear error if not in mapping."""
    key = mapping.get(display.strip() if display else "")
    if key is None:
        raise ValueError(
            f"Classifier returned unknown {kind}: {display!r}. "
            f"Expected one of: {list(mapping.keys())}"
        )
    return key


def domain_classification(query_text: str, domains_scheme):
    domain = classification_by_so(query_text, domains_scheme, system_prompt_clf_domain)

    print(f"{domain.reasoning=}")
    print(f"{domain.domain=}")

    return domain.domain

def collection_classification(query_text: str, collection_scheme):
    collection = classification_by_so(query_text, collection_scheme, system_prompt_clf_collection)

    print(f"{collection.reasoning=}")
    print(f"{collection.collection=}")

    return collection.collection

def search_context(query_text: str, domain: str, collection: str):
    results = collection_chroma.query(
        query_texts=[query_text],
        where={"$and": [{"domain": domain}, {"collection": collection}]},
        n_results=DCD_N_RESULTS,
    )
    return results


def rerank_results(query_text: str, chroma_results: dict, top_n: int = None):
    """Take raw Chroma results and return sorted list of top_n chunks by reranker score."""
    if top_n is None:
        top_n = DCD_RERANK_TOP_N
    documents = chroma_results["documents"][0]
    metadatas = chroma_results["metadatas"][0]
    ids = chroma_results["ids"][0]

    if not documents:
        return []

    pairs = [[query_text, doc] for doc in documents]
    scores = reranker_model.predict(pairs)

    reranked_data = []
    for i in range(len(documents)):
        reranked_data.append({
            "id": ids[i],
            "score": scores[i],
            "content": documents[i],
            "metadata": metadatas[i],
        })
    reranked_data.sort(key=lambda x: x["score"], reverse=True)

    return reranked_data[:top_n]


def pipeline_dcd(query_text: str, domains_scheme, collection_scheme):
    """Full pipeline: classify domain/collection -> search Chroma -> rerank -> return top chunks."""
    print("domain classification...")
    domain = domain_classification(query_text, domains_scheme)

    print("collection classification...")
    collection_name = collection_classification(query_text, collection_scheme)

    print("search context...")
    domain_key = _display_to_key(domain, DOMAIN_DISPLAY_TO_KEY, "domain")
    collection_key = _display_to_key(collection_name, COLLECTION_DISPLAY_TO_KEY, "collection")

    raw_results = search_context(
        query_text=query_text,
        domain=domain_key,
        collection=collection_key,
    )

    print("reranking...")
    reranked = rerank_results(query_text=query_text, chroma_results=raw_results)
    return reranked