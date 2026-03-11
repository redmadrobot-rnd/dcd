from config import NAIVE_RAG_N_RESULTS, collection_chroma


def pipeline_naive_rag(query_text: str):
    """Naive RAG: search entire DB, return top N chunks (no reranking)."""
    results = collection_chroma.query(
        query_texts=[query_text],
        n_results=NAIVE_RAG_N_RESULTS,
    )
    return [
        {
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]