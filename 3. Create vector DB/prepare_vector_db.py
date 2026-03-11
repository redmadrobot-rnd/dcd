"""
Build Chroma vector DB from documents in output/: chunk .txt files, embed, and add to collection.
Config: vector_db_config.yaml (paths, chunking, embedding model, batch size).
"""

import json
from pathlib import Path

import chromadb
import tiktoken
import yaml
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def load_config() -> dict:
    """Load vector_db_config.yaml from script directory."""
    config_path = Path(__file__).parent / "vector_db_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def tiktoken_len(text: str, tokenizer) -> int:
    """Return token count for text (used by RecursiveCharacterTextSplitter)."""
    return len(tokenizer.encode(text))


def chunk_documents(
    root_dir: str,
    text_splitter: RecursiveCharacterTextSplitter,
) -> list[dict]:
    """
    Walk root_dir recursively, find all .txt files, split into chunks, return list of chunk dicts.

    Expected folder structure:
        root_dir/
            domain1/
                collection1/
                    doc1.txt
                collection2/
                    doc2.txt
            domain2/...

    Each chunk dict: {"chunk": str, "metadata": {domain, collection, document, chunk_number}}.
    """
    chunks = []
    root = Path(root_dir)

    for domain_dir in root.iterdir():
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name

        for collection_dir in domain_dir.iterdir():
            if not collection_dir.is_dir():
                continue
            collection = collection_dir.name

            for txt_file in collection_dir.glob("*.txt"):
                try:
                    text = txt_file.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"Error reading {txt_file}: {e}")
                    continue

                chunk_texts = text_splitter.split_text(text)

                for idx, chunk_text in enumerate(chunk_texts):
                    chunk_dict = {
                        "chunk": chunk_text,
                        "metadata": {
                            "domain": domain,
                            "collection": collection,
                            "document": txt_file.stem,
                            "chunk_number": idx,
                        },
                    }
                    chunks.append(chunk_dict)

    return chunks


def main() -> None:
    config = load_config()

    chroma_path = config.get("chroma_path", "./my_db")
    collection_name = config.get("collection_name", "dcd_collection")
    embedding_model = config.get("embedding_model", "BAAI/bge-m3")
    tokenizer_encoding = config.get("tokenizer_encoding", "cl100k_base")
    chunk_size = config.get("chunk_size", 450)
    chunk_overlap = config.get("chunk_overlap", 30)
    separators = config.get("separators", ["\n\n", "\n", ".", " ", ""])
    documents_root = config.get("documents_root", "../output")
    chunks_output_file = config.get("chunks_output_file", "chunks_output.json")
    batch_size = config.get("batch_size", 10)

    tokenizer = tiktoken.get_encoding(tokenizer_encoding)

    def length_fn(text: str) -> int:
        return tiktoken_len(text, tokenizer)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_fn,
        separators=separators,
    )

    base_dir = Path(__file__).resolve().parent
    root_abs = (base_dir / documents_root).resolve()
    all_chunks = chunk_documents(str(root_abs), splitter)

    out_path = base_dir / chunks_output_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    client = chromadb.PersistentClient(path=str(base_dir / chroma_path.lstrip("./")))
    embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    client.create_collection(
        name=collection_name,
        embedding_function=embed,
        get_or_create=True,
    )
    collection = client.get_collection(name=collection_name, embedding_function=embed)

    docs = [el["chunk"] for el in all_chunks]
    metadata = [el["metadata"] for el in all_chunks]
    ids = [str(i) for i in range(len(docs))]

    for i in tqdm(range(0, len(docs), batch_size), desc="Adding to Chroma"):
        batch_ids = ids[i : i + batch_size]
        batch_docs = docs[i : i + batch_size]
        batch_metadata = metadata[i : i + batch_size]
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metadata,
        )


if __name__ == "__main__":
    main()
