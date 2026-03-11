"""
Mapping of domain and collection keys to human-readable names.
Used for question prefixes and RAG metadata filtering.

RAG: question → classifier returns domain_key, collection_key →
filter vector search by metadata.domain == domain_key and metadata.collection == collection_key
"""

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "dataset_config.yaml"
_dataset_config: dict | None = None


def _get_dataset_config() -> dict:
    """Load dataset_config.yaml (cached)."""
    global _dataset_config
    if _dataset_config is None:
        if not _CONFIG_PATH.is_file():
            _dataset_config = {}
        else:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                _dataset_config = yaml.safe_load(f) or {}
    return _dataset_config


def get_dataset_config() -> dict:
    """Return dataset config (qa_pairs_per_document, domain_display_names, collection_display_names)."""
    return _get_dataset_config()


def get_domain_display_name(config: dict, domain_key: str) -> str:
    """Return display name for domain. Priority: config.display_name → dataset_config → key."""
    names = _get_dataset_config().get("domain_display_names", {})
    for d in config.get("domains", []):
        if d.get("name") == domain_key:
            return d.get("display_name") or names.get(domain_key, domain_key)
    return names.get(domain_key, domain_key)


def get_collection_display_name(config: dict, domain_key: str, collection_key: str) -> str:
    """Return display name for collection. Priority: config.display_name → dataset_config → key."""
    names = _get_dataset_config().get("collection_display_names", {})
    for d in config.get("domains", []):
        if d.get("name") == domain_key:
            for c in d.get("collections", []):
                if c.get("name") == collection_key:
                    return c.get("display_name") or names.get(collection_key, collection_key)
    return names.get(collection_key, collection_key)


def build_metadata_mapping(config: dict) -> dict:
    """
    Build full mapping for RAG: domain_key → display_name, collection_key → display_name.
    Saved to metadata_mapping.json and used for metadata filtering.
    """
    domains = {}
    collections = {}

    for d in config.get("domains", []):
        key = d.get("name", "")
        if key:
            domains[key] = get_domain_display_name(config, key)
            for c in d.get("collections", []):
                ckey = c.get("name", "")
                if ckey:
                    collections[ckey] = get_collection_display_name(config, key, ckey)

    return {"domains": domains, "collections": collections}


def format_question_prefix(domain_display: str, collection_display: str) -> str:
    """Prefix for question: RC «X», section «Y»."""
    return f"RC «{domain_display}», section «{collection_display}»: "


def build_metadata_mapping_from_documents(output_dir: str | Path) -> dict:
    """
    Build metadata_mapping and config from existing output/ structure.
    Used by build_rag_dataset_from_documents.py when documents are pre-generated.

    Scans output/{domain}/{collection}/ for .txt files.
    Loads domains_variables.yaml for domain display names (complex_name).

    Returns:
        {
            "metadata_mapping": {domains: {...}, collections: {...}},
            "config": config-like dict for get_domain_display_name / get_collection_display_name
        }
    """
    output_path = Path(output_dir)
    base_dir = output_path.parent

    domains_vars_path = base_dir / "domains_variables.yaml"
    if not domains_vars_path.is_file():
        domains_vars_path = base_dir / "1. Create text dataset" / "domains_variables.yaml"
    domains_vars = {}
    if domains_vars_path.is_file():
        with open(domains_vars_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            domains_vars = data.get("domains", {})

    domains = {}
    collections = {}
    config_domains = []

    for domain_path in output_path.iterdir():
        if not domain_path.is_dir():
            continue
        domain_key = domain_path.name
        cfg = _get_dataset_config()
        domain_names = cfg.get("domain_display_names", {})
        coll_names = cfg.get("collection_display_names", {})
        domain_display = (
            domains_vars.get(domain_key, {}).get("complex_name")
            or domain_names.get(domain_key, domain_key)
        )
        domains[domain_key] = domain_display

        coll_list = []
        for collection_path in domain_path.iterdir():
            if not collection_path.is_dir():
                continue
            collection_key = collection_path.name
            collection_display = coll_names.get(
                collection_key, collection_key.replace("_", " ").title()
            )
            collections[collection_key] = collection_display
            coll_list.append({"name": collection_key, "display_name": collection_display})

        config_domains.append(
            {"name": domain_key, "display_name": domain_display, "collections": coll_list}
        )

    metadata_mapping = {"domains": domains, "collections": collections}
    config = {"domains": config_domains}

    return {"metadata_mapping": metadata_mapping, "config": config}
