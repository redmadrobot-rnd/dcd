import re
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    """Loads a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_templates_config() -> dict:
    """Loads templates_config.yaml."""
    config_path = Path(__file__).parent / "templates_config.yaml"
    return load_yaml(config_path)


def load_domains_variables() -> dict:
    """Loads domains_variables.yaml."""
    config_path = Path(__file__).parent / "domains_variables.yaml"
    return load_yaml(config_path)


def get_domain_list(domains_vars: dict) -> list[str]:
    """Returns the list of domains from domains_variables."""
    return list(domains_vars.get("domains", {}).keys())


def substitute_variables(text: str, variables: dict) -> str:
    """
    Substitutes variables in text.
    Variables are specified as {{variable_name}}.
    If variable is not found, '—' is substituted.
    """
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        return str(variables.get(var_name, "—"))

    return re.sub(r"\{\{(\w+)\}\}", replacer, text)


def generate_dataset() -> None:
    """Generates dataset from templates and domain variables."""
    base_dir = Path(__file__).parent
    templates_dir = base_dir / "templates"

    # Load configuration
    templates_config = load_templates_config()
    domains_vars = load_domains_variables()

    output_dir = "../output"
    output_path = base_dir / output_dir

    domains = get_domain_list(domains_vars)
    domain_variables = domains_vars.get("domains", {})

    if not domains:
        print("Error: domains not found in domains_variables.yaml")
        return

    print(f"Generating dataset for {len(domains)} domains...")
    print(f"Output folder: {output_path}")

    total_docs = 0

    for domain in domains:
        variables = domain_variables.get(domain, {})
        if not variables:
            print(f"  Warning: variables not found for domain {domain}")
            continue

        domain_output = output_path / domain
        domain_output.mkdir(parents=True, exist_ok=True)

        for collection, templates in templates_config.get("collections", {}).items():
            collection_output = domain_output / collection
            collection_output.mkdir(parents=True, exist_ok=True)

            for item in templates:
                template_name = item.get("template", "")
                output_name = item.get("output_name", template_name)

                template_path = templates_dir / collection / f"{template_name}.txt"

                if not template_path.exists():
                    print(f"  Warning: template not found: {template_path}")
                    continue

                # Read template
                with open(template_path, encoding="utf-8") as f:
                    template_text = f.read()

                # Substitute variables
                result_text = substitute_variables(template_text, variables)

                # Save result
                output_file = collection_output / f"{output_name}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result_text)

                total_docs += 1

        print(f"  {domain}: done")

    print(f"\nDone. Generated documents: {total_docs}")


if __name__ == "__main__":
    generate_dataset()
