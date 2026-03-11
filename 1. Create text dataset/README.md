# Dataset Generator

**What it does:** Builds the text corpus for RAG—one document per template per domain, with placeholders filled from domain variables.

Generates a text dataset by filling templates with per-domain variables. One run produces one `.txt` file per template per domain.

## Structure

```
Create text dataset/
├── generate_dataset.py      # run this
├── templates_config.yaml    # which templates to use
├── domains_variables.yaml   # variables per domain
└── templates/               # template files
    ├── apartments/
    │   ├── Pricing and Purchase Terms.txt
    │   └── ...
    ├── infrastructure/
    └── ...
```

**Output:** `../output/<domain>/<collection>/<output_name>.txt`

## Required files

### 1. `templates_config.yaml`

Maps collection names to a list of templates. Each item: `template` (filename without `.txt`) and optional `output_name`.

```yaml
collections:
  apartments:
    - template: "Pricing and Purchase Terms"
      output_name: "Pricing and Purchase Terms"
  infrastructure:
    - template: "Parking and Spaces"
```

### 2. `domains_variables.yaml`

One entry per domain under `domains:`. Each domain is a flat key-value map; keys are variable names used in templates.

```yaml
domains:
  my_domain:
    complex_name: Sunny Shore
    studio_price: 4.1M
    mortgage_rate: 5.9%
```

### 3. Template files (`templates/<collection>/<name>.txt`)

Plain text, UTF-8. Use `{{variable_name}}` for substitution. Missing variables become `—`.

**Example** `templates/apartments/Pricing.txt`:

```
Prices at {{complex_name}}: studio from {{studio_price}}, mortgage rate {{mortgage_rate}}.
```

## Run

```bash
pip install pyyaml
python generate_dataset.py
```

Resulting files appear under `../output/<domain>/<collection>/`.
