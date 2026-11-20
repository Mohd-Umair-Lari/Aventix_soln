# Aventix Real-Data KGRAG Notebook

This project contains a private Python/Jupyter implementation of a grounded Knowledge Graph Retrieval-Augmented Generation (KGRAG) workflow for optical point-cloud analysis.

The primary notebook is:

```text
aventix_kgrag_real.ipynb
```

The notebook is designed to read an authorized local dataset supplied by the user. Sensitive source data is not included, described, enumerated, summarized, or reproduced in this README.

## Notebook capabilities

The notebook provides a complete analysis workflow for authorized point-cloud data:

- Dataset discovery and schema validation.
- Per-object or per-well point isolation.
- Multi-sensor fusion.
- Statistical outlier filtering.
- Empty-reference subtraction.
- Geometric volume estimation.
- Optional surface-tension and meniscus corrections.
- Deterministic verification rules.
- Measurement provenance.
- In-memory NetworkX knowledge graph construction.
- Grounded natural-language retrieval.
- Optional Groq-based answer synthesis.

The language model is restricted to retrieved graph facts. It does not independently inspect source files, invent measurements, or decide verification outcomes.

## Privacy and data handling

- Keep all source data local and access-controlled.
- Do not commit source files, generated measurement exports, credentials, or notebook outputs containing sensitive values.
- Review notebook outputs before sharing or exporting the notebook.
- Avoid publishing screenshots, printed tables, graph labels, paths, identifiers, or LLM evidence packs derived from private data.
- The original source data should be treated as read-only.

The notebook may generate derived tables or visualizations during execution. These outputs can contain sensitive information and must be handled with the same access restrictions as the source data.

## Running privately

Install dependencies in an isolated environment:

```bash
pip install -r requirements.txt
```

Open the notebook locally and run it only against an authorized local dataset. Do not use a shared or public notebook service for confidential data.

For optional LLM support, configure credentials through a local `.env` file or environment variables. Never place credentials directly in notebook cells or commit `.env` files.

```text
GROQ_API_KEY=your_private_key
MODEL=llama-3.3-70b-versatile
```

If no LLM credential is configured, deterministic graph retrieval remains available locally.

## FastAPI backend

The notebook pipeline is also available as a local service. It loads the bundled dataset at startup, computes the derived measurements in memory, and never modifies the source CSV files.

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Endpoints:

- `GET /health` — pipeline and graph counts
- `GET /measurements?plate=AB0796-B&status=FAIL` — filtered well measurements
- `GET /summary?plate=EP384-A` — plate-level summary
- `GET /graph?node_type=Measurement` — graph nodes and edges
- `POST /query` with `{"question":"Why was well B3 faulty on plate AB0796-B?","use_llm":false}` — grounded retrieval/Q&A

Set `AVENTIX_DATASET_DIR` to use another authorized copy of the same dataset layout. `GROQ_API_KEY` remains optional; without it, `/query` uses the deterministic synthesis from the notebook.

## Safe sharing checklist

Before sharing the project, remove or review:

- Executed notebook outputs.
- Derived CSV files.
- Graph exports.
- Visualization images.
- Logs and error traces.
- Local paths and identifiers.
- Environment files and API keys.
