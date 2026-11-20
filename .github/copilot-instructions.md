# RetroAI architecture

- `retroai/core/`: startup configuration, per-workbench state, Pydantic schemas, RDKit utilities, and identifier resolution.
- `retroai/ord/`: reaction fingerprints, SMARTS/ΔFG gates, ORD parsing, and DuckDB precedent indexing.
- `retroai/ai/`: Gemini structured output, step classification, and Crossref/Europe PMC citation verification.
- `retroai/engine/`: lazy AiZynthFinder setup, route search/scoring, and memory cleanup.
- `retroai/ui/`: Colab callbacks, Ketcher, and the HTML workbench. The notebook only installs dependencies, fetches models, and calls `launch_workbench()`.

Import `retroai.core.config` first in `retroai/__init__.py`. Set the thread limits before importing ONNX Runtime, and patch `InferenceSession` before importing AiZynthFinder. Keep engine initialization, database creation, and network calls out of module imports. Use `get_secret()` for credentials; only the UI should depend on Colab callbacks. Pass `WorkbenchState` and ORD route steps explicitly rather than reading another module's globals.

Run `python -m pip install -e '.[test]'` and `python -m pytest` from the repository root. Tests must run without Colab, credentials, or downloaded models. Keep data and database artifacts out of Git.
