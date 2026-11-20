# RetroAI architecture

- `retroai/core/`: startup configuration, per-workbench state, Pydantic schemas, RDKit utilities, and identifier resolution.
- `retroai/ord/`: reaction fingerprints, SMARTS/ΔFG gates, ORD parsing, and DuckDB precedent indexing.
- `retroai/ai/`: Gemini structured output, step classification, and Crossref/Europe PMC citation verification.
- `retroai/engine/`: lazy AiZynthFinder setup, route search/scoring, and memory cleanup.
- `retroai/ui/`: Colab callbacks, Ketcher, and the HTML workbench. The notebook only installs dependencies, fetches models, and calls `launch_workbench()`.

## Repository Structure & Landmarks

- `retroai/ord/fingerprints.py`: packed RDKit/DRFP reaction fingerprints, multiplicity-aware functional-group deltas, progressive SMARTS gates, and reaction-core/scaffold extraction.
- `retroai/ord/parser.py`: offline ORD protobuf normalization, salt stripping, spectator classification, and precomputed fingerprint/ΔFG features.
- `retroai/ord/validator.py`: DuckDB schema migration/backfill, in-memory DRFP index, exact/DRFP/chemical validation funnel, online ORD retrieval, and Hugging Face protobuf-shard ingestion.
- `retroai/core/chem_utils.py`: shared primary-molecule extraction and chemistry analytics used by ORD and UI callers.
- `tests/test_ord_validator.py`: offline ORD pipeline, migration, matching, persistence, and integration coverage; tests must not fetch live data.
- `pyproject.toml`: canonical runtime and test dependencies; `requirements.txt` installs the editable project and repeats the explicit DRFP pin.
- `Retrosynthesis_Agent.ipynb`: the Colab dependency/model setup and `launch_workbench()` entry points only.

Import `retroai.core.config` first in `retroai/__init__.py`. Set the thread limits before importing ONNX Runtime, and patch `InferenceSession` before importing AiZynthFinder. Keep engine initialization, database creation, and network calls out of module imports. Use `get_secret()` for credentials; only the UI should depend on Colab callbacks. Pass `WorkbenchState` and ORD route steps explicitly rather than reading another module's globals.

Run `python -m pip install -e '.[test]'` and `python -m pytest` from the repository root. Tests must run without Colab, credentials, or downloaded models. Keep data and database artifacts out of Git.

## Mandatory Maintenance Rule
Before finishing any task or opening a PR that adds, renames, moves, or deletes a file, notebook cell (`# @title`), core function, or UI component:
1. Update the **Repository Structure & Landmarks** section of `.github/copilot-instructions.md` to reflect the exact current state.
2. Remove references to deleted features or renamed functions so future agents do not waste tokens searching for them.
3. Do **not** edit content inside `<!-- AUTO-MAP:START -->` and `<!-- AUTO-MAP:END -->` within the README.md manually if the auto-mapper workflow is active; update the human-written architectural notes and "Do Not Touch" boundaries instead.
4. When a Pull Requested is merged, update the README.md file to briefly add a date-stamped description the new features were added, along with the issue link and PR link to reflect the cuurent state of the notebook. 
