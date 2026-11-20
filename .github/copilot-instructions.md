# RetroAI architecture

- `retroai/core/`: startup configuration, per-workbench state, Pydantic schemas, RDKit utilities, and identifier resolution.
- `retroai/ord/`: reaction fingerprints, SMARTS/ΔFG gates, ORD parsing, and DuckDB precedent indexing.
- `retroai/ai/`: Gemini structured output, step classification, and Crossref/Europe PMC citation verification.
- `retroai/engine/`: lazy AiZynthFinder setup, route search/scoring, and memory cleanup.
- `retroai/ui/`: Colab callbacks, Ketcher, and the HTML workbench. The notebook only installs dependencies, fetches models, and calls `launch_workbench()`.

Import `retroai.core.config` first in `retroai/__init__.py`. Set the thread limits before importing ONNX Runtime, and patch `InferenceSession` before importing AiZynthFinder. Keep engine initialization, database creation, and network calls out of module imports. Use `get_secret()` for credentials; only the UI should depend on Colab callbacks. Pass `WorkbenchState` and ORD route steps explicitly rather than reading another module's globals.

Run `python -m pip install -e '.[test]'` and `python -m pytest` from the repository root. Tests must run without Colab, credentials, or downloaded models. Keep data and database artifacts out of Git.

## Mandatory Maintenance Rule
Before finishing any task or opening a PR that adds, renames, moves, or deletes a file, notebook cell (`# @title`), core function, or UI component:
1. Update the **Repository Structure & Landmarks** section of `.github/copilot-instructions.md` to reflect the exact current state.
2. Remove references to deleted features or renamed functions so future agents do not waste tokens searching for them.
3. Do **not** edit content inside `<!-- AUTO-MAP:START -->` and `<!-- AUTO-MAP:END -->` manually if the auto-mapper workflow is active; update the human-written architectural notes and "Do Not Touch" boundaries instead.
4. When a Pull Requested is merged, update the README.md file to briefly add a date-stamped description the new features were added, along with the issue link and PR link to reflect the cuurent state of the notebook. 
