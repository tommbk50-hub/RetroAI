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
3. Do **not** edit content inside `<!-- AUTO-MAP:START -->
- `Retrosynthesis_Agent.ipynb`
  - Cell 1: `1. Install RetroAI and download models`
  - Cell 2: `2. Launch the workbench`
- `retroai/__init__.py`
- `retroai/ai/__init__.py`
- `retroai/ai/classifier.py` — `def _merge_step_stoichiometry()`, `def _ranked_ord_conditions()`, `def _fg_search_terms()`, `def _step_transformation_keywords()`, `def rdkit_classify_step()`, `def _match_rdkit_rule()`, `def analyze_step_heuristics()`, `def _step_rdkit_analytics()`
- `retroai/ai/gemini_client.py` — `def get_gemini_client()`, `def _build_genai_config()`, `def call_gemini_with_config()`
- `retroai/ai/literature.py` — `def _clean_doi()`, `def _literature_snippet()`, `def _verify_doi_metadata()`, `def fetch_verified_literature_dois()`
- `retroai/core/__init__.py`
- `retroai/core/chem_utils.py` — `def _extract_primary_smiles()`, `def _canonical_molecule_smiles()`, `def _compute_descriptors_and_ro5()`, `def mol_or_rxn_to_svg_html()`, `def rdkit_enumerate_byproducts()`, `def rdkit_predict_lcms_ions()`, `def rdkit_predict_workup_properties()`
- `retroai/core/config.py` — `def get_secret()`, `def set_active_genai_settings()`, `def get_process_ram_mb()`
- `retroai/core/resolver.py` — `def _resolve_chemical_identifier()`
- `retroai/core/schemas.py` — `def _clamp_safety_score()`, `def _clamp_step_yield()`, `class RankedConditionSet`, `class ReagentStoichItem`, `class StepClassificationItem`, `class StepClassificationBatch`, `class StrategySummary`
- `retroai/core/state.py` — `def _default_search_params()`, `class WorkbenchState`
- `retroai/engine/__init__.py`
- `retroai/engine/aizynth.py` — `def get_finder()`, `def _harvest_uspto_templates_from_finder()`, `def _flatten_reaction_tree()`, `def cleanup_aizynth_memory()`, `def register_custom_lab_stock()`
- `retroai/engine/pipeline.py` — `def deduplicate_retro_state()`, `def compute_composite_route_score()`, `def run_retrosynthesis_pipeline()`
- `retroai/ord/__init__.py`
- `retroai/ord/fingerprints.py` — `def compute_folded_rxn_fp()`, `def compute_drfp_bytes()`, `def tanimoto_packed_fp_batch()`, `def compute_rxn_center_diff_fp()`, `def _load_smartsrx_data()`, `def _build_rdkit_internal_fg_catalog()`, `def _parse_reaction_mols()`, `def compute_rdkit_internal_delta_fg()`
- `retroai/ord/parser.py` — `def _ord_enum_name()`, `def _safe_compound_smiles()`, `def _canonical_component_smiles()`, `def _standardize_primary_smiles()`, `def _known_spectator_role()`, `def _smiles_to_english_name()`, `def _humanize_condition_list()`, `def _safe_compound_name()`
- `retroai/ord/validator.py` — `def _ord_gate_specs()`, `def _counter_from_json()`, `def _delta_cache_key()`, `def _submit_and_poll_ord_session()`, `def _fetch_step_ord_records_worker()`, `class ORDRetrosynthesisValidator`, `def get_ord_validator()`
- `retroai/ui/__init__.py`
- `retroai/ui/ketcher.py` — `class _QuietHandler`, `def _ensure_ketcher_bundle()`
- `retroai/ui/workbench.py` — `def _scale_route_eln()`, `def _eln_step_rows()`, `def _eln_step_text()`, `def _render_eln_step_html()`, `def _generate_strategy_header()`, `def _enrich_route_for_chemforge()`, `def _render_step_source()`, `def _render_ord_step_panel_html()`
- `tests/test_chem_utils.py` — `def test_extract_primary_smiles()`, `def test_descriptors_use_the_primary_product()`, `def test_byproducts_include_deduplicated_product_and_predicted_impurity()`, `def test_lcms_adducts_and_bromine_isotope_note()`, `def test_workup_properties_compare_product_to_reactants()`
- `tests/test_ord_validator.py` — `def _ord_record()`, `def _insert_ord_record()`, `def offline_fg_catalog()`, `def test_internal_delta_fg_reports_lost_alcohol_and_formed_aldehyde()`, `def test_drfp_and_vectorized_packed_tanimoto_are_exact()`, `def test_delta_fg_preserves_multiplicity_and_does_not_veto_carboxylates()`, `def test_progressive_smarts_tries_exact_template_before_mcs()`, `def test_ord_gate_prefers_smarts_then_delta_and_rejects_competing_electrophiles()`
- `tests/test_runtime.py` — `def test_headless_imports_do_not_initialize_data_services()`, `def test_onnx_session_patch_enforces_memory_options()`, `def test_get_secret_uses_environment_without_colab()`, `def test_workbench_registers_the_callbacks_used_by_the_frontend()`, `def test_agent_reply_preserves_only_plain_formatting_tags()`
<!-- AUTO-MAP:END -->` within the README.md manually if the auto-mapper workflow is active; update the human-written architectural notes and "Do Not Touch" boundaries instead.
4. When a Pull Requested is merged, update the README.md file to briefly add a date-stamped description the new features were added, along with the issue link and PR link to reflect the cuurent state of the notebook. 
