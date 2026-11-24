# RetroAI architecture

- `retroai/core/`: startup configuration, per-workbench state, Pydantic schemas, RDKit utilities, and identifier resolution.
- `retroai/cbyd/`: optional CbyD route parsing, condition/yield normalization, hazard tagging, and release-aware DuckDB ingestion.
- `retroai/ord/`: reaction fingerprints, SMARTS/ΔFG gates, ORD parsing, and DuckDB precedent indexing.
- `retroai/ai/`: Gemini structured output, step classification, and Crossref/Europe PMC citation verification.
- `retroai/engine/`: lazy AiZynthFinder setup, route search/scoring, and memory cleanup.
- `retroai/ui/`: Colab callbacks, Ketcher, and the HTML workbench. The notebook only installs dependencies, fetches models, and calls `launch_workbench()`.

## Repository Structure & Landmarks

- `retroai/ord/fingerprints.py`: packed RDKit/DRFP reaction fingerprints, multiplicity-aware functional-group deltas, medicinal-chemistry family classification, progressive semantic/MCS SMARTS gates, and compact reaction-core extraction.
- `retroai/ord/parser.py`: offline ORD protobuf normalization, salt stripping, spectator classification, and precomputed fingerprint/ΔFG features.
- `retroai/ord/validator.py`: DuckDB schema migration/backfill, refreshable dual ORD/CbyD DRFP indexes, exact/chemical validation, chemoselective CbyD re-ranking, stereoselective lookup, online diagnostics, and Hugging Face protobuf/Parquet ingestion.
- `retroai/ord/index_sync.py`: import-safe manifest validation, checksum-verified atomic CbyD index download, and transactional CbyD-only installation into the shared ORD database.
- `retroai/ai/classifier.py`: offline RDKit family classification, concrete unvalidated condition/stoichiometry baselines, ORD/CbyD enrichment, Gemini citation sanitization, and explicit scenario-yield planning.
- `retroai/ui/workbench.py`: ORD/CbyD telemetry plus Colab callbacks, route rendering, and empirical/scenario-yield ELN scaling.
- `retroai/cbyd/advisor.py`: CbyD-backed stereochemical-control risk analysis and curated chiral-pool/auxiliary suggestions.
- `retroai/cbyd/parser.py`: CbyD step normalization, structured yield/condition parsing, spectator functional groups, and deterministic academic hazard flags.
- `retroai/cbyd/ingest.py`: offline route JSON filtering/deduplication, `cbyd_reactions` persistence, and schema/versioned CbyD release metadata.
- `.github/workflows/build-indexes.yml`: weekly/manual schema-aware CbyD release indexing and rolling `indexes-latest` asset publication.
- `tests/test_cbyd.py`: offline CbyD ingestion, provenance, chemoselective fallback, multi-step yield, and hazard coverage.
- `tests/test_cbyd_integration.py`: CbyD integration, SAScore/SPS key-step, loop, stock, and stereochemical-risk regression tests.
- `tests/benchmarks/benchmark_cbyd.py`: opt-in native CbyD route-tree recovery, convergence, Lock Selected, and stock-coverage benchmark.
- `retroai/core/chem_utils.py`: shared primary-molecule extraction, SAScore/SPS complexity metrics, stereochemical change detection, and chemistry analytics used by ORD, engine, and UI callers.
- `retroai/engine/pipeline.py`: route search/scoring, LLS complexity payloads, strategic key-step tagging, and unproductive-loop penalties.
- `tests/test_ord_validator.py`: offline ORD pipeline, migration, matching, persistence, and integration coverage; tests must not fetch live data.
- `tests/test_index_sync.py`: offline manifest, checksum, atomic download, CbyD/ORD merge, fallback, and import-safety coverage for prebuilt indexes.
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
- `retroai/ai/classifier.py` — `def _merge_step_stoichiometry()`, `def _baseline_condition_profile()`, `def _ranked_ord_conditions()`, `def _condition_items()`, `def _cbyd_precedents()`, `def _fg_search_terms()`, `def _step_transformation_keywords()`, `def _has_new_cyclic_protecting_group()`
- `retroai/ai/gemini_client.py` — `def _with_evidence_guardrails()`, `def get_gemini_client()`, `def _build_genai_config()`, `def call_gemini_with_config()`
- `retroai/ai/literature.py` — `def _clean_doi()`, `def _literature_snippet()`, `def cbyd_precedent_literature()`, `def _verify_doi_metadata()`, `def fetch_verified_literature_dois()`
- `retroai/cbyd/__init__.py`
- `retroai/cbyd/advisor.py` — `def _catalog_patterns()`, `def _catalog_matches()`, `def _has_chiral_control()`, `def advise_stereochemical_control()`
- `retroai/cbyd/ingest.py` — `def ensure_cbyd_schema()`, `def _collect_route_tree_flags()`, `def ingest_cbyd_routes()`, `def main()`
- `retroai/cbyd/parser.py` — `def _spectator_salt_keys()`, `def _as_list()`, `def _condition_text()`, `def parse_yield_fields()`, `def _standardize_reaction()`, `def assess_condition_hazards()`, `def parse_cbyd_step()`
- `retroai/core/__init__.py`
- `retroai/core/chem_utils.py` — `def _extract_primary_smiles()`, `def _canonical_molecule_smiles()`, `def _sascore_calculator()`, `def _spacial_score_calculator()`, `def _mol_from_smiles_or_mol()`, `def rdkit_calculate_sascore()`, `def rdkit_calculate_spacial_score()`, `def _reaction_side_mols()`
- `retroai/core/config.py` — `def get_secret()`, `def set_active_genai_settings()`, `def get_process_ram_mb()`
- `retroai/core/resolver.py` — `def _resolve_chemical_identifier()`
- `retroai/core/schemas.py` — `def _clamp_safety_score()`, `def _clamp_step_yield()`, `class RankedConditionSet`, `class ReagentStoichItem`, `class CbyDPrecedent`, `class StepClassificationItem`, `class StepClassificationBatch`, `class StrategySummary`
- `retroai/core/state.py` — `def _default_search_params()`, `class WorkbenchState`
- `retroai/engine/__init__.py`
- `retroai/engine/aizynth.py` — `def _common_lab_reagent_keys()`, `def is_common_lab_reagent()`, `def configure_finder_runtime_stock()`, `def get_finder()`, `def _harvest_uspto_templates_from_finder()`, `def _flatten_reaction_tree()`, `def cleanup_aizynth_memory()`, `def register_custom_lab_stock()`
- `retroai/engine/pipeline.py` — `def _longest_linear_step_indices()`, `def _organic_reaction_endpoints()`, `def detect_unproductive_loops()`, `def annotate_route_strategy()`, `def deduplicate_retro_state()`, `def compute_composite_route_score()`, `def run_retrosynthesis_pipeline()`
- `retroai/ord/__init__.py`
- `retroai/ord/fingerprints.py` — `def compute_folded_rxn_fp()`, `def compute_drfp_bytes()`, `def tanimoto_packed_fp_batch()`, `def compute_rxn_center_diff_fp()`, `def _load_smartsrx_data()`, `def _build_rdkit_internal_fg_catalog()`, `def _parse_reaction_mols()`, `def compute_rdkit_internal_delta_fg()`
- `retroai/ord/parser.py` — `def _ord_enum_name()`, `def _safe_compound_smiles()`, `def _canonical_component_smiles()`, `def _standardize_primary_smiles()`, `def _known_spectator_role()`, `def _smiles_to_english_name()`, `def _humanize_condition_list()`, `def _safe_compound_name()`
- `retroai/ord/validator.py` — `def _ord_gate_specs()`, `def _counter_from_json()`, `def _delta_cache_key()`, `def _json_value()`, `def _submit_and_poll_ord_session()`, `def _fetch_step_ord_records_worker()`, `def _parse_parquet_ord_row()`, `def _iter_hf_ord_parquet()`
- `retroai/ui/__init__.py`
- `retroai/ui/ketcher.py` — `class _QuietHandler`, `def _ensure_ketcher_bundle()`
- `retroai/ui/workbench.py` — `def _scale_route_eln()`, `def _eln_step_rows()`, `def _eln_step_text()`, `def _render_eln_step_html()`, `def _generate_strategy_header()`, `def _enrich_route_for_chemforge()`, `def _render_complexity_sparkline_html()`, `def _render_step_source()`
- `tests/benchmarks/benchmark_cbyd.py` — `def load_route_tree_jsonl()`, `def _walk()`, `def _record_label()`, `def _target_smiles()`, `def _molecule_smiles()`, `def _intermediate_smiles()`, `def _is_convergent()`, `def _selected_record()`
- `tests/conftest.py` — `def pytest_addoption()`, `def pytest_configure()`
- `tests/test_cbyd.py` — `def _route()`, `def test_yield_parser_preserves_telescoped_and_selectivity_fields()`, `def test_cbyd_standardization_strips_salt_and_tracks_mapped_survival()`, `def test_cbyd_ingestion_deduplicates_filters_and_records_metadata()`, `def test_ingestion_uses_route_tree_reachability_and_rejects_bad_jsonl()`, `def test_validator_rescues_step_with_cbyd_without_counting_telescoped_yield()`
- `tests/test_cbyd_integration.py` — `def _route()`, `def _step()`, `def test_route_tree_schema_and_cbyd_json_ingestion()`, `def test_stereoselective_yield_fields_and_chemoselective_reranking()`, `def test_validator_merges_strong_ord_and_cbyd_analog_sets()`, `def test_complexity_key_steps_and_loop_penalties()`, `def test_stereochemical_creation_is_detected_and_flagged()`, `def test_common_lab_reagent_allowlist_is_opt_in_and_canonical()`
- `tests/test_chem_utils.py` — `def test_extract_primary_smiles()`, `def test_descriptors_use_the_primary_product()`, `def test_byproducts_include_deduplicated_product_and_predicted_impurity()`, `def test_lcms_adducts_and_bromine_isotope_note()`, `def test_workup_properties_compare_product_to_reactants()`
- `tests/test_ord_validator.py` — `def _ord_record()`, `def _insert_ord_record()`, `def offline_fg_catalog()`, `def test_internal_delta_fg_reports_lost_alcohol_and_formed_aldehyde()`, `def test_drfp_and_vectorized_packed_tanimoto_are_exact()`, `def test_delta_fg_preserves_multiplicity_and_does_not_veto_carboxylates()`, `def test_aromatic_acid_amine_coupling_has_concrete_unvalidated_fallback()`, `def test_progressive_smarts_tries_exact_template_before_mcs()`
- `tests/test_runtime.py` — `def test_headless_imports_do_not_initialize_data_services()`, `def test_onnx_session_patch_enforces_memory_options()`, `def test_get_secret_uses_environment_without_colab()`, `def test_workbench_registers_the_callbacks_used_by_the_frontend()`, `def test_agent_reply_preserves_only_plain_formatting_tags()`
<!-- AUTO-MAP:END -->` within the README.md manually if the auto-mapper workflow is active; update the human-written architectural notes and "Do Not Touch" boundaries instead.
4. When a Pull Requested is merged, update the README.md file to briefly add a date-stamped description the new features were added, along with the issue link and PR link to reflect the cuurent state of the notebook. 
