# RetroAI

RetroAI combines AiZynthFinder route search, RDKit analytics, Open Reaction Database precedents, and optional Gemini enrichment.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tommbk50-hub/RetroAI/blob/main/Retrosynthesis_Agent.ipynb)

<img width="1103" height="625" alt="image" src="https://github.com/user-attachments/assets/6529d147-a8db-49fc-9b84-ecef22a24232" />


Run both notebook cells to install the package, download the USPTO models and ZINC stock, and launch the Ketcher workbench. Set `KDD_GEMINI_API_KEY` in Colab Secrets for Gemini features. For headless use, set it as an environment variable instead. `HF_TOKEN` is optional.

## How it works?
Retrosynthesis is planning backward: start with the molecule you want to make and look for simpler starting materials that could lead to it. RetroAI suggests several possible routes so you can compare them.
### Plan a route
1. **Choose a target.** Draw it in the Ketcher sketchpad, or enter a name, database ID, or SMILES (a text representation of a molecule) in the search box and click **Load**. You can also pick a built-in example.
2. **Run the search.** In **Synthesis**, click **Plan Synthesis Route**. The defaults are enough for a first run. In the search settings, you can limit the number of steps or how long the search runs. Use **Lock Selected** to keep only routes that contain your selected fragment in an intermediate or starting material, or **Register Lab Stock** to count materials you already have.
3. **Read the route cards.** **Solved** means all starting materials are listed in stock; **Partial** means some are missing. Open **Why might this fail?** to see flagged risks, and inspect individual steps for similar recorded reactions where available.
### How routes are found and scored
AiZynthFinder uses patterns learned from patent reactions to suggest ways to break the target into simpler pieces. It explores different branches in this way, looking for paths that lead back to starting materials listed in the ZINC stock catalog. RetroAI also checks the proposed routes against any lab stock you registered.
RetroAI looks for similar experiments in the **Open Reaction Database (ORD)**. It uses evidence such as reported yields and failures, when available, to score the support for each step and flag the least-supported step as the **bottleneck**. The ORD confidence score summarizes that evidence; it is not a probability that the route will work.
ORD and optional **Chemistry by Design (CbyD)** records are standardized and desalted before indexing. Validation uses a local DuckDB funnel: exact product-connectivity hits and DRFP candidates are shortlisted first, then chemically checked with reaction-center SMARTS and functional-group changes before empirical ranking. CbyD candidates are additionally re-ranked by spectator functional groups that survived the precedent reaction. Missing precedents are explicitly marked **Unvalidated** rather than receiving synthetic yield evidence; ELN scaling then requires an explicit scenario yield until empirical data is available.
The overall ranking also considers the search score, the number of steps, starting material availability, and estimated safety. Use **Rank By** to emphasize practical considerations such as stock and safety (**CRO-Practical**), shorter routes (**Discovery (Fast)**), stronger experimental evidence (**Evidence-Backed (ORD)**), or estimated safety and environmental impact (**Green / Safe**). **Re-classify & Re-score (ORD + AI)** refreshes the evidence and scores for existing routes without starting a new search. A high score is not a substitute for lab validation or safety review.
The **Strategy / Complexity-Building** profile adds RDKit SAScore/SPS jumps along the longest linear sequence, highlights strategic key steps, and penalizes reversible protection/deprotection or esterification/hydrolysis loops. Route cards include the resulting sparkline and step-level complexity payload.
Steps that create or invert tetrahedral or alkene stereochemistry are checked for chiral catalysts, ligands, pool-derived materials, or auxiliaries. Uncontrolled stereochemical changes appear in **Why might this fail?** with analogous stereoselective CbyD precedents and curated chiral-pool/Evans/Ellman/Myers options.
### Explore the other tabs
- **Reaction** checks a reaction you draw or enter against similar recorded experiments.
- **Analytical** predicts signals you might see in a mass spectrometry report. With Gemini enabled, it can also help interpret a pasted report.
- **Chemist AI Agent** and **Structure Q&A** use Gemini to discuss your routes, molecules, and supporting evidence. Route search itself does not require Gemini.

## Local development

From the repository root with Python 3.11 or newer:

```sh
python -m pip install -e '.[test]'
python -m pytest
```

Non-UI modules and tests do not require Colab, model data, or API keys. For route searches, use the notebook's separate AiZynthFinder and Reaction-Utils install and run `download_public_data data` from the directory in which you will run the search.

The notebook installs Reaction-Utils with the engine using `--no-deps` because its published metadata would otherwise downgrade `urllib3` to a vulnerable 1.x release. RetroAI declares the ONNX engine's required libraries itself. The package also permits protobuf 5.29.6 because published `ord-schema` wheels require protobuf below 6; pinning protobuf to 6.31.1 or newer would make the base install unsatisfiable.

### Optional CbyD precedent index

Download the CC BY 4.0 CbyD `v2026.11` release and ingest its route JSON files before
launching the workbench:

```sh
git clone --branch v2026.11 --depth 1 https://github.com/chemistry-by-design/cbyd.git data/cbyd
python -m retroai.cbyd.ingest data/cbyd/data/routes \
  --release v2026.11
```

The optional route-tree file can be generated in CbyD with `python build_route_tree.py`
and supplied with `--route-tree-jsonl data/cbyd/data/route_tree.jsonl` so RetroAI only
indexes reactions reachable in the generated tree. By default, RetroAI writes the
`cbyd_reactions` and `cbyd_metadata` tables beside `ord_reactions` in its local DuckDB
database. CbyD is wholly optional: absent tables degrade to the existing ORD-only flow.
If `data/cbyd/data/routes` is already present, notebook Cell 1 ingests release
`v2026.11` automatically into the same `data/ord_precedents.duckdb` database. The
workbench header reports ORD and CbyD counts independently.
Ingested DOI, citation, mapping-confidence, parsed yield, chemoselectivity, and hazard
metadata remain available to the Chemist AI Agent and Reaction tab.
The Synthesis settings enable a common lab-reagent allowlist by default for reagents that
commercial catalogs often omit (including MeI, halogens, peroxide, thionyl chloride,
water/ammonia, acetic anhydride, mesyl chloride, and trimethylsilyl chloride). It can be
disabled per search without changing registered in-house stock.

Run the offline CbyD ingestion/precedent benchmark smoke suite with:

```sh
python -m pytest -q tests/test_cbyd.py tests/test_cbyd_integration.py
```

Generate CbyD's native AiZynthFinder route trees with the upstream `build_route_tree.py`,
then opt in to the target-recovery, stock-coverage, convergence, and Lock Selected benchmark:

```sh
python -m pytest -m benchmark \
  --cbyd-trees data/cbyd/data/route_tree.jsonl \
  tests/benchmarks/benchmark_cbyd.py
```


## Live Repository Map
<!-- AUTO-MAP:START -->
<!-- AUTO-MAP:END -->

> The generated markers above are reserved. The human-readable map below is derived from
> `retroai/` and should be updated whenever a module, public seam, state contract, or ownership
> boundary changes.

### Runtime map

```text
Retrosynthesis_Agent.ipynb
`- retroai.ui.workbench.launch_workbench()
   |- initialize one WorkbenchState, the AiZynthFinder facade, and the ORD validator
   |- register the Colab callbacks used by the embedded HTML/JavaScript UI
   `- render the workbench

Synthesis callback
`- _Workbench._cb_run_synthesis()
   `- retroai.engine.pipeline.run_retrosynthesis_pipeline()
      |- engine.aizynth: get_finder(), harvest templates, flatten routes
      |- ord.validator: prefetch/index precedents and validate steps
      |- ai.classifier: classify steps with RDKit, then optional Gemini
      |- ord.validator.score_route(): aggregate route evidence
      `- workbench renderers: turn the compact result into HTML

Manual reaction callback
`- _Workbench._cb_analyze_reaction()
   `- ord prefetch -> ai.classifier.batch_classify_with_gemini()
      `- evidence rendering + RDKit analytical cards

CbyD ingestion (optional/offline)
`- python -m retroai.cbyd.ingest
   |- cbyd.parser: normalize CbyD steps, yields, conditions, and spectator groups
   |- cbyd.advisor: stereochemical risk and chiral-pool/auxiliary alternatives
   `- cbyd.ingest: deduplicate into cbyd_reactions and record the release version
```

Dependency direction is intentionally one way:

```text
ui -> engine + ai + ord + core
engine -> ai + ord + core
ai -> ord + core
cbyd -> ord + core
ord -> core
core -> third-party runtime/chemistry libraries
```

Do not invert these layers to make a small UI or engine change convenient. In particular,
chemistry shared by ORD and UI belongs in `core`, evidence admission belongs in `ord`, and
Colab callback behavior belongs in `ui`.

### Package file ownership

| Path | Owns | Main edit locations and seams |
| --- | --- | --- |
| `retroai/__init__.py` | Package bootstrap | Imports `core.config` first so thread limits and the ONNX `InferenceSession` patch are installed before engine imports. Keep this file side-effect-light. |
| `retroai/core/config.py` | Process setup, credentials, and active Gemini settings | `get_secret()`, `set_active_genai_settings()`, `GEMINI_MODEL_CATALOG`, and ONNX memory options. This module is the import-time configuration exception. |
| `retroai/core/state.py` | One workbench session's mutable state | `WorkbenchState`, `active_route_steps()`, search defaults, route results, caches, stock, and canvas state. Pass the instance explicitly; do not replace it with notebook or module globals. |
| `retroai/core/schemas.py` | Shared structured-output schemas and normalization | Pydantic models for condition rankings, stoichiometry, step classifications, and strategy summaries; clamps safety/yield at the schema boundary. |
| `retroai/core/chem_utils.py` | Reusable RDKit molecule and reaction analytics | Primary-molecule extraction, descriptors, SVG cards, byproduct guesses, LC-MS ions, workup predictions, lazy SAScore/SPS complexity jumps, and configuration-aware stereochemical comparison. Use here for chemistry logic needed by more than one upper layer. |
| `retroai/core/resolver.py` | Chemical identifier resolution | `_resolve_chemical_identifier()` handles direct structures plus explicit network lookups for external identifiers. |
| `retroai/engine/aizynth.py` | AiZynthFinder lifecycle, templates, stock, and cleanup | `get_finder()`, `_harvest_uspto_templates_from_finder()`, `_flatten_reaction_tree()`, `cleanup_aizynth_memory()`, and `register_custom_lab_stock()`. Finder loading remains lazy and requires `data/config.yml`. |
| `retroai/engine/pipeline.py` | End-to-end route search and route-level scoring | `run_retrosynthesis_pipeline()`, `compute_composite_route_score()`, and result compaction. This is the orchestration seam between search, classification, ORD evidence, stock, and state. |
| `retroai/ord/fingerprints.py` | Reaction fingerprints and chemical matching | Packed RDKit/DRFP fingerprints, functional-group deltas, medicinal-chemistry family classification, progressive semantic/MCS SMARTS tiers, gate decisions, and compact online-search core extraction. Add chemistry-admission rules here, not in the UI. |
| `retroai/ord/parser.py` | ORD protobuf-to-record normalization | `parse_ord_reaction_protobuf()` desalts and canonicalizes species, classifies spectators, extracts yield/conditions/provenance, normalizes stoichiometry, and precomputes indexed features. Its output is a DuckDB row contract. |
| `retroai/ord/validator.py` | ORD persistence, ingestion, retrieval, and route confidence | `ORDRetrosynthesisValidator`, database migration/backfill, refreshable dual ORD/CbyD in-memory indexes, online/protobuf/Parquet ingestion diagnostics, `validate_step()`, and `score_route()`. Construction may create `data/ord_precedents.duckdb`; imports must not. |
| `retroai/cbyd/parser.py` | CbyD normalization and condition parsing | CbyD step validation, canonical/desalted deduplication, structured yield/condition fields, spectator survival tags, and deterministic hazard flags. |
| `retroai/cbyd/ingest.py` | Optional CbyD DuckDB ingestion | CLI/function for route JSON ingestion, mapping-confidence/route-tree filtering, deduplication, `cbyd_reactions`, and release metadata. |
| `retroai/cbyd/advisor.py` | Stereochemical controls and CbyD alternatives | Flags uncontrolled creation/inversion, queries annotated stereoselective precedents, and exposes a curated chiral-pool/auxiliary catalog. |
| `retroai/ai/gemini_client.py` | Gemini client/configuration and fallback calls | `get_gemini_client()`, `_build_genai_config()`, and `call_gemini_with_config()`. Keep model selection and retry behavior here. |
| `retroai/ai/classifier.py` | Step-level RDKit/ORD/Gemini enrichment | `rdkit_classify_step()`, `batch_classify_with_gemini()`, `_classify_route_steps()`, concrete family-specific unvalidated baselines, condition ranking, stoichiometry transfer, evidence context, and the per-state enrichment cache. RDKit/ORD baseline must remain usable without Gemini. |
| `retroai/ai/literature.py` | DOI and literature verification | Crossref identity checks and Europe PMC/PubMed candidate discovery. Network access happens only when these functions are called. |
| `retroai/ui/ketcher.py` | Ketcher asset delivery | `_ensure_ketcher_bundle()` downloads/caches the pinned bundle, serves auxiliary assets on loopback, and inlines the main document assets. |
| `retroai/ui/workbench.py` | Colab UI, callbacks, HTML rendering, ELN scaling, and agent tools | `launch_workbench()`, `_Workbench`, ORD/CbyD telemetry, route/step renderers, callback names, empirical-or-explicit-scenario scaled stoichiometry, and the embedded workbench frontend. Changes to a callback must update both the HTML caller and `output.register_callback`. |
| `retroai/{ai,core,engine,ord,ui}/__init__.py` | Package markers | Currently intentionally empty. Do not use them for database, network, model, or Colab initialization. |

### Core data contracts

- Reaction strings use forward synthesis orientation: `reactants>>products`.
  AiZynthFinder supplies retrosynthetic reactions, so `engine/aizynth.py` reverses them before
  they enter ORD and AI classification.
- A route step carries `forward_smiles`, `step_index`, optional `precursor_steps`, and optional
  USPTO metadata (`retro_template`, `classification`, `template_code`,
  `mapped_forward_smiles`). `ai/classifier.py` uses that identity for caching and batches.
- `core/state.py::WorkbenchState.latest_retro_results` is the compact session result. Routes
  contain search/ORD scores, stock leaves, and step dictionaries; step dictionaries contain
  `ord_validation`, structured Gemini/RDKit classification, and planning fields.
- `WorkbenchState.active_route_steps()` is the explicit bridge from UI state to ORD shard
  filtering. Never make `ord` read UI state or notebook globals.
- `parse_ord_reaction_protobuf()` defines the persisted ORD record shape. A new persisted
  field also requires a corresponding DuckDB `ALTER TABLE`/migration, index ingestion, and
  validation or parser coverage.
- `cbyd.ingest` defines the separate `cbyd_reactions` row contract. Keep ingestion offline
  and optional; an absent CbyD table must never block validator construction or ORD search.
- Pydantic models in `core/schemas.py` define Gemini's JSON contract. If a step field is added,
  update the schema first, then classification/rendering consumers. `scenario_step_yield` is
  planning-only and must remain separate from `assumed_step_yield`, which represents empirical
  evidence.
- `_render_routes_html_from_results()` renders SVG and HTML on demand. Store compact JSON-like
  route data in state; do not reintroduce reaction trees, RDKit objects, or large SVG blobs.

### Where to make common changes

| Change | Start here | Follow-through |
| --- | --- | --- |
| Add or change a search constraint, route score, or stock rule | `retroai/engine/pipeline.py` | Mirror UI parameter collection in `_cb_run_synthesis()` and any agent rerun tool; cover pipeline behavior in `tests/test_ord_validator.py`. |
| Change AiZynthFinder loading, template harvesting, or memory cleanup | `retroai/engine/aizynth.py` | Preserve lazy imports and `cleanup_aizynth_memory()` as the single tree-release path. |
| Change ORD matching or false-positive admission | `retroai/ord/fingerprints.py` and `ORDRetrosynthesisValidator._validate_step()` | Add exact, progressive-SMARTS, and reject-counterexample tests in `tests/test_ord_validator.py`. |
| Change ORD protobuf normalization or indexed fields | `retroai/ord/parser.py` | Update validator schema migration/backfill, bulk ingestion, and parser/migration tests together. |
| Change CbyD route normalization or release ingestion | `retroai/cbyd/parser.py` and `retroai/cbyd/ingest.py` | Preserve route-tree flag filtering, canonical deduplication, release metadata, and validator compatibility. |
| Change complexity key-step or stereochemical advice | `retroai/core/chem_utils.py`, `retroai/engine/pipeline.py`, and `retroai/cbyd/advisor.py` | Keep the LLS payload JSON-like, preserve lazy RDKit imports, route stereochemical decisions through the evidence gate, and render risks in the existing failure panel. |
| Add an AI-enriched step field or citation rule | `retroai/core/schemas.py` and `retroai/ai/classifier.py` | Revalidate the returned object, enforce evidence allowlists, then consume it in `retroai/ui/workbench.py`. |
| Add chemistry analytics used across tabs | `retroai/core/chem_utils.py` | Call from AI/UI rather than duplicating RDKit logic; extend `tests/test_chem_utils.py`. |
| Add or rename a frontend action | `retroai/ui/workbench.py` | Keep the JavaScript `invokeFunction(...)` name aligned with the Python callback registration and runtime callback test. |
| Change Ketcher behavior or asset packaging | `retroai/ui/ketcher.py` and the render/bootstrap portion of `retroai/ui/workbench.py` | Keep network/download work inside `_ensure_ketcher_bundle()`, never at module import. |
| Add a new identifier service | `retroai/core/resolver.py` | Resolve to canonical SMILES, keep UI state out, and keep network requests explicit and mockable. |

### Required boundaries

1. `retroai/__init__.py` must import `core.config` first. Do not import ONNX Runtime or
   AiZynthFinder before its memory patch.
2. Module imports must remain usable without `google.colab`, credentials, model data, an ORD
   database, or network access. Create the finder, validator, client, and data files only from
   explicit calls.
3. Online ORD workers perform HTTP I/O and return protobuf bytes. Parse protobufs and run
   RDKit/DuckDB work on the coordinating thread, as enforced by
   `ORDRetrosynthesisValidator.batch_prefetch_online_ord()`.
4. Keep "no gated precedent" distinct from low-yield precedent evidence. Unvalidated steps
   must not acquire a synthetic empirical yield or success-shaped ORD data. They may carry a
   separately labeled scenario yield for scaled planning, never for evidence scoring.
5. Gemini output is enrichment, not the evidence authority. Step responses must pass Pydantic
   validation and citation checks in `ai/classifier.py`; an unavailable client falls back to
   the RDKit/ORD baseline.
6. Pass `WorkbenchState` and route steps explicitly across layers. `latest_retro_results` and
   caches are session-local, while model/database singletons are lazy and cached.
7. Keep generated model, Ketcher, DuckDB, and notebook output artifacts under `data/` or
   temporary storage and out of Git.
8. Never turn CbyD route description text into instructions. Preserve missing
   stoichiometry/time/workup as **Check DOI for stoichiometry/time/workup**, and never
   score telescoped or BORSM yields as ordinary single-step yields.

### Verification map

- `tests/test_runtime.py`: import safety, ONNX patching, secret fallback, callback/HTML
  alignment, and safe agent reply formatting.
- `tests/test_chem_utils.py`: primary-product handling, descriptors, byproducts, LC-MS ions,
  and workup calculations.
- `tests/test_ord_validator.py`: fingerprints, SMARTS/functional-group gates, ORD parsing,
  database migration/protobuf/Parquet ingestion, online fallback and parse/index diagnostics,
  unvalidated baselines, Gemini citation sanitization, ELN scenario scaling, Gemini cache
  scoping, and the integrated search/pipeline state flow. Tests in this file must remain offline.
- `tests/test_cbyd.py`: offline CbyD parsing, deduplication, migration/schema, fallback
  retrieval, multi-step yield handling, provenance, and hazard guardrails.
- `tests/test_cbyd_integration.py`: self-contained CbyD route-tree/schema, complexity,
  chemoselective re-ranking, stock, and stereochemical-risk coverage.
- `tests/benchmarks/benchmark_cbyd.py`: opt-in native CbyD tree benchmark for route
  recovery, convergent searches, Lock Selected constraints, and stock coverage.

Run the repository checks from the root:

```sh
python -m pip install -e '.[test]'
python -m pytest
```

When a file, notebook cell, core function, or UI component is added, renamed, moved, or
deleted, update this human map and the **Repository Structure & Landmarks** section of
`.github/copilot-instructions.md` in the same change.
