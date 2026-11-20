import gzip
import json
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest
from rdkit import Chem

from retroai.ai import classifier
from retroai.core.state import WorkbenchState
from retroai.engine import aizynth, pipeline
from retroai.ord import fingerprints, parser as ord_parser, validator as ord_validator
from ord_schema.proto import dataset_pb2, reaction_pb2


OXIDATION = "CCO>>CC=O"
RETRO_TEMPLATE = "[C:1]=[O:2]>>[C:1][O:2]"


def _ord_record(ord_id, reaction_smiles, yield_pct):
    reaction = reaction_pb2.Reaction(reaction_id=ord_id)
    reactant_smi, product_smi = reaction_smiles.split(">>")
    for smi in reactant_smi.split("."):
        compound = reaction.inputs["starting material"].components.add(reaction_role="REACTANT")
        compound.identifiers.add(type=reaction_pb2.CompoundIdentifier.SMILES, value=smi)
    product = reaction.outcomes.add().products.add(is_desired_product=True)
    product.identifiers.add(type=reaction_pb2.CompoundIdentifier.SMILES, value=product_smi)
    product.measurements.add(type=reaction_pb2.ProductMeasurement.YIELD).percentage.value = yield_pct
    return ord_parser.parse_ord_reaction_protobuf(reaction)


def _insert_ord_record(validator, record):
    with duckdb.connect(validator.db_path) as con:
        columns = ",".join(record)
        placeholders = ",".join("?" for _ in record)
        con.execute(
            f"INSERT OR REPLACE INTO ord_reactions ({columns}) VALUES ({placeholders})",
            list(record.values()),
        )
    validator._refresh_index()


@pytest.fixture(autouse=True)
def offline_fg_catalog(monkeypatch):
    """Keep the internal RDKit catalog active without fetching optional SMARTSRX rules."""
    catalog = fingerprints._build_rdkit_internal_fg_catalog
    catalog.cache_clear()
    monkeypatch.setattr(fingerprints, "_load_smartsrx_data", lambda: [])
    yield
    catalog.cache_clear()


def test_internal_delta_fg_reports_lost_alcohol_and_formed_aldehyde():
    consumed, formed, reactant_groups = fingerprints.compute_rdkit_internal_delta_fg(OXIDATION)
    assert "RDKitFG_Alcohol" in consumed
    assert "RDKitFG_Aldehyde" in formed
    assert consumed <= reactant_groups
    assert fingerprints.compute_rdkit_internal_delta_fg("invalid>>invalid") == (
        frozenset(), frozenset(), frozenset()
    )


def test_drfp_and_vectorized_packed_tanimoto_are_exact(monkeypatch):
    packed = fingerprints.compute_drfp_bytes("[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]")
    assert len(packed) == 256
    assert packed == fingerprints.compute_drfp_bytes(OXIDATION)

    matrix = np.asarray(
        [
            [0b10110000, 0b00000001],
            [0b10000000, 0],
            [0, 0],
        ],
        dtype=np.uint8,
    )
    query = bytes([0b10010000, 0b00000001])
    similarities = fingerprints.tanimoto_packed_fp_batch(query, matrix)
    expected = [0.75, 1.0 / 3.0, 0.0]
    assert np.allclose(similarities, expected, atol=1e-6)

    monkeypatch.setattr(fingerprints, "DrfpEncoder", None)
    assert fingerprints.compute_drfp_bytes(OXIDATION) is None


def test_delta_fg_preserves_multiplicity_and_does_not_veto_carboxylates():
    rxn = "NCCN.CC(=O)Cl>>NCCNC(=O)C"
    consumed, _, starting = fingerprints.compute_rdkit_internal_delta_fg(rxn)
    assert starting["RDKitFG_Amine.Primary"] == 2
    assert consumed["RDKitFG_Amine.Primary"] == 1
    assert "fr_COO" not in fingerprints._COMPETING_ELECTROPHILES


def test_progressive_smarts_tries_exact_template_before_mcs():
    tiers = fingerprints.build_progressive_reaction_smarts(OXIDATION, RETRO_TEMPLATE)
    assert [tier["tier"] for tier in tiers] == [1, 2, 3]
    assert tiers[0]["smarts"] == "[C:1][O:2]>>[C:1]=[O:2]"
    assert all(tier["queries"] for tier in tiers)


def test_ord_gate_prefers_smarts_then_delta_and_rejects_competing_electrophiles():
    tiers = fingerprints.build_progressive_reaction_smarts(OXIDATION, RETRO_TEMPLATE)
    delta = fingerprints.compute_rdkit_internal_delta_fg(OXIDATION)
    reactants, products = fingerprints._parse_reaction_mols(OXIDATION)
    assert fingerprints._match_ord_gate(reactants, products, delta, tiers, delta) == (
        tiers[0]["label"], tiers[0]["base_sim"]
    )
    assert fingerprints._match_ord_gate(reactants, products, delta, [], delta) == ("RDKit ΔFG match", 0.58)

    competing = "CC(=O)Cl.CCO>>CC=O"
    reactants, products = fingerprints._parse_reaction_mols(competing)
    competing_delta = fingerprints.compute_rdkit_internal_delta_fg(competing)
    assert fingerprints._match_ord_gate(reactants, products, competing_delta, tiers, delta) is None


def test_relaxed_mcs_and_achiral_fallback_match_substituted_and_flat_analogs():
    exact_queries = fingerprints._compile_rxn_smarts_queries("[C@H](O)C>>[C@H](N)C")
    flat_r, flat_p = fingerprints._parse_reaction_mols("CC(O)C>>CC(N)C")
    assert not fingerprints.matches_reaction_smarts_mols(flat_r, flat_p, exact_queries)
    assert fingerprints.matches_reaction_smarts_mols(
        flat_r, flat_p, exact_queries, use_chirality=False
    )
    chiral_delta = fingerprints.compute_rdkit_internal_delta_fg("C[C@H](O)C>>C[C@H](N)C")
    flat_delta = fingerprints.compute_rdkit_internal_delta_fg("CC(O)C>>CC(N)C")
    gate = fingerprints._match_ord_gate(
        flat_r,
        flat_p,
        flat_delta,
        [{"label": "chiral template", "base_sim": 0.80, "queries": exact_queries}],
        chiral_delta,
    )
    assert gate == ("chiral template (achiral)", 0.78)

    query = "B(O)c1ccccc1.Brc1ccccc1>>c1ccc(-c2ccccc2)cc1"
    analog = "B(O)c1ccc(C)cc1.Brc1ccccc1>>Cc1ccc(-c2ccccc2)cc1"
    tiers = fingerprints.build_progressive_reaction_smarts(query)
    analog_parts = fingerprints._parse_reaction_mols(analog)
    gate = fingerprints._match_ord_gate(
        *analog_parts,
        fingerprints.compute_rdkit_internal_delta_fg(analog),
        tiers,
        fingerprints.compute_rdkit_internal_delta_fg(query),
    )
    assert gate[0].startswith("Tier 2: MCS Reaction Center")


def test_core_extractor_uses_largest_reactant_and_reduced_product_scaffold():
    rxn = "CCOC(=O)c1ccc(Br)cc1.B(O)c1ccccc1>>CCOC(=O)c1ccc(-c2ccccc2)cc1"
    cores = fingerprints._rdkit_extract_reacting_cores(rxn)
    assert cores["core_reactant"] == "CCOC(=O)c1ccc(Br)cc1"
    product = Chem.MolFromSmiles(cores["core_product"])
    scaffold = Chem.MolFromSmiles(cores["core_product_scaffold"])
    assert scaffold.GetNumHeavyAtoms() >= 4
    assert scaffold.GetNumHeavyAtoms() < product.GetNumHeavyAtoms()


def test_parser_desalts_spectators_and_precomputes_features(monkeypatch):
    reaction = reaction_pb2.Reaction(reaction_id="ord-parser-test")
    for smi in ("NCCN.Cl", "CC(=O)Cl"):
        compound = reaction.inputs["starting material"].components.add(reaction_role="REACTANT")
        compound.identifiers.add(type=reaction_pb2.CompoundIdentifier.SMILES, value=smi)
    solvent = reaction.inputs["starting material"].components.add(reaction_role="UNSPECIFIED")
    solvent.identifiers.add(type=reaction_pb2.CompoundIdentifier.SMILES, value="O")

    outcome = reaction.outcomes.add()
    desired = outcome.products.add()
    desired.identifiers.add(
        type=reaction_pb2.CompoundIdentifier.SMILES, value="NCCNC(=O)C.Cl"
    )
    desired.measurements.add(
        type=reaction_pb2.ProductMeasurement.YIELD
    ).percentage.value = 75
    byproduct = outcome.products.add()
    byproduct.identifiers.add(type=reaction_pb2.CompoundIdentifier.SMILES, value="Cl")

    record = ord_parser.parse_ord_reaction_protobuf(reaction)
    assert record["rxn_smiles"] == "NCCN.CC(=O)Cl>>CC(=O)NCCN"
    assert record["solvents"] == "water"
    assert len(record["drfp_bytes"]) == 256
    assert record["diff_fp_bytes"]
    assert "RDKitFG_Amine.Primary" in json.loads(record["consumed_fgs"])
    assert "fr_amide" in json.loads(record["formed_fgs"])

    monkeypatch.setattr(
        ord_parser.requests,
        "get",
        lambda *args, **kwargs: pytest.fail("offline condition naming must not use the network"),
    )
    ord_parser._smiles_to_english_name.cache_clear()
    assert ord_parser._smiles_to_english_name("CC(C)(C)Br") == "CC(C)(C)Br"
    assert ord_parser._humanize_condition_list("CC(C)(C)Br") == "CC(C)(C)Br"


def test_validator_indexes_a_local_ord_shard_and_filters_precedents(tmp_path, monkeypatch):
    dataset = dataset_pb2.Dataset()
    for reactant, ord_id, yield_pct in [("CCO", "ord-test-alcohol", 72), ("CC#N", "ord-test-nitrile", 20)]:
        reaction = dataset.reactions.add(reaction_id=ord_id)
        compound = reaction.inputs["starting material"].components.add(reaction_role="REACTANT")
        compound.identifiers.add(type=reaction_pb2.CompoundIdentifier.SMILES, value=reactant)
        product = reaction.outcomes.add().products.add(is_desired_product=True)
        product.identifiers.add(type=reaction_pb2.CompoundIdentifier.SMILES, value="CC=O")
        product.measurements.add(type=reaction_pb2.ProductMeasurement.YIELD).percentage.value = yield_pct
    shard = tmp_path / "synthetic.pb.gz"
    shard.write_bytes(gzip.compress(dataset.SerializeToString()))
    monkeypatch.setattr(
        ord_validator, "list_repo_files", lambda **kwargs: ["ignored.parquet", shard.name]
    )

    def download_shard(**kwargs):
        assert kwargs["filename"].endswith((".pb", ".pb.gz"))
        return shard

    monkeypatch.setattr(ord_validator, "hf_hub_download", download_shard)

    db_path = tmp_path / "ord.duckdb"
    validator = ord_validator.ORDRetrosynthesisValidator(db_path)
    assert db_path.is_file()
    assert validator.get_index_size() == 0
    step = {"forward_smiles": OXIDATION, "retro_template": RETRO_TEMPLATE}
    assert validator.validate_step(OXIDATION, template_info=step)["smarts_matched_count"] == 0
    revision = validator.index_revision

    assert validator.sync_huggingface_parquet_shards(route_steps=[step])["indexed_count"] == 1
    assert validator.index_revision == revision + 1
    assert validator.sync_huggingface_parquet_shards(route_steps=[step])["added"] == 0
    assert validator.sync_huggingface_parquet_shards()["indexed_count"] == 2

    report = validator.validate_step(OXIDATION, template_info=step)
    assert report["exact_product_hits"] == 1
    assert report["smarts_matched_count"] == 1
    assert report["matched_smarts_label"] == "Tier 1: Exact USPTO Template SMARTS"
    assert [hit["ord_id"] for hit in report["top_precedents"]] == ["ord-test-alcohol"]
    assert validator.score_route([OXIDATION], step_reports=[report])["ord_mean_yield_pct"] == 72.0


def test_legacy_database_migrates_and_backfills_ord_features(tmp_path):
    db_path = tmp_path / "legacy.duckdb"
    record = _ord_record("legacy-oxidation", OXIDATION, 68)
    with duckdb.connect(str(db_path)) as con:
        con.execute("""
            CREATE TABLE ord_reactions (
                ord_id VARCHAR PRIMARY KEY,
                rxn_smiles VARCHAR,
                prod_inchikey VARCHAR,
                prod_connectivity VARCHAR,
                yield_pct DOUBLE,
                catalysts VARCHAR,
                reagents VARCHAR,
                solvents VARCHAR,
                temp_c DOUBLE,
                atmosphere VARCHAR,
                undesired_byproducts VARCHAR,
                reference VARCHAR,
                fp_bytes BLOB
            )
        """)
        base_columns = [
            "ord_id", "rxn_smiles", "prod_inchikey", "prod_connectivity", "yield_pct",
            "catalysts", "reagents", "solvents", "temp_c", "atmosphere",
            "undesired_byproducts", "reference", "fp_bytes",
        ]
        con.execute(
            f"INSERT INTO ord_reactions ({','.join(base_columns)}) "
            f"VALUES ({','.join('?' for _ in base_columns)})",
            [record[column] for column in base_columns],
        )

    validator = ord_validator.ORDRetrosynthesisValidator(db_path)
    with duckdb.connect(str(db_path), read_only=True) as con:
        drfp_bytes, diff_fp_bytes, consumed_fgs, formed_fgs = con.execute(
            "SELECT drfp_bytes, diff_fp_bytes, consumed_fgs, formed_fgs FROM ord_reactions"
        ).fetchone()
    assert len(drfp_bytes) == 256
    assert diff_fp_bytes
    assert "RDKitFG_Alcohol" in json.loads(consumed_fgs)
    assert "RDKitFG_Aldehyde" in json.loads(formed_fgs)
    assert validator._drfp_matrix.shape == (1, 256)


def test_funnel_rejects_same_product_transformation_and_unrelated_reactions(tmp_path):
    validator = ord_validator.ORDRetrosynthesisValidator(tmp_path / "selective.duckdb")
    query = "NCCN.CC(=O)Cl>>NCCNC(=O)C"
    records = [
        _ord_record("amide-coupling", query, 82),
        _ord_record(
            "boc-deprotection",
            "CC(C)(C)OC(=O)NCCNC(=O)C>>NCCNC(=O)C",
            95,
        ),
        _ord_record(
            "suzuki-coupling",
            "B(O)c1ccccc1.Brc1ccccc1>>c1ccc(-c2ccccc2)cc1",
            90,
        ),
    ]
    for record in records:
        _insert_ord_record(validator, record)

    report = validator.validate_step(query, top_k=5)
    assert report["exact_product_hits"] == 1
    assert report["smarts_matched_count"] == 1
    assert [record["ord_id"] for record in report["top_precedents"]] == ["amide-coupling"]


def test_unvalidated_confidence_is_lower_than_documented_negative_data(tmp_path):
    validator = ord_validator.ORDRetrosynthesisValidator(tmp_path / "confidence.duckdb")
    unvalidated = validator.validate_step(OXIDATION, template_info={
        "retro_template": RETRO_TEMPLATE,
    })
    assert unvalidated["is_validated"] is False
    assert unvalidated["matched_smarts_label"] == "No verified match (Unvalidated)"
    assert unvalidated["step_ord_confidence"] == 0.20

    _insert_ord_record(
        validator, _ord_record("negative-hte", OXIDATION, 0)
    )
    validated = validator.validate_step(
        OXIDATION, template_info={"retro_template": RETRO_TEMPLATE}
    )
    assert validated["is_validated"] is True
    assert validated["hte_failure_rate_pct"] == 100.0
    assert validated["step_ord_confidence"] > unvalidated["step_ord_confidence"]


def test_online_ord_failures_are_retried_only_after_incomplete_calls(tmp_path, monkeypatch):
    validator = ord_validator.ORDRetrosynthesisValidator(tmp_path / "retry.duckdb")
    monkeypatch.setattr(
        validator, "_auto_sync_hf_ord_if_sparse", lambda *args, **kwargs: 0
    )
    calls = 0

    def worker(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("transient timeout")
        return []

    monkeypatch.setattr(ord_validator, "_fetch_step_ord_records_worker", worker)
    first = validator.batch_prefetch_online_ord(
        [OXIDATION], max_workers=1, wall_timeout_sec=2.0
    )
    assert first["queried_steps"] == 1
    assert OXIDATION not in validator._online_queried_steps

    second = validator.batch_prefetch_online_ord(
        [OXIDATION], max_workers=1, wall_timeout_sec=2.0
    )
    assert calls == 2
    assert OXIDATION in validator._online_queried_steps

    validator.batch_prefetch_online_ord(
        [OXIDATION], max_workers=1, wall_timeout_sec=2.0
    )
    assert calls == 2


def test_online_ord_uses_largest_reactant_and_scaffold_fallback(monkeypatch):
    requests = []

    def submit(params, **kwargs):
        requests.append(dict(params))
        return []

    monkeypatch.setattr(ord_validator, "_submit_and_poll_ord_session", submit)
    forward = "CCO.CCOC(=O)c1ccc(Br)cc1>>CCOC(=O)c1ccc(-c2ccccc2)cc1"
    ord_validator._fetch_step_ord_records_worker(
        forward,
        similarity_threshold=0.40,
        per_step_timeout=4.0,
        core_product="c1ccc(-c2ccccc2)cc1",
        core_reactant="CCOC(=O)c1ccc(Br)cc1",
    )
    assert requests[0]["component"] == "CCOC(=O)c1ccc(Br)cc1;input;similarity"
    assert requests[1]["component"] == "c1ccc(-c2ccccc2)cc1;output;similarity"
    assert requests[1]["similarity"] == "0.30"


def test_padded_reaction_uses_harvested_template_and_preserves_input_key(tmp_path):
    validator = ord_validator.ORDRetrosynthesisValidator(tmp_path / "ord.duckdb")
    state = WorkbenchState(step_templates={
        OXIDATION: {"retro_template": RETRO_TEMPLATE, "classification": "Alcohol oxidation"},
    })
    padded = f" {OXIDATION}\n"
    results = classifier.batch_classify_with_gemini(
        [padded], use_gemini=False, state=state, ord_validator=validator
    )
    assert list(results) == [padded]
    assert results[padded]["reaction_name"] == "Alcohol oxidation"


def test_gemini_cache_is_scoped_to_the_ord_validator(tmp_path, monkeypatch):
    first = ord_validator.ORDRetrosynthesisValidator(tmp_path / "first.duckdb")
    second = ord_validator.ORDRetrosynthesisValidator(tmp_path / "second.duckdb")
    assert first.index_revision == second.index_revision
    state = WorkbenchState()
    offline_result = classifier.rdkit_classify_step(OXIDATION, ord_validator=first)
    response = SimpleNamespace(text=json.dumps({"steps": [offline_result]}))
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return response, "test-model", "medium"

    monkeypatch.setattr(classifier, "get_gemini_client", object)
    monkeypatch.setattr(classifier, "fetch_verified_literature_dois", lambda *args: [])
    monkeypatch.setattr(classifier, "call_gemini_with_config", generate)

    initial = classifier.batch_classify_with_gemini([OXIDATION], state=state, ord_validator=first)
    assert initial[OXIDATION]["gemini_enriched"]
    assert classifier.batch_classify_with_gemini([OXIDATION], state=state, ord_validator=first) == initial
    assert len(calls) == 1

    switched = classifier.batch_classify_with_gemini([OXIDATION], state=state, ord_validator=second)
    assert switched[OXIDATION]["gemini_tag"] != initial[OXIDATION]["gemini_tag"]
    assert len(calls) == 2


def test_pipeline_applies_templates_and_lab_stock_without_leaking_canvas_locks(tmp_path, monkeypatch):
    node = SimpleNamespace(
        reaction_smiles=lambda: "CC=O>>CCO",
        metadata={"retro_template": RETRO_TEMPLATE, "classification": "Alcohol oxidation"},
    )
    tree = SimpleNamespace(
        reactions=lambda: [node],
        child_reactions=lambda node: [],
        leafs=lambda: [SimpleNamespace(smiles="CCO")],
        molecules=lambda: [SimpleNamespace(smiles=smi) for smi in ("CCO", "CC=O")],
        in_stock=lambda leaf: False,
        is_solved=False,
    )
    finder = SimpleNamespace(config=SimpleNamespace(search=SimpleNamespace()), expansion_policy={})
    finder.tree_search = lambda: setattr(finder, "tree", object())
    finder.build_routes = lambda: setattr(finder, "routes", [{"reaction_tree": tree, "score": {"state score": 0.8}}])
    monkeypatch.setattr(pipeline, "get_finder", lambda: finder)

    validator = ord_validator.ORDRetrosynthesisValidator(tmp_path / "ord.duckdb")
    state = WorkbenchState()
    state.retro_search_params["validate_ord"] = False
    aizynth.register_custom_lab_stock(["CCO"], state=state)

    results = pipeline.run_retrosynthesis_pipeline("CC=O", use_gemini=False, state=state, ord_validator=validator)

    assert results is state.latest_retro_results
    assert results["is_solved"]
    assert state.step_templates[OXIDATION]["retro_template"] == RETRO_TEMPLATE
    assert results["routes"][0]["steps"][0]["reaction_name"] == "Alcohol oxidation"
    assert results["routes"][0]["starting_materials"]["precursors"][0]["source"] == "Lab Stock"
    assert finder.tree is None
    assert finder.routes == []

    state.canvas_lock_smarts = "[N]"
    locked = pipeline.run_retrosynthesis_pipeline("CC=O", use_gemini=False, state=state, ord_validator=validator)
    assert locked["locked_substructure_smarts"] == "[N]"
    assert locked["routes"] == []

    other_state = WorkbenchState()
    other_state.retro_search_params["validate_ord"] = False
    unlocked = pipeline.run_retrosynthesis_pipeline(
        "CC=O", use_gemini=False, state=other_state, ord_validator=validator
    )
    assert unlocked["locked_substructure_smarts"] == ""
    assert len(unlocked["routes"]) == 1
