import gzip
import json
from types import SimpleNamespace

import pytest

from retroai.ai import classifier
from retroai.core.state import WorkbenchState
from retroai.engine import aizynth, pipeline
from retroai.ord import fingerprints, validator as ord_validator
from ord_schema.proto import dataset_pb2, reaction_pb2


OXIDATION = "CCO>>CC=O"
RETRO_TEMPLATE = "[C:1]=[O:2]>>[C:1][O:2]"


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
    monkeypatch.setattr(ord_validator, "list_repo_files", lambda **kwargs: [shard.name])
    monkeypatch.setattr(ord_validator, "hf_hub_download", lambda **kwargs: shard)

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
