import gzip

import pytest

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
