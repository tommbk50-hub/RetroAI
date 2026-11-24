import json
from types import SimpleNamespace

import duckdb
from rdkit import Chem

from retroai.cbyd.advisor import advise_stereochemical_control
from retroai.cbyd.ingest import ingest_cbyd_routes
from retroai.ai.classifier import classify_step_strategy_role
from retroai.core.chem_utils import rdkit_detect_stereochemical_changes
from retroai.engine import pipeline
from retroai.engine.aizynth import (
    COMMON_LAB_REAGENTS,
    _flatten_reaction_tree,
    is_common_lab_reagent,
)
from retroai.ord import validator as ord_validator
from retroai.ord.validator import ORDRetrosynthesisValidator


def _route(reaction_id, reaction, *, reagents, yield_value="75%", annotation=""):
    return {
        "background": {
            "doi": f"10.1000/{reaction_id}",
            "citation": f"Peer reviewed {reaction_id}",
            "drug_name": "Integration fixture",
        },
        "synthesis": [{
            "reaction_id": reaction_id,
            "reaction_smiles_mapped": reaction,
            "mapping_confidence": 0.95,
            "conditions": {
                "GENERAL_REAGENTS": reagents,
                "GENERAL_REAGENTS_SMILES": [],
                "CATALYSTS": [],
                "LIGANDS": [],
                "SOLVENTS": ["MeCN"],
                "SOLVENTS_SMILES": ["CC#N"],
                "MODIFIERS": [],
                "TEMPERATURE": ["20"],
                "YIELD": [f"{yield_value} {annotation}".strip()],
            },
        }],
    }


def _step(index, reaction, name, precursor_steps=()):
    return {
        "step_index": index,
        "forward_smiles": reaction,
        "reaction_name": name,
        "typical_conditions": name,
        "precursor_steps": list(precursor_steps),
        "key_step_eligible": True,
    }


def test_route_tree_schema_and_cbyd_json_ingestion(tmp_path):
    routes = tmp_path / "routes"
    routes.mkdir()
    (routes / "oxidation.json").write_text(
        json.dumps(_route("oxidation", "CCO>>CC=O", reagents=["DMP"])),
        encoding="utf-8",
    )
    (tmp_path / "route_tree.jsonl").write_text(
        json.dumps({
            "id": "oxidation",
            "type": "mol",
            "smiles": "CC=O",
            "children": [{
                "type": "reaction",
                "reaction_id": "oxidation",
                "children": [{"type": "mol", "smiles": "CCO", "children": []}],
            }],
        }) + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "integration.duckdb"

    imported = ingest_cbyd_routes(
        routes,
        db_path,
        release_version="v2026.11",
        route_tree_jsonl=tmp_path / "route_tree.jsonl",
    )

    assert imported["indexed"] == 1
    validator = ORDRetrosynthesisValidator(db_path)
    assert validator.get_cbyd_index_size() == 1
    with duckdb.connect(str(db_path), read_only=True) as con:
        columns = {row[0] for row in con.execute("DESCRIBE cbyd_reactions").fetchall()}
    assert {"dr", "ee", "er", "survived_fgs", "release_version"} <= columns
    target_node = SimpleNamespace(
        reaction_smiles=lambda: "CC=O>>CCO",
        metadata={},
    )
    precursor_node = SimpleNamespace(
        reaction_smiles=lambda: "CCO>>CCCl",
        metadata={},
    )
    tree = SimpleNamespace(
        reactions=lambda: [target_node, precursor_node],
        child_reactions=lambda node: [precursor_node] if node is target_node else [],
    )
    flattened = _flatten_reaction_tree(tree, {
        id(target_node): {"classification": "Target-producing"},
        id(precursor_node): {"classification": "Precursor"},
    })
    assert [step["step_index"] for step in flattened] == [1, 2]
    assert flattened[1]["precursor_steps"] == [1]
    assert flattened[1]["forward_smiles"] == "CCO>>CC=O"


def test_stereoselective_yield_fields_and_chemoselective_reranking(tmp_path):
    routes = tmp_path / "routes"
    routes.mkdir()
    tbs_reaction = "CC(C)(C)[Si](C)(C)OCCO>>CC(C)(C)[Si](C)(C)OCC=O"
    (routes / "mild.json").write_text(
        json.dumps(_route(
            "mild-tbs",
            tbs_reaction,
            reagents=["Dess-Martin periodinane"],
            yield_value="82%",
            annotation="(2 steps), 96% ee, 19:1 dr",
        )),
        encoding="utf-8",
    )
    (routes / "harsh.json").write_text(
        json.dumps(_route(
            "jones",
            "CCCO>>CCC=O",
            reagents=["Jones reagent"],
            yield_value="68%",
            annotation="(2 steps)",
        )),
        encoding="utf-8",
    )
    db_path = tmp_path / "precedents.duckdb"
    validator = ORDRetrosynthesisValidator(db_path)
    ingest_cbyd_routes(routes, db_path, release_version="v2026.11")
    validator._refresh_index()

    report = validator.validate_step(tbs_reaction, template_info={}, top_k=3)
    hits = report["top_precedents"]
    assert hits
    assert hits[0]["precedent_id"] == "mild-tbs"
    assert "Dess-Martin" in hits[0]["reagents"]
    assert report["mean_analog_yield_pct"] is None
    assert hits[0]["step_count"] == 2
    assert hits[0]["dr"] == "19:1 dr"
    assert hits[0]["ee"] == "96% ee"


def test_validator_merges_strong_ord_and_cbyd_analog_sets(tmp_path, monkeypatch):
    reaction = "CCO>>CC=O"
    db_path = tmp_path / "dual.duckdb"
    validator = ORDRetrosynthesisValidator(db_path)
    ord_record = {
        "ord_id": "ord-strong",
        "rxn_smiles": reaction,
        "prod_inchikey": Chem.MolToInchiKey(Chem.MolFromSmiles("CC=O")),
        "prod_connectivity": Chem.MolToInchiKey(Chem.MolFromSmiles("CC=O")).split("-")[0],
        "yield_pct": 90.0,
        "catalysts": "None",
        "reagents": "DMP",
        "solvents": "DCM",
        "temp_c": 20.0,
        "atmosphere": "NITROGEN",
        "undesired_byproducts": "None",
        "reference": "ORD",
        "fp_bytes": ord_validator.compute_folded_rxn_fp(reaction),
    }
    with duckdb.connect(str(db_path)) as con:
        columns = ",".join(ord_record)
        con.execute(
            f"INSERT INTO ord_reactions ({columns}) VALUES ({','.join('?' for _ in ord_record)})",
            list(ord_record.values()),
        )
    validator._refresh_index()
    routes = tmp_path / "routes"
    routes.mkdir()
    (routes / "cbyd.json").write_text(
        json.dumps(_route("cbyd-dual", "CCCO>>CCC=O", reagents=["TEMPO"])),
        encoding="utf-8",
    )
    ingest_cbyd_routes(routes, db_path, release_version="v2026.11")

    report = validator.validate_step(reaction)

    assert report["smarts_matched_count"] == 2
    assert {hit["provider"] for hit in report["top_precedents"]} == {"ORD", "CbyD"}
    assert report["top_precedents"][0]["provider"] == "ORD"


def test_complexity_key_steps_and_loop_penalties(monkeypatch):
    steps = [
        _step(1, "CC(=O)O>>CCCO", "Esterification", [2]),
        _step(2, "CCCO>>CC(=O)O", "Ester hydrolysis", [3]),
        _step(3, "BrCCO>>OCCO", "C-C bond construction"),
    ]
    monkeypatch.setattr(
        pipeline,
        "rdkit_reaction_complexity",
        lambda reaction: {
            "product_sascore": 4.0,
            "product_sps": 500.0,
            "reactant_max_sascore": 3.2,
            "reactant_sum_sps": 300.0,
            "delta_sascore": 0.7,
            "delta_sps": 200.0,
            "complexity_error": "",
        },
    )
    route = {"steps": steps}

    annotated = pipeline.annotate_route_strategy(route)

    assert annotated["lls_step_indices"] == [1, 2, 3]
    assert 3 in annotated["strategic_key_step_indices"]
    assert annotated["unproductive_loops"][0]["type"] == "esterification_hydrolysis_loop"
    assert pipeline.compute_composite_route_score(
        annotated, "strategy_complexity"
    ) < pipeline.compute_composite_route_score(
        {**annotated, "unproductive_loop_count": 0}, "strategy_complexity"
    )
    assert classify_step_strategy_role("Oxidative cyclization")[0] == "bond_construction"
    assert classify_step_strategy_role("Simple alcohol oxidation")[0] == "simple_redox"


def test_stereochemical_creation_is_detected_and_flagged():
    mapped = "[CH3:1][C:2](=[O:3])[CH2:4][CH3:5]>>[CH3:1][C@H:2]([OH:3])[CH2:4][CH3:5]"
    report = rdkit_detect_stereochemical_changes(mapped)

    assert report["mapping_available"]
    assert report["creates_stereochemistry"]
    assert "tetrahedral_creation" in {event["type"] for event in report["events"]}
    advice = advise_stereochemical_control(
        mapped,
        report,
        conditions="NaBH4, MeOH",
    )
    assert "Stereochemical Control Risk" in advice["stereochemical_control_risk"]
    assert advice["chiral_pool_suggestions"]
    assert advice["chiral_auxiliary_suggestions"]
    ez = rdkit_detect_stereochemical_changes(r"F/C=C/F>>F/C=C\F")
    assert not ez["mapping_available"]
    assert ez["inverts_stereochemistry"]
    assert "alkene_ez_inversion" in {event["type"] for event in ez["events"]}


def test_common_lab_reagent_allowlist_is_opt_in_and_canonical():
    assert len(COMMON_LAB_REAGENTS) == 10
    assert is_common_lab_reagent("C[Si](C)(C)Cl")
    assert is_common_lab_reagent("O")
    assert not is_common_lab_reagent(Chem.MolToSmiles(Chem.MolFromSmiles("CCO")))
