import json

import duckdb

from retroai.ai import classifier
from retroai.cbyd.ingest import ingest_cbyd_routes
from retroai.cbyd.parser import parse_yield_fields
from retroai.ord.validator import ORDRetrosynthesisValidator


OXIDATION = "CCO>>CC=O"
RETRO_TEMPLATE = "[C:1]=[O:2]>>[C:1][O:2]"


def _route(reaction_id, mapped=OXIDATION, yield_value="75%", confidence=0.9):
    return {
        "background": {
            "doi": "10.1000/example",
            "citation": "Example route, 2026",
            "drug_name": "Example drug",
        },
        "synthesis": [{
            "reaction_id": reaction_id,
            "reaction_smiles_mapped": mapped,
            "mapping_confidence": confidence,
            "conditions": {
                "GENERAL_REAGENTS": ["PDC"],
                "GENERAL_REAGENTS_SMILES": ["[O-][Cr](=O)(=O)O[Cr](=O)(=O)[O-]"],
                "CATALYSTS": [],
                "LIGANDS": [],
                "SOLVENTS": ["DCM"],
                "SOLVENTS_SMILES": ["ClCCl"],
                "MODIFIERS": [],
                "TEMPERATURE": ["0"],
                "YIELD": [yield_value],
                "raw_description": "PDC; DCM; 0 °C; " + yield_value,
            },
        }],
    }


def test_yield_parser_preserves_telescoped_and_selectivity_fields():
    parsed = parse_yield_fields(["75% (2 steps), 96% ee, 19:1 dr"])
    assert parsed["yield_pct"] == 75.0
    assert parsed["step_count"] == 2
    assert parsed["dr"] == "19:1 dr"
    assert parsed["ee"] == "96% ee"

    borsm = parse_yield_fields(["64% (BORSM)"])
    assert borsm["is_borsm"]
    assert borsm["yield_pct"] == 64.0


def test_cbyd_ingestion_deduplicates_filters_and_records_metadata(tmp_path):
    routes = tmp_path / "routes"
    routes.mkdir()
    (routes / "first.json").write_text(json.dumps(_route("first")), encoding="utf-8")
    (routes / "second.json").write_text(json.dumps(_route("second")), encoding="utf-8")
    (routes / "low.json").write_text(
        json.dumps(_route("low", confidence=0.4)), encoding="utf-8"
    )
    invalid = _route("invalid")
    invalid["synthesis"][0]["invalid"] = True
    (routes / "invalid.json").write_text(json.dumps(invalid), encoding="utf-8")
    db_path = tmp_path / "precedents.duckdb"

    result = ingest_cbyd_routes(routes, db_path, release_version="v2026.11")

    assert result["parsed"] == 1
    assert result["skipped"]["duplicate"] == 1
    assert result["skipped"]["low_confidence_or_unparseable"] == 2
    with duckdb.connect(str(db_path)) as con:
        row = con.execute(
            "SELECT route_count, release_version, hazard_flags, survived_fgs "
            "FROM cbyd_reactions"
        ).fetchone()
        release = con.execute(
            "SELECT value FROM cbyd_metadata WHERE key = 'release_version'"
        ).fetchone()[0]
    assert row[0] == 2
    assert row[1] == "v2026.11"
    assert "Chromium" in row[2]
    assert json.loads(row[3]) == {}
    assert release == "v2026.11"


def test_validator_rescues_step_with_cbyd_without_counting_telescoped_yield(tmp_path):
    routes = tmp_path / "routes"
    routes.mkdir()
    (routes / "route.json").write_text(
        json.dumps(_route("peer-reviewed-oxidation", yield_value="75% (2 steps)")),
        encoding="utf-8",
    )
    db_path = tmp_path / "precedents.duckdb"
    validator = ORDRetrosynthesisValidator(db_path)
    ingest_cbyd_routes(routes, db_path, release_version="v2026.11")
    validator._refresh_index()

    report = validator.validate_step(
        OXIDATION, template_info={"retro_template": RETRO_TEMPLATE}
    )

    assert report["is_validated"]
    assert report["cbyd_matched_count"] == 1
    assert report["mean_analog_yield_pct"] is None
    hit = report["top_precedents"][0]
    assert hit["provider"] == "CbyD"
    assert hit["provenance"] == "CbyD (Peer-Reviewed)"
    assert hit["doi"] == "10.1000/example"
    assert hit["step_count"] == 2
    assert hit["yield_pct"] == 75.0
    assert hit["hazard_flags"]

    classified = classifier.rdkit_classify_step(OXIDATION, ord_validator=validator)
    assert classified["safety_green_score"] <= 2
    assert classified["data_gap_warnings"][0] == "Check DOI for stoichiometry/time/workup"
    assert "telescoped 2-step yield" in classified["ranked_conditions"][0]["expected_yield_range"]
    assert classified["cbyd_precedents"][0]["step_count"] == 2
