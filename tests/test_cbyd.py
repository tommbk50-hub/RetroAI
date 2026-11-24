import json
from types import SimpleNamespace

import duckdb

from retroai.ai import classifier
from retroai.core.state import WorkbenchState
from retroai.cbyd.ingest import ingest_cbyd_routes
from retroai.cbyd.parser import assess_condition_hazards, parse_cbyd_step, parse_yield_fields
from retroai.ord.fingerprints import compute_spectator_functional_groups
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
    assert parse_yield_fields(["96% ee, 75% yield"])["yield_pct"] == 75.0


def test_cbyd_standardization_strips_salt_and_tracks_mapped_survival():
    step = _route("salted")["synthesis"][0]
    step["reaction_smiles_mapped"] = "CC[OH:1].[Cl-]>>CC=[O:1]"
    parsed = parse_cbyd_step(
        step, _route("salted")["background"], min_mapping_confidence=0.5
    )
    assert parsed["canonical_reaction"] == "CCO>>CC=O"

    shifted = compute_spectator_functional_groups(
        "[CH2:1]=[CH2:2].[CH2:3][CH2:4]>>[CH2:1][CH2:2].[CH2:3]=[CH2:4]"
    )
    assert shifted["reactant"]["alkene"] == 1
    assert shifted["product"]["alkene"] == 1
    assert "alkene" not in shifted["survived"]
    assert "Osmium" in " ".join(
        assess_condition_hazards(catalysts=["OsO4"])["hazard_flags"]
    )


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

    for path in routes.glob("*.json"):
        path.unlink()
    (routes / "replacement.json").write_text(
        json.dumps(_route("replacement", mapped="CCCO>>CCC=O")), encoding="utf-8"
    )
    ingest_cbyd_routes(routes, db_path, release_version="v2026.12")
    with duckdb.connect(str(db_path)) as con:
        assert con.execute(
            "SELECT reaction_id, release_version FROM cbyd_reactions"
        ).fetchall() == [("replacement", "v2026.12")]


def test_cbyd_ingestion_fits_small_duckdb_memory_limit(tmp_path, monkeypatch):
    routes = tmp_path / "routes"
    routes.mkdir()
    for length in range(1, 81):
        chain = "C" * length
        mapped = f"{chain}[CH2:1][OH:2]>>{chain}[CH:1]=[O:2]"
        (routes / f"r{length}.json").write_text(
            json.dumps(_route(f"r{length}", mapped=mapped)), encoding="utf-8"
        )
    connect = duckdb.connect
    monkeypatch.setattr(
        duckdb,
        "connect",
        lambda database, *args, **kwargs: connect(
            database, config={"memory_limit": "64MB"}
        ),
    )

    result = ingest_cbyd_routes(routes, tmp_path / "limited.duckdb")

    assert result["parsed"] == result["indexed"] == 80


def test_cbyd_ingestion_keeps_last_duplicate_reaction_id_and_reingests(tmp_path):
    routes = tmp_path / "routes"
    routes.mkdir()
    (routes / "a.json").write_text(json.dumps(_route("shared")), encoding="utf-8")
    (routes / "b.json").write_text(
        json.dumps(_route("shared", mapped="CCCO>>CCC=O")), encoding="utf-8"
    )
    db_path = tmp_path / "dupes.duckdb"

    for release in ("v2026.11", "v2026.12"):
        result = ingest_cbyd_routes(routes, db_path, release_version=release)
        assert result["parsed"] == 2
        assert result["indexed"] == 1
        with duckdb.connect(str(db_path)) as con:
            assert con.execute(
                "SELECT reaction_id, route_file, release_version FROM cbyd_reactions"
            ).fetchall() == [("shared", "b.json", release)]


def test_ingestion_uses_route_tree_reachability_and_rejects_bad_jsonl(tmp_path):
    routes = tmp_path / "routes"
    routes.mkdir()
    route = _route("keep")
    route["synthesis"].append(dict(_route("orphan")["synthesis"][0], reaction_id="orphan"))
    (routes / "route.json").write_text(json.dumps(route), encoding="utf-8")
    route_tree = tmp_path / "route_tree.jsonl"
    route_tree.write_text(json.dumps({
        "id": "route",
        "type": "mol",
        "children": [{
            "type": "reaction",
            "reaction_id": "keep",
            "children": [],
        }],
    }) + "\n", encoding="utf-8")
    db_path = tmp_path / "tree.duckdb"

    result = ingest_cbyd_routes(
        routes, db_path, route_tree_jsonl=route_tree, release_version="v2026.11"
    )
    assert result["parsed"] == 1
    with duckdb.connect(str(db_path)) as con:
        assert con.execute("SELECT reaction_id FROM cbyd_reactions").fetchall() == [("keep",)]

    route_tree.write_text("{not json}\n", encoding="utf-8")
    try:
        ingest_cbyd_routes(routes, db_path, route_tree_jsonl=route_tree)
    except ValueError as exc:
        assert "Invalid CbyD route-tree JSON" in str(exc)
    else:
        raise AssertionError("Malformed route tree JSON must fail visibly")


def test_validator_rescues_step_with_cbyd_without_counting_telescoped_yield(tmp_path, monkeypatch):
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
    assert all(
        row["equivalents"] == 0
        for row in classified["reagent_stoichiometry"]
        if row["role"] == "Reactant"
    )

    proposed = json.loads(json.dumps(classified))
    proposed["experimental_procedure"] = "Invented procedure"
    proposed["source_doi_or_patent"] = "10.1000/example"
    response = SimpleNamespace(text=json.dumps({"steps": [proposed]}))
    monkeypatch.setattr(classifier, "get_gemini_client", object)
    monkeypatch.setattr(classifier, "fetch_verified_literature_dois", lambda *args: [])
    monkeypatch.setattr(
        classifier,
        "call_gemini_with_config",
        lambda **kwargs: (response, "test-model", "medium"),
    )
    enriched = classifier.batch_classify_with_gemini(
        [OXIDATION], state=WorkbenchState(), ord_validator=validator
    )[OXIDATION]
    assert enriched["gemini_enriched"]
    assert enriched["experimental_procedure"] == ""
    assert all(
        row["equivalents"] == 0
        for row in enriched["reagent_stoichiometry"]
        if row["role"] == "Reactant"
    )
