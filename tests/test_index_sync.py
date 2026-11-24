import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
import requests

import retroai
from retroai.cbyd.ingest import ingest_cbyd_routes
from retroai.ord import index_sync
from retroai.ord.validator import ORDRetrosynthesisValidator


OXIDATION = "CCO>>CC=O"
RETRO_TEMPLATE = "[C:1]=[O:2]>>[C:1][O:2]"


class FakeResponse:
    def __init__(self, *, json_body=None, content=b"", status_code=200):
        self.json_body = json_body
        self.content = content
        self.status_code = status_code
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} response", response=self
            )

    def json(self):
        return self.json_body

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset:offset + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _route():
    return {
        "background": {
            "doi": "10.1000/prebuilt",
            "citation": "Prebuilt CbyD fixture",
            "drug_name": "Fixture",
        },
        "synthesis": [{
            "reaction_id": "prebuilt-oxidation",
            "reaction_smiles_mapped": OXIDATION,
            "mapping_confidence": 0.95,
            "conditions": {
                "GENERAL_REAGENTS": ["PDC"],
                "GENERAL_REAGENTS_SMILES": [],
                "CATALYSTS": [],
                "LIGANDS": [],
                "SOLVENTS": ["DCM"],
                "SOLVENTS_SMILES": [],
                "MODIFIERS": [],
                "TEMPERATURE": ["0"],
                "YIELD": ["80%"],
            },
        }],
    }


def _build_prebuilt(tmp_path):
    routes = tmp_path / "routes"
    routes.mkdir()
    (routes / "fixture.json").write_text(json.dumps(_route()), encoding="utf-8")
    path = tmp_path / "cbyd_v-test.duckdb"
    result = ingest_cbyd_routes(routes, path, release_version="v-test")
    payload = path.read_bytes()
    manifest = {
        "index_schema_version": index_sync.INDEX_SCHEMA_VERSION,
        "updated_at": "2026-01-01T00:00:00+00:00",
        "cbyd": {
            "release": "v-test",
            "file": path.name,
            "rows": result["indexed"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "license": "CC-BY-4.0",
            "duckdb_version": duckdb.__version__,
        },
    }
    return path, payload, manifest


def test_importing_index_sync_has_no_network_or_database_side_effects(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(retroai.__file__).resolve().parents[1])
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import duckdb
import requests

def fail(*args, **kwargs):
    raise AssertionError("index_sync import performed I/O")

requests.get = fail
duckdb.connect = fail
from retroai.ord import index_sync
assert index_sync.INDEX_SCHEMA_VERSION > 0
""",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    assert not (tmp_path / "data").exists()


def test_fetch_manifest_validates_schema_version(monkeypatch, tmp_path):
    _, _, manifest = _build_prebuilt(tmp_path)
    monkeypatch.setattr(
        index_sync.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(json_body=manifest),
    )
    assert (
        index_sync.fetch_index_manifest()["cbyd"]["release"] == "v-test"
    )

    incompatible = {
        **manifest,
        "index_schema_version": index_sync.INDEX_SCHEMA_VERSION + 1,
    }
    monkeypatch.setattr(
        index_sync.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(json_body=incompatible),
    )
    with pytest.raises(index_sync.IndexSchemaMismatch, match="update RetroAI"):
        index_sync.fetch_index_manifest()


def test_checksum_failure_does_not_replace_existing_download(
    monkeypatch, tmp_path
):
    source, payload, manifest = _build_prebuilt(tmp_path)
    manifest["cbyd"]["sha256"] = "0" * 64

    def fake_get(url, **kwargs):
        return (
            FakeResponse(json_body=manifest)
            if url.endswith("/manifest.json")
            else FakeResponse(content=payload)
        )

    monkeypatch.setattr(index_sync.requests, "get", fake_get)
    target = tmp_path / "data" / source.name
    target.parent.mkdir()
    target.write_bytes(b"previous-complete-file")

    with pytest.raises(index_sync.IndexSyncError, match="checksum mismatch"):
        index_sync.sync_prebuilt_cbyd_index(
            tmp_path / "data" / "ord_precedents.duckdb",
            base_url="https://indexes.invalid",
            index_path=target,
        )

    assert target.read_bytes() == b"previous-complete-file"
    assert not list(target.parent.glob(".*.part"))


def test_atomic_sync_merges_cbyd_without_clobbering_ord(
    monkeypatch, tmp_path
):
    source, payload, manifest = _build_prebuilt(tmp_path)

    def fake_get(url, **kwargs):
        return (
            FakeResponse(json_body=manifest)
            if url.endswith("/manifest.json")
            else FakeResponse(content=payload)
        )

    monkeypatch.setattr(index_sync.requests, "get", fake_get)
    data_dir = tmp_path / "data"
    db_path = data_dir / "ord_precedents.duckdb"
    validator = ORDRetrosynthesisValidator(db_path)
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO ord_reactions (ord_id, rxn_smiles, prod_connectivity)
            VALUES ('ord-preserved', ?, 'preserved-connectivity')
            """,
            [OXIDATION],
        )
    validator.refresh_index()

    result = index_sync.sync_prebuilt_cbyd_index(
        db_path,
        base_url="https://indexes.invalid",
        index_path=data_dir / source.name,
    )

    assert result["status"] == "synced"
    assert result["downloaded"]
    assert not list(data_dir.glob(".*.part"))
    with duckdb.connect(str(db_path), read_only=True) as con:
        assert con.execute(
            "SELECT ord_id, rxn_smiles FROM ord_reactions"
        ).fetchall() == [("ord-preserved", OXIDATION)]
        assert con.execute(
            "SELECT count(*) FROM cbyd_reactions"
        ).fetchone()[0] == 1
    assert validator.get_index_size() == 1
    assert validator.get_cbyd_index_size() == 1
    report = validator.validate_step(
        OXIDATION,
        template_info={"retro_template": RETRO_TEMPLATE},
    )
    assert report["cbyd_matched_count"] == 1
    assert {hit["provider"] for hit in report["top_precedents"]} == {
        "ORD",
        "CbyD",
    }

    second = index_sync.sync_prebuilt_cbyd_index(
        db_path,
        base_url="https://indexes.invalid",
        index_path=data_dir / source.name,
    )
    assert second["status"] == "current"
    assert not second["installed"]

    ingest_cbyd_routes(
        tmp_path / "routes",
        db_path,
        release_version="v-test",
    )
    with duckdb.connect(str(db_path), read_only=True) as con:
        assert con.execute(
            "SELECT count(*) FROM cbyd_metadata "
            "WHERE key = 'prebuilt_sha256'"
        ).fetchone()[0] == 0


def test_validator_ignores_incompatible_cbyd_schema(tmp_path):
    validator = ORDRetrosynthesisValidator(tmp_path / "precedents.duckdb")
    with duckdb.connect(str(validator.db_path)) as con:
        con.execute("CREATE TABLE cbyd_reactions (reaction_id VARCHAR)")
        con.execute("CREATE TABLE cbyd_metadata (key VARCHAR, value VARCHAR)")
        con.execute(
            "INSERT INTO cbyd_metadata VALUES "
            "('index_schema_version', '999')"
        )
    validator.refresh_index()
    assert validator.get_cbyd_index_size() == 0
    assert validator.validate_step(OXIDATION)["cbyd_matched_count"] == 0


@pytest.mark.parametrize("network_failure", ["http", "connection"])
def test_best_effort_sync_gracefully_falls_back(
    monkeypatch, tmp_path, network_failure
):
    def unavailable(*args, **kwargs):
        if network_failure == "http":
            return FakeResponse(status_code=404)
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(index_sync.requests, "get", unavailable)
    result = index_sync.try_sync_prebuilt_cbyd_index(
        tmp_path / "ord_precedents.duckdb",
        base_url="https://indexes.invalid",
    )
    assert result["status"] == "unavailable"
    assert "Prebuilt CbyD index is unavailable" in result["error"]
    assert not (tmp_path / "ord_precedents.duckdb").exists()
