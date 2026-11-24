"""Ingest Chemistry by Design route JSON files into the local DuckDB index."""

import argparse
import json
import os
from pathlib import Path

import duckdb

from retroai.ord.index_sync import INDEX_SCHEMA_VERSION
from retroai.ord.validator import ORD_DB_PATH

from .parser import parse_cbyd_step


CBYD_SCHEMA = {
    "reaction_id": "VARCHAR PRIMARY KEY",
    "canonical_reaction": "VARCHAR",
    "rxn_smiles": "VARCHAR",
    "prod_connectivity": "VARCHAR",
    "mapping_confidence": "DOUBLE",
    "doi": "VARCHAR",
    "citation": "VARCHAR",
    "drug_name": "VARCHAR",
    "route_file": "VARCHAR",
    "release_version": "VARCHAR",
    "general_reagents": "VARCHAR",
    "catalysts": "VARCHAR",
    "ligands": "VARCHAR",
    "solvents": "VARCHAR",
    "modifiers": "VARCHAR",
    "temperature": "VARCHAR",
    "general_reagents_smiles": "VARCHAR",
    "catalysts_smiles": "VARCHAR",
    "ligands_smiles": "VARCHAR",
    "solvents_smiles": "VARCHAR",
    "modifiers_smiles": "VARCHAR",
    "yield_raw": "VARCHAR",
    "yield_pct": "DOUBLE",
    "step_count": "INTEGER",
    "is_borsm": "BOOLEAN",
    "dr": "VARCHAR",
    "ee": "VARCHAR",
    "er": "VARCHAR",
    "raw_description": "VARCHAR",
    "fp_bytes": "BLOB",
    "drfp_bytes": "BLOB",
    "diff_fp_bytes": "BLOB",
    "consumed_fgs": "VARCHAR",
    "formed_fgs": "VARCHAR",
    "reactant_fgs": "VARCHAR",
    "product_fgs": "VARCHAR",
    "survived_fgs": "VARCHAR",
    "hazard_flags": "VARCHAR",
    "safer_alternatives": "VARCHAR",
    "route_count": "INTEGER",
}


def ensure_cbyd_schema(con):
    """Create or migrate the CbyD table and release metadata table."""
    definitions = ", ".join(f"{name} {kind}" for name, kind in CBYD_SCHEMA.items())
    con.execute(f"CREATE TABLE IF NOT EXISTS cbyd_reactions ({definitions})")
    existing = {row[0] for row in con.execute("DESCRIBE cbyd_reactions").fetchall()}
    for name, kind in CBYD_SCHEMA.items():
        if name not in existing:
            ddl = kind.replace(" PRIMARY KEY", "").replace(" UNIQUE", "")
            con.execute(f"ALTER TABLE cbyd_reactions ADD COLUMN {name} {ddl}")
    con.execute("""
        CREATE TABLE IF NOT EXISTS cbyd_metadata (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_cbyd_conn ON cbyd_reactions (prod_connectivity)")


def _collect_route_tree_flags(
    route_tree_path: str | os.PathLike | None,
) -> tuple[set[str], set[str], set[str]]:
    """Return represented routes, flagged routes, and reachable reaction IDs."""
    if not route_tree_path:
        return set(), set(), set()
    represented_routes, flagged_routes, reachable_reactions = set(), set(), set()
    with Path(route_tree_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid CbyD route-tree JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict) or not record.get("id"):
                raise ValueError(
                    f"CbyD route-tree line {line_number} must be an object with an id"
                )
            route_id = str(record["id"])
            represented_routes.add(route_id)
            if any(record.get(flag) for flag in ("invalid", "warning", "warnings", "orphan")):
                flagged_routes.add(route_id)
            stack = [record]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    flagged = any(
                        item.get(flag)
                        for flag in ("invalid", "warning", "warnings", "orphan")
                    )
                    reaction_id = item.get("reaction_id")
                    if reaction_id and not flagged and item.get("type") == "reaction":
                        reachable_reactions.add(str(reaction_id))
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
    return represented_routes, flagged_routes, reachable_reactions


def ingest_cbyd_routes(
    routes_dir: str | os.PathLike,
    db_path: str | os.PathLike = ORD_DB_PATH,
    *,
    release_version: str = "",
    min_mapping_confidence: float = 0.5,
    route_tree_jsonl: str | os.PathLike | None = None,
) -> dict:
    """Normalize, deduplicate, and index all CbyD route JSON files."""
    route_dir = Path(routes_dir)
    paths = sorted(route_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No CbyD route JSON files found in {route_dir}")
    represented_routes, skipped_routes, reachable_reactions = _collect_route_tree_flags(route_tree_jsonl)
    records, skipped = {}, {"invalid": 0, "low_confidence_or_unparseable": 0, "duplicate": 0}
    for path in paths:
        if route_tree_jsonl and (
            path.stem in skipped_routes or path.stem not in represented_routes
        ):
            skipped["invalid"] += 1
            continue
        try:
            route = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            skipped["invalid"] += 1
            continue
        background = route.get("background") or {}
        for step in route.get("synthesis") or []:
            if route_tree_jsonl and str(step.get("reaction_id") or "") not in reachable_reactions:
                skipped["invalid"] += 1
                continue
            parsed = parse_cbyd_step(
                step,
                background,
                route_file=path.name,
                release_version=release_version,
                min_mapping_confidence=min_mapping_confidence,
            )
            if parsed is None:
                skipped["low_confidence_or_unparseable"] += 1
                continue
            key = parsed["canonical_reaction"]
            if key in records:
                records[key]["route_count"] += 1
                skipped["duplicate"] += 1
                continue
            records[key] = parsed

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        ensure_cbyd_schema(con)
        con.execute("BEGIN")
        try:
            con.execute("DELETE FROM cbyd_reactions")
            for record in records.values():
                values = [record[name] for name in CBYD_SCHEMA]
                placeholders = ",".join("?" for _ in CBYD_SCHEMA)
                columns = ",".join(CBYD_SCHEMA)
                con.execute(
                    f"INSERT OR REPLACE INTO cbyd_reactions ({columns}) VALUES ({placeholders})",
                    values,
                )
            if release_version:
                con.execute(
                    "INSERT OR REPLACE INTO cbyd_metadata VALUES ('release_version', ?)",
                    [release_version],
                )
            con.execute(
                "INSERT OR REPLACE INTO cbyd_metadata VALUES ('source_license', 'CC BY 4.0')"
            )
            con.execute(
                "INSERT OR REPLACE INTO cbyd_metadata VALUES "
                "('index_schema_version', ?)",
                [str(INDEX_SCHEMA_VERSION)],
            )
            con.execute(
                "INSERT OR REPLACE INTO cbyd_metadata VALUES "
                "('duckdb_version', ?)",
                [duckdb.__version__],
            )
            con.execute(
                "DELETE FROM cbyd_metadata WHERE key = 'prebuilt_sha256'"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        indexed = con.execute("SELECT count(*) FROM cbyd_reactions").fetchone()[0]
    return {
        "files": len(paths),
        "parsed": len(records),
        "indexed": indexed,
        "release_version": release_version,
        "skipped": skipped,
        "db_path": str(db_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Index CbyD route JSON files for RetroAI.")
    parser.add_argument("routes_dir", help="Path to the CbyD data/routes directory")
    parser.add_argument("--db", default=ORD_DB_PATH, help="RetroAI DuckDB path")
    parser.add_argument("--release", default="v2026.11", help="CbyD release tag")
    parser.add_argument("--min-mapping-confidence", type=float, default=0.5)
    parser.add_argument("--route-tree-jsonl", help="Optional CbyD route_tree.jsonl validity input")
    args = parser.parse_args(argv)
    stats = ingest_cbyd_routes(
        args.routes_dir,
        args.db,
        release_version=args.release,
        min_mapping_confidence=args.min_mapping_confidence,
        route_tree_jsonl=args.route_tree_jsonl,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
