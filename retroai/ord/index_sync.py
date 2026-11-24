"""Download and install optional, prebuilt RetroAI reaction indexes."""

import hashlib
import json
import os
import re
import tempfile
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import quote

import duckdb
import requests


INDEX_SCHEMA_VERSION = 1
DEFAULT_INDEX_BASE_URL = (
    "https://github.com/tommbk50-hub/RetroAI/releases/download/indexes-latest"
)
REQUIRED_CBYD_COLUMNS = frozenset({
    "reaction_id",
    "canonical_reaction",
    "rxn_smiles",
    "prod_connectivity",
    "mapping_confidence",
    "doi",
    "citation",
    "drug_name",
    "route_file",
    "release_version",
    "general_reagents",
    "catalysts",
    "ligands",
    "solvents",
    "modifiers",
    "temperature",
    "general_reagents_smiles",
    "catalysts_smiles",
    "ligands_smiles",
    "solvents_smiles",
    "modifiers_smiles",
    "yield_raw",
    "yield_pct",
    "step_count",
    "is_borsm",
    "dr",
    "ee",
    "er",
    "raw_description",
    "fp_bytes",
    "drfp_bytes",
    "diff_fp_bytes",
    "consumed_fgs",
    "formed_fgs",
    "reactant_fgs",
    "product_fgs",
    "survived_fgs",
    "hazard_flags",
    "safer_alternatives",
    "route_count",
})


class IndexSyncError(RuntimeError):
    """Base error for an unavailable or invalid prebuilt index."""


class IndexSchemaMismatch(IndexSyncError):
    """The published index is not compatible with this RetroAI checkout."""


def _manifest_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/manifest.json"


def fetch_index_manifest(
    base_url: str = DEFAULT_INDEX_BASE_URL,
    *,
    timeout: float = 20.0,
) -> dict:
    """Fetch and validate release metadata without performing other side effects."""
    url = _manifest_url(base_url)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        manifest = response.json()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f" (HTTP {status})" if status else ""
        raise IndexSyncError(
            f"Prebuilt CbyD index is unavailable{detail} at {url}: {exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise IndexSyncError(f"Prebuilt index manifest at {url} is invalid: {exc}") from exc
    return validate_index_manifest(manifest)


def validate_index_manifest(manifest: dict) -> dict:
    """Return a validated manifest, rejecting unsupported index schema versions."""
    if not isinstance(manifest, dict):
        raise IndexSyncError("Prebuilt index manifest must be a JSON object.")
    try:
        published_version = int(manifest["index_schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexSyncError("Prebuilt index manifest has no valid index_schema_version.") from exc
    if published_version != INDEX_SCHEMA_VERSION:
        raise IndexSchemaMismatch(
            "Prebuilt CbyD index schema "
            f"{published_version} is incompatible with RetroAI schema "
            f"{INDEX_SCHEMA_VERSION}; update RetroAI or rebuild the index."
        )
    cbyd = manifest.get("cbyd")
    if not isinstance(cbyd, dict):
        raise IndexSyncError("Prebuilt index manifest has no cbyd entry.")
    filename = cbyd.get("file")
    if (
        not isinstance(filename, str)
        or not filename.endswith(".duckdb")
        or "\\" in filename
        or Path(filename).name != filename
    ):
        raise IndexSyncError("Prebuilt index manifest contains an unsafe CbyD filename.")
    release = cbyd.get("release")
    if not isinstance(release, str) or not release.strip():
        raise IndexSyncError("Prebuilt index manifest has no CbyD release tag.")
    checksum = cbyd.get("sha256")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum.lower()):
        raise IndexSyncError("Prebuilt index manifest contains an invalid CbyD sha256.")
    try:
        rows = int(cbyd["rows"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexSyncError("Prebuilt index manifest contains an invalid CbyD row count.") from exc
    if rows <= 0:
        raise IndexSyncError("Prebuilt CbyD index is unexpectedly empty.")
    if not isinstance(cbyd.get("license"), str) or not cbyd["license"].strip():
        raise IndexSyncError("Prebuilt index manifest has no CbyD license.")
    if not isinstance(cbyd.get("duckdb_version"), str) or not re.match(
        r"^\d+\.\d+", cbyd["duckdb_version"]
    ):
        raise IndexSyncError("Prebuilt index manifest has no valid DuckDB version.")
    normalized = dict(manifest)
    normalized["index_schema_version"] = published_version
    normalized["cbyd"] = {**cbyd, "sha256": checksum.lower(), "rows": rows}
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duckdb_series(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", str(version))
    return (int(match.group(1)), int(match.group(2))) if match else (-1, -1)


def _validate_prebuilt_index(path: Path, manifest: dict) -> None:
    """Open a downloaded index read-only and validate its schema and metadata."""
    expected = manifest["cbyd"]
    try:
        with duckdb.connect(str(path), read_only=True) as con:
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            }
            if not {"cbyd_reactions", "cbyd_metadata"} <= tables:
                raise IndexSyncError(
                    "Prebuilt CbyD database is missing cbyd_reactions or cbyd_metadata."
                )
            columns = {
                row[0] for row in con.execute("DESCRIBE cbyd_reactions").fetchall()
            }
            missing = sorted(REQUIRED_CBYD_COLUMNS - columns)
            if missing:
                raise IndexSyncError(
                    "Prebuilt CbyD database is missing required columns: "
                    + ", ".join(missing)
                )
            rows = con.execute("SELECT count(*) FROM cbyd_reactions").fetchone()[0]
            if rows <= 0 or rows != expected["rows"]:
                raise IndexSyncError(
                    f"Prebuilt CbyD row count {rows} does not match manifest "
                    f"{expected['rows']}."
                )
            invalid_fingerprints = con.execute(
                """
                SELECT count(*) FROM cbyd_reactions
                WHERE drfp_bytes IS NULL OR octet_length(drfp_bytes) <> 256
                """
            ).fetchone()[0]
            if invalid_fingerprints:
                raise IndexSyncError(
                    "Prebuilt CbyD database has missing or incompatible DRFP fingerprints."
                )
            metadata = dict(con.execute(
                "SELECT key, value FROM cbyd_metadata"
            ).fetchall())
    except IndexSyncError:
        raise
    except Exception as exc:
        raise IndexSyncError(
            f"Prebuilt CbyD database cannot be opened read-only: {exc}. "
            "Confirm that RetroAI's pinned DuckDB version is installed."
        ) from exc
    if metadata.get("index_schema_version") != str(INDEX_SCHEMA_VERSION):
        raise IndexSchemaMismatch(
            "Prebuilt CbyD database metadata does not match RetroAI's index schema "
            f"{INDEX_SCHEMA_VERSION}."
        )
    if expected.get("release") and metadata.get("release_version") != expected["release"]:
        raise IndexSyncError("Prebuilt CbyD release metadata does not match the manifest.")
    producer_version = expected.get("duckdb_version") or metadata.get("duckdb_version")
    if producer_version and _duckdb_series(producer_version) != _duckdb_series(duckdb.__version__):
        raise IndexSyncError(
            "Prebuilt CbyD database was produced with DuckDB "
            f"{producer_version}, but this environment has {duckdb.__version__}. "
            "Install the DuckDB version range pinned by RetroAI."
        )


def _download_prebuilt_index(
    url: str,
    target: Path,
    expected_sha256: str,
    manifest: dict,
    *,
    timeout: float,
    progress: bool,
    force: bool,
) -> bool:
    if target.is_file() and not force and _sha256(target) == expected_sha256:
        _validate_prebuilt_index(target, manifest)
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".part", dir=target.parent
    )
    temporary = Path(temporary_name)
    os.close(file_descriptor)
    try:
        try:
            response = requests.get(url, stream=True, timeout=(timeout, timeout * 2))
            response.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            detail = f" (HTTP {status})" if status else ""
            raise IndexSyncError(
                f"Prebuilt CbyD download is unavailable{detail} at {url}: {exc}"
            ) from exc
        try:
            total = int(response.headers.get("content-length") or 0)
        except (TypeError, ValueError):
            total = 0
        progress_bar = nullcontext(None)
        if progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading CbyD index",
                )
            except ImportError:
                pass
        with response, progress_bar as bar, temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        actual = _sha256(temporary)
        if actual != expected_sha256:
            raise IndexSyncError(
                f"Prebuilt CbyD checksum mismatch: expected {expected_sha256}, got {actual}."
            )
        _validate_prebuilt_index(temporary, manifest)
        os.replace(temporary, target)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _destination_is_current(destination: Path, manifest: dict) -> bool:
    if not destination.is_file():
        return False
    try:
        with duckdb.connect(str(destination), read_only=True) as con:
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            }
            if not {"cbyd_reactions", "cbyd_metadata"} <= tables:
                return False
            rows = con.execute("SELECT count(*) FROM cbyd_reactions").fetchone()[0]
            metadata = dict(con.execute(
                "SELECT key, value FROM cbyd_metadata"
            ).fetchall())
    except Exception:
        return False
    return (
        rows == manifest["cbyd"]["rows"]
        and metadata.get("index_schema_version") == str(INDEX_SCHEMA_VERSION)
        and metadata.get("release_version") == manifest["cbyd"].get("release")
        and metadata.get("prebuilt_sha256") == manifest["cbyd"]["sha256"]
    )


def _install_cbyd_tables(source: Path, destination: Path, checksum: str) -> None:
    """Replace only CbyD tables in the local ORD database in one transaction."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sql = str(source.resolve()).replace("'", "''")
    with duckdb.connect(str(destination)) as con:
        con.execute(f"ATTACH '{source_sql}' AS prebuilt_cbyd (READ_ONLY)")
        con.execute("BEGIN TRANSACTION")
        try:
            for table in ("cbyd_reactions", "cbyd_metadata"):
                ddl = con.execute(
                    """
                    SELECT sql FROM duckdb_tables()
                    WHERE database_name = 'prebuilt_cbyd' AND table_name = ?
                    """,
                    [table],
                ).fetchone()[0]
                con.execute(f"DROP TABLE IF EXISTS {table}")
                con.execute(ddl)
                con.execute(
                    f"INSERT INTO {table} "
                    f"SELECT * FROM prebuilt_cbyd.{table}"
                )
            con.execute(
                "DELETE FROM cbyd_metadata WHERE key = 'prebuilt_sha256'"
            )
            con.execute(
                "INSERT INTO cbyd_metadata VALUES ('prebuilt_sha256', ?)",
                [checksum],
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_cbyd_conn "
                "ON cbyd_reactions (prod_connectivity)"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.execute("DETACH prebuilt_cbyd")


def sync_prebuilt_cbyd_index(
    db_path: str | os.PathLike | None = None,
    *,
    base_url: str = DEFAULT_INDEX_BASE_URL,
    index_path: str | os.PathLike | None = None,
    timeout: float = 20.0,
    progress: bool = False,
    force: bool = False,
) -> dict:
    """Atomically download and merge the latest schema-compatible CbyD index."""
    manifest = fetch_index_manifest(base_url, timeout=timeout)
    db_path = Path(
        db_path or os.environ.get("RETROAI_ORD_DB_PATH", "data/ord_precedents.duckdb")
    )
    asset = manifest["cbyd"]
    target = (
        Path(index_path)
        if index_path is not None
        else db_path.parent / asset["file"]
    )
    if target.resolve() == db_path.resolve():
        raise IndexSyncError(
            "The prebuilt index cache path must be separate from the shared ORD database."
        )
    downloaded = _download_prebuilt_index(
        f"{base_url.rstrip('/')}/{quote(asset['file'])}",
        target,
        asset["sha256"],
        manifest,
        timeout=timeout,
        progress=progress,
        force=force,
    )
    if not force and _destination_is_current(db_path, manifest):
        return {
            "status": "current",
            "downloaded": downloaded,
            "installed": False,
            "release": asset.get("release", ""),
            "rows": asset["rows"],
            "index_path": str(target),
            "db_path": str(db_path),
        }
    _install_cbyd_tables(target, db_path, asset["sha256"])
    return {
        "status": "synced",
        "downloaded": downloaded,
        "installed": True,
        "release": asset.get("release", ""),
        "rows": asset["rows"],
        "index_path": str(target),
        "db_path": str(db_path),
    }


def try_sync_prebuilt_cbyd_index(
    db_path: str | os.PathLike | None = None, **kwargs
) -> dict:
    """Best-effort notebook sync that preserves ORD-only operation on failure."""
    try:
        return sync_prebuilt_cbyd_index(db_path, **kwargs)
    except (IndexSyncError, OSError, TypeError, ValueError, duckdb.Error) as exc:
        return {"status": "unavailable", "error": str(exc)}
