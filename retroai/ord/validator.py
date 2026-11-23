"""Index ORD precedents in DuckDB and validate route steps."""

import base64
import functools
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import disable_progress_bars
from ord_schema import datasets as ord_datasets
from ord_schema.proto import reaction_pb2 as ord_reaction_pb2
from rdkit import Chem, DataStructs
from requests.adapters import HTTPAdapter

from retroai.core.chem_utils import _extract_primary_smiles
from retroai.core.config import get_secret

from .fingerprints import (
    _compile_rxn_smarts_queries,
    _match_ord_gate,
    _parse_reaction_mols,
    _rdkit_extract_reacting_cores,
    build_progressive_reaction_smarts,
    compute_drfp_bytes,
    compute_folded_rxn_fp,
    compute_rdkit_internal_delta_fg,
    compute_rxn_center_diff_fp,
    compute_spectator_functional_groups,
    matches_reaction_smarts_mols,
    tanimoto_packed_fp_batch,
)
from .parser import _humanize_condition_list, parse_ord_reaction_protobuf


ORD_DB_PATH = "/dev/shm/ord_precedents.duckdb" if os.path.isdir("/dev/shm") else "data/ord_precedents.duckdb"
ORD_ONLINE_API_BASE = "https://open-reaction-database.org/api"


# Persistent pooled HTTP session for multi-threaded online ORD API requests
_ORD_HTTP_SESSION = requests.Session()
_ord_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=1)
_ORD_HTTP_SESSION.mount("https://", _ord_adapter)
_ORD_HTTP_SESSION.mount("http://", _ord_adapter)
_ORD_HTTP_SESSION.headers.update({
    "User-Agent": "ChemForge-Colab-ParallelValidator/2.5",
    "Accept": "application/json",
})


def _ord_gate_specs(forward_smiles_list, route_steps):
    if route_steps is None:
        route_steps = [{"forward_smiles": smi} for smi in forward_smiles_list or [] if smi]
    return dict.fromkeys((step["forward_smiles"].strip(), step.get("retro_template") or "", step.get("mapped_forward_smiles") or "")
                         for step in route_steps if ">>" in step.get("forward_smiles", ""))


def _counter_from_json(value) -> Counter:
    try:
        items = json.loads(value) if isinstance(value, str) else value
        return Counter(items) if isinstance(items, (list, tuple)) else Counter()
    except (TypeError, ValueError, json.JSONDecodeError):
        return Counter()


def _delta_cache_key(delta) -> tuple:
    return tuple(tuple(sorted(Counter(groups).items())) for groups in delta[:2])


def _json_value(value, fallback):
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        return parsed if isinstance(parsed, type(fallback)) else fallback
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _submit_and_poll_ord_session(
    params: List[tuple],
    timeout_sec: float = 3.2,
    poll_interval: float = 0.25,
) -> List[bytes]:
    """
    Thread-safe asynchronous query submission and polling against https://open-reaction-database.org/api
    using the shared connection-pooled HTTP session.
    """
    deadline = time.time() + timeout_sec
    try:
        sub_resp = _ORD_HTTP_SESSION.get(
            f"{ORD_ONLINE_API_BASE}/submit_query",
            params=params,
            timeout=1.8,
        )
        if sub_resp.status_code != 200:
            raise RuntimeError(f"ORD query submission returned HTTP {sub_resp.status_code}")
        try:
            payload = sub_resp.json()
            task_id = payload.get("task_id", payload) if isinstance(payload, dict) else str(payload)
        except Exception:
            task_id = sub_resp.text.strip().strip('"')
        if not task_id:
            raise RuntimeError("ORD query submission returned no task identifier")

        while time.time() < deadline:
            remaining = max(0.4, deadline - time.time())
            poll_resp = _ORD_HTTP_SESSION.get(
                f"{ORD_ONLINE_API_BASE}/fetch_query_result",
                params={"task_id": task_id},
                timeout=min(1.5, remaining),
            )
            if poll_resp.status_code == 200:
                items = poll_resp.json()
                if not isinstance(items, list):
                    raise RuntimeError("ORD query result was not a list")
                out_protos = []
                for item in items:
                    b64_proto = item.get("proto") if isinstance(item, dict) else None
                    if b64_proto:
                        try:
                            out_protos.append(base64.b64decode(b64_proto))
                        except Exception:
                            continue
                return out_protos
            elif poll_resp.status_code in (202, 404):
                time.sleep(poll_interval)
            else:
                raise RuntimeError(f"ORD query polling returned HTTP {poll_resp.status_code}")
    except requests.RequestException as exc:
        raise RuntimeError(f"ORD query request failed: {exc}") from exc
    raise TimeoutError("ORD query timed out before returning a completed result")


def _fetch_step_ord_records_worker(
    forward_smiles: str,
    limit: int = 20,
    similarity_threshold: float = 0.40,
    per_step_timeout: float = 6.0,
    core_product: str = "",
    core_reactant: str = "",
) -> List[bytes]:
    """
    Worker function executed inside ThreadPoolExecutor.
    Performs pure HTTP network I/O ONLY and returns raw protobuf bytes
    (avoids multi-threaded C++ Protobuf/RDKit segfaults).
    """
    s = (forward_smiles or "").strip()
    if ">>" not in s:
        return []

    react_s, prod_s = [p.strip() for p in s.split(">>", 1)]
    r_list = [x.strip() for x in react_s.split(".") if x.strip()][:2]
    p_smi = _extract_primary_smiles(prod_s)
    if not p_smi:
        return []
    if not core_reactant:
        core_reactant = _rdkit_extract_reacting_cores(s)["core_reactant"]

    t0 = time.time()
    tier1_params = [
        ("component", f"{p_smi};output;similarity"),
        ("similarity", str(similarity_threshold)),
        ("use_stereochemistry", "false"),
        ("limit", str(limit)),
    ]
    if core_reactant or r_list:
        tier1_params.append(("component", f"{core_reactant or r_list[0]};input;similarity"))

    raw_protos = _submit_and_poll_ord_session(
        tier1_params,
        timeout_sec=per_step_timeout,
        poll_interval=0.25,
    )

    elapsed = time.time() - t0
    if not raw_protos and core_product and elapsed < (per_step_timeout - 1.0):
        fallback_threshold = max(0.30, similarity_threshold - 0.10)
        tier2_params = [
            ("component", f"{core_product};output;similarity"),
            ("similarity", f"{fallback_threshold:.2f}"),
            ("use_stereochemistry", "false"),
            ("limit", str(limit)),
        ]
        raw_protos = _submit_and_poll_ord_session(
            tier2_params,
            timeout_sec=max(1.0, per_step_timeout - elapsed),
            poll_interval=0.25,
        )

    return raw_protos


class ORDRetrosynthesisValidator:
    """
    Validates proposed retrosynthesis steps against local ORD shards and live online ORD queries:
      1. Exact connectivity/DRFP retrieval, then SMARTS/ΔFG chemical admission and ranking
      2. Empirical Feasibility & 'Negative Data' Filtering (Yield stats, HTE <15% failure rate, ORD side-products)
      3. Forward Condition Grounding (Consensus catalyst, reagent, solvent, temperature, atmosphere, DOI)
      4. Route-Level Composite Scoring (ORD Empirical Confidence Score)
    """

    def __init__(self, db_path: str = ORD_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._scanned_shards_by_gate = {}
        self._loaded_shards = set()
        self._online_queried_steps = set()
        self.cache_identity = uuid4().hex
        self.index_revision = 0
        self._validate_cached = functools.lru_cache(maxsize=256)(self._validate_step)
        self._init_db_and_seed()
        self._refresh_index()

    def _init_db_and_seed(self):
        con = duckdb.connect(self.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS ord_reactions (
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
        for column in ("procedure_details", "doi", "patent_id", "condition_stoichiometry"):
            con.execute(f"ALTER TABLE ord_reactions ADD COLUMN IF NOT EXISTS {column} VARCHAR DEFAULT ''")
        con.execute("ALTER TABLE ord_reactions ADD COLUMN IF NOT EXISTS duration_h DOUBLE")
        con.execute("ALTER TABLE ord_reactions ADD COLUMN IF NOT EXISTS drfp_bytes BLOB")
        con.execute("ALTER TABLE ord_reactions ADD COLUMN IF NOT EXISTS diff_fp_bytes BLOB")
        con.execute("ALTER TABLE ord_reactions ADD COLUMN IF NOT EXISTS consumed_fgs VARCHAR DEFAULT '[]'")
        con.execute("ALTER TABLE ord_reactions ADD COLUMN IF NOT EXISTS formed_fgs VARCHAR DEFAULT '[]'")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ord_conn ON ord_reactions (prod_connectivity)")
        # Purge any legacy hardcoded seed records so only real ORD reactions exist in DuckDB
        con.execute("""
            DELETE FROM ord_reactions WHERE ord_id IN (
                'ord-9f4b21a08c114e2a91b2', 'ord-7e3c90b12a444d8f82a1', 'ord-2a1d88f09c314b1e77c9',
                'ord-6b8e11c45d904a3f90d4', 'ord-c491e02b8a114f3d92e5', 'ord-4c2f77e19b024c5d81e3',
                'ord-8d5a33f90e124f6b71c8', 'ord-5a9c12d84f014e7a63b2', 'ord-1b7e44a20c884d9c55f1',
                'ord-3f6d99c11a234b4e88a7', 'ord-0e8b55d31c774a2f66d0'
            )
        """)
        self._backfill_missing_features(con)
        con.close()

    def _backfill_missing_features(self, con):
        """Populate DRFP and functional-group features for legacy rows in bounded batches."""
        last_ord_id = ""
        while True:
            rows = con.execute(
                """
                SELECT ord_id, rxn_smiles
                FROM ord_reactions
                WHERE ord_id > ? AND (
                    drfp_bytes IS NULL OR diff_fp_bytes IS NULL
                    OR consumed_fgs IS NULL OR formed_fgs IS NULL
                )
                ORDER BY ord_id
                LIMIT 256
                """,
                [last_ord_id],
            ).fetchall()
            if not rows:
                return
            for ord_id, rxn_smiles in rows:
                diff_fp = compute_rxn_center_diff_fp(rxn_smiles or "")
                consumed, formed, _ = compute_rdkit_internal_delta_fg(rxn_smiles or "")
                consumed_fgs = sorted(consumed.elements()) if hasattr(consumed, "elements") else sorted(consumed)
                formed_fgs = sorted(formed.elements()) if hasattr(formed, "elements") else sorted(formed)
                con.execute(
                    """
                    UPDATE ord_reactions
                    SET drfp_bytes = ?, diff_fp_bytes = ?, consumed_fgs = ?, formed_fgs = ?
                    WHERE ord_id = ?
                    """,
                    [compute_drfp_bytes(rxn_smiles or ""),
                     diff_fp.ToBinary() if diff_fp is not None else None,
                     json.dumps(consumed_fgs), json.dumps(formed_fgs), ord_id],
                )
            last_ord_id = rows[-1][0]

    def _refresh_index(self):
        with duckdb.connect(self.db_path, read_only=True) as con:
            self._index_size = con.execute("SELECT count(*) FROM ord_reactions").fetchone()[0]
            rows = con.execute(
                "SELECT ord_id, drfp_bytes FROM ord_reactions WHERE drfp_bytes IS NOT NULL ORDER BY ord_id"
            ).fetchall()
            has_cbyd = con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'cbyd_reactions'"
            ).fetchone()[0] > 0
            if has_cbyd:
                self._cbyd_index_size = con.execute("SELECT count(*) FROM cbyd_reactions").fetchone()[0]
                cbyd_rows = con.execute(
                    "SELECT reaction_id, drfp_bytes FROM cbyd_reactions "
                    "WHERE drfp_bytes IS NOT NULL ORDER BY reaction_id"
                ).fetchall()
            else:
                self._cbyd_index_size = 0
                cbyd_rows = []
        indexed = [
            (ord_id, np.frombuffer(bytes(fingerprint), dtype=np.uint8).copy())
            for ord_id, fingerprint in rows
            if fingerprint and len(fingerprint) == 256
        ]
        self._drfp_ord_ids = np.asarray([ord_id for ord_id, _ in indexed], dtype=object)
        self._drfp_matrix = (
            np.stack([fingerprint for _, fingerprint in indexed]).astype(np.uint8, copy=False)
            if indexed
            else np.empty((0, 256), dtype=np.uint8)
        )
        cbyd_indexed = [
            (reaction_id, np.frombuffer(bytes(fingerprint), dtype=np.uint8).copy())
            for reaction_id, fingerprint in cbyd_rows
            if fingerprint and len(fingerprint) == 256
        ]
        self._drfp_cbyd_ids = np.asarray([reaction_id for reaction_id, _ in cbyd_indexed], dtype=object)
        self._drfp_cbyd_matrix = (
            np.stack([fingerprint for _, fingerprint in cbyd_indexed]).astype(np.uint8, copy=False)
            if cbyd_indexed
            else np.empty((0, 256), dtype=np.uint8)
        )
        self.index_revision += 1
        self._validate_cached.cache_clear()

    def get_index_size(self) -> int:
        return self._index_size

    def get_cbyd_index_size(self) -> int:
        return self._cbyd_index_size

    def _query_cbyd_precedents(
        self, s, prod_conn, tiers, delta, query_drfp, query_bits, *, top_k=50
    ) -> tuple[list, int]:
        """Retrieve and chemically gate CbyD candidates, then rerank by spectator overlap."""
        if not self._cbyd_index_size:
            return [], 0
        candidate_ids, similarities = set(), {}
        with duckdb.connect(self.db_path, read_only=True) as con:
            exact_ids = set()
            if prod_conn:
                exact_ids = {
                    row[0] for row in con.execute(
                        "SELECT reaction_id FROM cbyd_reactions WHERE prod_connectivity = ?",
                        [prod_conn],
                    ).fetchall()
                }
                candidate_ids.update(exact_ids)
            if self._cbyd_index_size <= 300:
                candidate_ids.update(row[0] for row in con.execute(
                    "SELECT reaction_id FROM cbyd_reactions"
                ).fetchall())
            elif query_drfp and self._drfp_cbyd_matrix.size:
                scores = tanimoto_packed_fp_batch(query_drfp, self._drfp_cbyd_matrix)
                eligible = np.flatnonzero(scores >= 0.10)
                ranked = sorted(eligible, key=lambda idx: (-float(scores[idx]), str(self._drfp_cbyd_ids[idx])))[:200]
                for idx in ranked:
                    reaction_id = str(self._drfp_cbyd_ids[idx])
                    candidate_ids.add(reaction_id)
                    similarities[reaction_id] = float(scores[idx])
            if not candidate_ids:
                return [], 0
            placeholders = ",".join("?" for _ in candidate_ids)
            cursor = con.execute(
                f"SELECT * FROM cbyd_reactions WHERE reaction_id IN ({placeholders})",
                sorted(candidate_ids),
            )
            columns = [col[0] for col in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

        query_spectators = compute_spectator_functional_groups(s)
        query_tags = set(query_spectators["survived"])
        admitted = []
        for rec in rows:
            rxn = rec.get("canonical_reaction") or rec.get("rxn_smiles") or ""
            gate = _match_ord_gate(
                *_parse_reaction_mols(rxn),
                (_counter_from_json(rec.get("consumed_fgs")), _counter_from_json(rec.get("formed_fgs")), Counter()),
                tiers,
                delta,
            )
            if not gate:
                continue
            label, base = gate
            bits = int.from_bytes(rec.get("fp_bytes") or compute_folded_rxn_fp(rxn) or b"", "big")
            union = (query_bits | bits).bit_count()
            structural = (query_bits & bits).bit_count() / union if union else 0.0
            drfp = similarities.get(rec["reaction_id"])
            if drfp is None and query_drfp and rec.get("drfp_bytes"):
                drfp = float(tanimoto_packed_fp_batch(
                    query_drfp,
                    np.frombuffer(bytes(rec["drfp_bytes"]), dtype=np.uint8).reshape(1, -1),
                )[0])
            drfp = drfp or 0.0
            candidate_tags = set(_json_value(rec.get("survived_fgs"), {}))
            overlap = len(query_tags & candidate_tags) / len(query_tags) if query_tags else 0.0
            transformation_similarity = 0.45 * base + 0.55 * (0.50 * structural + 0.50 * drfp)
            rec.update({
                "ord_id": rec["reaction_id"],
                "precedent_id": rec["reaction_id"],
                "provider": "CbyD",
                "provenance": "CbyD (Peer-Reviewed)",
                "similarity": 0.85 * transformation_similarity + 0.15 * overlap,
                "transformation_similarity": transformation_similarity,
                "spectator_overlap": round(overlap, 3),
                "tolerated_functional_groups": sorted(candidate_tags),
                "matched_smarts_label": label,
                "reagents": rec.get("general_reagents") or "None",
            })
            admitted.append(rec)
        admitted.sort(key=lambda r: (-r["similarity"], r["reaction_id"]))
        return admitted[:top_k], len(exact_ids & {r["reaction_id"] for r in admitted})

    def batch_prefetch_online_ord(
        self,
        forward_smiles_list: List[str],
        max_workers: int = 4,
        limit_per_step: int = 20,
        wall_timeout_sec: float = 15.0,
        *, route_steps=None,
    ) -> dict:
        """
        Parallelizes online ORD lookups across all unique reaction steps in a retrosynthesis run:
          1. Filters out steps that already have strong local DuckDB precedents (sim >= 0.55)
          2. Fires concurrent HTTP queries via ThreadPoolExecutor (up to 16 threads)
          3. Commits all newly discovered Reaction records in one DuckDB transaction
        """
        specs = _ord_gate_specs(forward_smiles_list, route_steps)
        if not specs:
            return {"queried_steps": 0, "added_records": 0, "elapsed_sec": 0.0, "indexed_count": self.get_index_size()}

        t_start = time.time()
        steps_to_fetch = set()
        for smi, template, mapped in specs:
            if smi in self._online_queried_steps:
                continue
            local_check = self._validate_cached(smi, 1, template, mapped)
            if local_check.get("max_rxn_similarity", 0.0) < 0.55 and local_check.get("exact_product_hits", 0) == 0:
                steps_to_fetch.add(smi)

        if not steps_to_fetch:
            hf_added = self._auto_sync_hf_ord_if_sparse(forward_smiles_list, route_steps)
            return {
                "queried_steps": 0,
                "added_records": hf_added,
                "elapsed_sec": round(time.time() - t_start, 3),
                "indexed_count": self.get_index_size(),
            }

        raw_proto_batches = []
        n_workers = min(max_workers, len(steps_to_fetch))
        per_step_budget = min(6.0, max(2.0, wall_timeout_sec - 0.5))
        cores = {smi: _rdkit_extract_reacting_cores(smi) for smi in steps_to_fetch}

        executor = ThreadPoolExecutor(max_workers=n_workers)
        future_to_smi = {
            executor.submit(
                _fetch_step_ord_records_worker,
                smi,
                limit_per_step,
                0.40,
                per_step_budget,
                cores[smi]["core_product_scaffold"],
                cores[smi]["core_reactant"],
            ): smi
            for smi in steps_to_fetch
        }

        try:
            for fut in as_completed(future_to_smi, timeout=wall_timeout_sec):
                try:
                    pb_list = fut.result(timeout=0.05)
                    if pb_list:
                        raw_proto_batches.extend(pb_list)
                    self._online_queried_steps.add(future_to_smi[fut])
                except Exception as exc:
                    logging.warning("Online ORD lookup for %r failed: %s", future_to_smi[fut], exc)
                    continue
        except Exception:
            pass
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        # Parse Protobufs and run RDKit on the main thread ONLY (thread-safe)
        all_new_records = {}
        for pb_bytes in raw_proto_batches:
            try:
                rxn_pb = ord_reaction_pb2.Reaction()
                rxn_pb.ParseFromString(pb_bytes)
                rec = parse_ord_reaction_protobuf(rxn_pb)
                if rec and rec.get("ord_id") and rec["ord_id"] not in all_new_records:
                    all_new_records[rec["ord_id"]] = rec
            except Exception:
                continue

        added_cnt = 0
        if all_new_records:
            con = duckdb.connect(self.db_path)
            df_bulk = pd.DataFrame(list(all_new_records.values()))
            con.execute("INSERT OR REPLACE INTO ord_reactions BY NAME SELECT * FROM df_bulk")
            con.close()
            self._refresh_index()
            added_cnt = len(all_new_records)

        # Auto-index real ORD reactions from Hugging Face when the local DuckDB index is empty/sparse
        added_cnt += self._auto_sync_hf_ord_if_sparse(forward_smiles_list, route_steps)

        return {
            "queried_steps": len(steps_to_fetch),
            "added_records": added_cnt,
            "elapsed_sec": round(time.time() - t_start, 2),
            "indexed_count": self.get_index_size(),
        }

    def _auto_sync_hf_ord_if_sparse(self, forward_smiles_list: List[str], route_steps=None) -> int:
        """Stream bounded shards when one or more active queries lack any gated analogs."""
        specs = _ord_gate_specs(forward_smiles_list, route_steps)
        if all(self._validate_cached(smi, 1, template, mapped)["smarts_matched_count"] for smi, template, mapped in specs):
            return 0
        try:
            hf_sync = self.sync_huggingface_parquet_shards(
                max_shards=3,
                max_reactions_per_shard=4000,
                forward_smiles_list=forward_smiles_list,
                route_steps=route_steps,
            )
        except Exception:
            return 0
        return int(hf_sync.get("added", 0) or 0)

    def sync_huggingface_parquet_shards(self, max_shards: int = 3, max_reactions_per_shard: int = 5000,
                                        custom_smarts_list: Optional[List[str]] = None,
                                        forward_smiles_list: Optional[List[str]] = None, *, route_steps=None) -> dict:
        """Index bounded ORD protobuf shards, filtering to the active steps when available."""
        gates = [(build_progressive_reaction_smarts(smi, template, mapped), compute_rdkit_internal_delta_fg(smi))
                 for smi, template, mapped in _ord_gate_specs(forward_smiles_list, route_steps) if _parse_reaction_mols(smi)[0]]
        custom_smarts = tuple(sorted(set(custom_smarts_list or [])))
        custom_queries = [q for sm in custom_smarts if (q := _compile_rxn_smarts_queries(sm))]
        # A filtered scan only covers its own gate. A different target may need the same shard.
        gate_key = (frozenset((tuple(t['smarts'] for t in tiers), _delta_cache_key(delta)) for tiers, delta in gates),
                    custom_smarts, max_reactions_per_shard)
        if gate_key not in self._scanned_shards_by_gate and len(self._scanned_shards_by_gate) >= 32:
            del self._scanned_shards_by_gate[next(iter(self._scanned_shards_by_gate))]
        scanned = self._scanned_shards_by_gate.setdefault(gate_key, set())

        def admit(smi):
            reactants, products = _parse_reaction_mols(smi)
            if not gates and not custom_queries:
                return bool(reactants)  # Manual sync before a route exists; validation will still gate these rows.
            delta = compute_rdkit_internal_delta_fg(smi)
            return any(_match_ord_gate(reactants, products, delta, tiers, query_delta) for tiers, query_delta in gates) or any(
                matches_reaction_smarts_mols(reactants, products, q) for q in custom_queries
            )

        repo_id = "open-reaction-database/ord-data"
        token = get_secret("HF_TOKEN")
        disable_progress_bars()
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        try:
            all_files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
            parquet_files = [f for f in all_files if f.endswith(".parquet")]
            protobuf_files = [f for f in all_files if f.endswith((".pb.gz", ".pb"))]
            if parquet_files:
                logging.warning(
                    "Skipping %d unsupported ORD Parquet shard(s); only protobuf .pb/.pb.gz shards are loaded",
                    len(parquet_files),
                )
            # Prioritize diverse benchmark datasets (including ord_dataset-00005539a1e04c809a9a78647bea649c) first
            priority = [f for f in protobuf_files if "00005539a1e04c809a9a78647bea649c" in f]
            others = [f for f in protobuf_files if f not in priority]
            ordered_files = priority + others
            target_shards = [f for f in ordered_files if f not in scanned][:max(1, int(max_shards))]
        except Exception as e:
            return {"error": f"Hugging Face Hub query failed: {e}", "indexed_count": self.get_index_size()}

        con = duckdb.connect(self.db_path)
        added_total = 0
        for shard in target_shards:
            try:
                local_path = hf_hub_download(repo_id=repo_id, filename=shard, repo_type="dataset", token=token)
                dataset = ord_datasets.load_dataset(local_path)
                records = {}
                rxn_iter = getattr(dataset, "reactions", None)
                if rxn_iter is None and hasattr(dataset, "iter_reactions"):
                    rxn_iter = (r for _, r in dataset.iter_reactions())
                for idx, rxn in enumerate(rxn_iter or []):
                    if idx >= max_reactions_per_shard:
                        break
                    parsed = parse_ord_reaction_protobuf(rxn, admit=admit)
                    if parsed:
                        records[parsed["ord_id"]] = parsed
                if records:
                    df_new = pd.DataFrame(list(records.values()))
                    con.execute("INSERT OR REPLACE INTO ord_reactions BY NAME SELECT * FROM df_new")
                    added_total += len(records)
                scanned.add(shard)
                self._loaded_shards.add(shard)
            except Exception as exc:
                logging.warning("Skipping ORD shard %s: %s", shard, exc)
                continue
        con.close()
        if added_total:
            self._refresh_index()
        return {
            "added": added_total,
            "indexed_count": self.get_index_size(),
            "shards_loaded": len(self._loaded_shards),
        }

    def validate_step(self, forward_smiles: str, top_k: int = 3, *, template_info=None) -> dict:
        s = (forward_smiles or "").strip()
        info = template_info or {}
        return self._validate_cached(s, top_k, info.get("retro_template", ""), info.get("mapped_forward_smiles", ""))

    def _validate_step(self, s, top_k, template, mapped):
        """Run exact, DRFP, chemical-gate, and empirical-ranking validation stages."""
        _, products = _parse_reaction_mols(s)
        primary_product = Chem.MolFromSmiles(_extract_primary_smiles(s))
        prod_conn = Chem.MolToInchiKey(primary_product).split("-")[0] if primary_product else ""
        tiers = build_progressive_reaction_smarts(s, template, mapped) if products else []
        delta = compute_rdkit_internal_delta_fg(s)
        functional_group_audit = compute_spectator_functional_groups(s)
        matched_smarts_label, best_gate_base, exact_hits = "No verified match (Unvalidated)", 0.0, 0
        analog_records = []
        if tiers or delta[0] or delta[1]:
            query_drfp = compute_drfp_bytes(s)
            q_bits = int.from_bytes(compute_folded_rxn_fp(s) or b"", "big")
            q_diff = None
            exact_ord_ids, candidate_ord_ids = set(), set()
            drfp_similarity_by_id = {}

            with duckdb.connect(self.db_path, read_only=True) as con:
                if prod_conn:
                    exact_ord_ids = {
                        row[0]
                        for row in con.execute(
                            "SELECT ord_id FROM ord_reactions WHERE prod_connectivity = ?",
                            [prod_conn],
                        ).fetchall()
                    }
                candidate_ord_ids.update(exact_ord_ids)

                if self._index_size <= 300:
                    candidate_ord_ids.update(
                        row[0] for row in con.execute("SELECT ord_id FROM ord_reactions").fetchall()
                    )
                elif query_drfp and self._drfp_matrix.size:
                    similarities = tanimoto_packed_fp_batch(query_drfp, self._drfp_matrix)
                    eligible = np.flatnonzero(similarities >= 0.15)
                    ranked = sorted(
                        eligible,
                        key=lambda idx: (-float(similarities[idx]), str(self._drfp_ord_ids[idx])),
                    )[:200]
                    for idx in ranked:
                        ord_id = str(self._drfp_ord_ids[idx])
                        candidate_ord_ids.add(ord_id)
                        drfp_similarity_by_id[ord_id] = float(similarities[idx])

                if candidate_ord_ids:
                    placeholders = ",".join("?" for _ in candidate_ord_ids)
                    cursor = con.execute(
                        f"SELECT * FROM ord_reactions WHERE ord_id IN ({placeholders})",
                        sorted(candidate_ord_ids),
                    )
                    columns = [col[0] for col in cursor.description]
                    candidate_rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                else:
                    candidate_rows = []

            for rec in candidate_rows:
                rxn = rec["rxn_smiles"] or ""
                candidate_delta = (
                    _counter_from_json(rec.get("consumed_fgs")),
                    _counter_from_json(rec.get("formed_fgs")),
                    Counter(),
                )
                gate = _match_ord_gate(
                    *_parse_reaction_mols(rxn), candidate_delta, tiers, delta
                )
                if not gate:
                    continue
                label, base = gate
                if base > best_gate_base or (
                    base == best_gate_base and matched_smarts_label == "RDKit ΔFG match"
                ):
                    matched_smarts_label, best_gate_base = label, base
                exact_hits += rec["ord_id"] in exact_ord_ids

                c_bits = int.from_bytes(rec["fp_bytes"] or compute_folded_rxn_fp(rxn) or b"", "big")
                union = (q_bits | c_bits).bit_count()
                struct_sim = (q_bits & c_bits).bit_count() / union if union else 0.0

                drfp_sim = drfp_similarity_by_id.get(rec["ord_id"])
                if drfp_sim is None and query_drfp and rec.get("drfp_bytes"):
                    drfp_sim = float(tanimoto_packed_fp_batch(
                        query_drfp, np.frombuffer(bytes(rec["drfp_bytes"]), dtype=np.uint8).reshape(1, -1)
                    )[0])
                if drfp_sim is None:
                    if q_diff is None:
                        q_diff = compute_rxn_center_diff_fp(s)
                    c_diff = None
                    if q_diff is not None and rec.get("diff_fp_bytes"):
                        try:
                            c_diff = DataStructs.UIntSparseIntVect(bytes(rec["diff_fp_bytes"]))
                        except (TypeError, ValueError, RuntimeError):
                            pass
                    if q_diff is not None and c_diff is None:
                        c_diff = compute_rxn_center_diff_fp(rxn)
                    drfp_sim = (
                        float(DataStructs.TanimotoSimilarity(q_diff, c_diff))
                        if c_diff is not None else 0.0
                    )

                rec["similarity"] = (
                    0.45 * base + 0.55 * (0.50 * struct_sim + 0.50 * drfp_sim)
                )
                rec["matched_smarts_label"] = label
                rec.pop("fp_bytes", None)
                rec.pop("drfp_bytes", None)
                rec.pop("diff_fp_bytes", None)
                rec["provider"] = "ORD"
                rec["provenance"] = "ORD"
                rec["precedent_id"] = rec["ord_id"]
                analog_records.append(rec)

            current_max = max((r["similarity"] for r in analog_records), default=0.0)
            if current_max < 0.55 and self._cbyd_index_size:
                cbyd_records, cbyd_exact_hits = self._query_cbyd_precedents(
                    s, prod_conn, tiers, delta, query_drfp, q_bits, top_k=50
                )
                exact_hits += cbyd_exact_hits
                analog_records.extend(cbyd_records)

        analog_records.sort(key=lambda x: (-x["similarity"], -(x["yield_pct"] if x["yield_pct"] is not None else -1.0), x["ord_id"]))
        if analog_records:
            matched_smarts_label = analog_records[0]["matched_smarts_label"]

        scoreable_yields = []
        for record in analog_records:
            value = record.get("yield_pct")
            if value is None or np.isnan(value):
                continue
            if record.get("provider") == "CbyD" and (
                int(record.get("step_count") or 1) > 1 or record.get("is_borsm")
            ):
                continue
            scoreable_yields.append(record)
        reported_yields = [float(r["yield_pct"]) for r in scoreable_yields]
        top_k_yields = [float(r["yield_pct"]) for r in scoreable_yields[: max(3, top_k)]]

        mean_yield = round(float(np.mean(top_k_yields)), 1) if top_k_yields else None
        median_yield = round(float(np.median(reported_yields)), 1) if reported_yields else None
        hte_fail_rate = (
            round(sum(1 for y in reported_yields if y < 15.0) / len(reported_yields) * 100.0, 1)
            if reported_yields else 0.0
        )

        ord_byproducts = []
        for r in analog_records[:5]:
            byp = (r.get("undesired_byproducts") or "").strip()
            if byp and byp != "None" and byp not in ord_byproducts:
                ord_byproducts.append(f"{byp} [{r['ord_id']}]")

        neg_data_warn = "None"
        if hte_fail_rate >= 30.0:
            neg_data_warn = (
                f"Evidence Negative-Data Alert: {hte_fail_rate:.0f}% of close analogs reported <15% yield "
                f"(median yield {median_yield or 0:.0f}%). Inspect catalyst/base pairing carefully."
            )
        elif mean_yield is not None and mean_yield < 45.0:
            neg_data_warn = f"Low Empirical Evidence Yield: Top analogs average {mean_yield:.1f}% yield."

        grounded_source = analog_records[0] if analog_records else None
        for rec in analog_records[:max(1, top_k)]:
            for key in ("catalysts", "reagents", "solvents"):
                rec[key] = _humanize_condition_list(rec[key])
        grounded_conditions = ""
        grounded_ref = ""
        if grounded_source:
            parts_cond = []
            if grounded_source.get("catalysts") and grounded_source["catalysts"] != "None":
                parts_cond.append(f"Cat: {grounded_source['catalysts']}")
            if grounded_source.get("reagents") and grounded_source["reagents"] != "None":
                parts_cond.append(f"Reag: {grounded_source['reagents']}")
            if grounded_source.get("solvents") and grounded_source["solvents"] != "Unspecified":
                parts_cond.append(f"Solv: {grounded_source['solvents']}")
            if grounded_source.get("temp_c") is not None and not np.isnan(grounded_source["temp_c"]):
                parts_cond.append(f"{grounded_source['temp_c']:.0f} °C")
            if grounded_source.get("atmosphere") and grounded_source["atmosphere"] not in ("UNSPECIFIED", "None"):
                parts_cond.append(f"{grounded_source['atmosphere']}")
            grounded_conditions = ", ".join(parts_cond)
            reference = grounded_source.get("reference") or grounded_source.get("citation") or grounded_source["provenance"]
            grounded_ref = (
                f"{grounded_source['ord_id']} ({grounded_source.get('yield_pct', 'N/A')}% yield, "
                f"{grounded_source.get('provenance', 'ORD')}: {reference})"
            )

        max_sim = float(analog_records[0]["similarity"]) if analog_records else 0.0
        if analog_records:
            norm_yield = (mean_yield / 100.0) if mean_yield is not None else 0.60
            fail_penalty_factor = 1.0 - 0.35 * ((hte_fail_rate / 100.0) if hte_fail_rate else 0.0)
            raw_step_conf = (0.60 * max_sim + 0.40 * norm_yield) * fail_penalty_factor
            step_ord_confidence = round(float(min(0.99, max(0.21, raw_step_conf))), 3)
        else:
            step_ord_confidence = 0.20

        return {
            "is_validated": bool(analog_records),
            "exact_product_hits": exact_hits,
            "smarts_matched_count": len(analog_records),
            "cbyd_matched_count": sum(1 for record in analog_records if record.get("provider") == "CbyD"),
            "matched_smarts_label": matched_smarts_label,
            "rdkit_delta_fg_signature": dict(zip(
                ("consumed_fgs", "formed_fgs", "reactant_fgs"),
                (sorted(Counter(groups).elements()) for groups in delta),
            )),
            "functional_group_audit": {
                "reactant_groups": sorted(functional_group_audit["reactant"]),
                "product_groups": sorted(functional_group_audit["product"]),
                "survived_non_reacting_groups": sorted(functional_group_audit["survived"]),
            },
            "max_rxn_similarity": round(max_sim, 3),
            "mean_analog_yield_pct": mean_yield,
            "median_analog_yield_pct": median_yield,
            "hte_failure_rate_pct": hte_fail_rate,
            "negative_data_warning": neg_data_warn,
            "ord_recorded_byproducts": ord_byproducts,
            "grounded_conditions_consensus": grounded_conditions,
            "grounded_reference_summary": grounded_ref,
            "step_ord_confidence": step_ord_confidence,
            "top_precedents": [
                {
                    "ord_id": r["ord_id"],
                    "precedent_id": r.get("precedent_id", r["ord_id"]),
                    "provider": r.get("provider", "ORD"),
                    "provenance": r.get("provenance", "ORD"),
                    "rxn_smiles": r.get("rxn_smiles", ""),
                    "similarity": round(r["similarity"], 3),
                    "transformation_similarity": round(r.get("transformation_similarity", r["similarity"]), 3),
                    "spectator_overlap": r.get("spectator_overlap", 0.0),
                    "matched_smarts_label": r["matched_smarts_label"],
                    "yield_pct": None if (r.get("yield_pct") is None or np.isnan(r["yield_pct"])) else round(float(r["yield_pct"]), 1),
                    "catalysts": r.get("catalysts", "None"),
                    "reagents": r.get("reagents", "None"),
                    "solvents": r.get("solvents", "Unspecified"),
                    "ligands": r.get("ligands", ""),
                    "modifiers": r.get("modifiers", ""),
                    "temperature": r.get("temperature", ""),
                    "yield_raw": r.get("yield_raw", ""),
                    "step_count": int(r.get("step_count") or 1),
                    "is_borsm": bool(r.get("is_borsm") or False),
                    "dr": r.get("dr", ""),
                    "ee": r.get("ee", ""),
                    "er": r.get("er", ""),
                    "tolerated_functional_groups": r.get("tolerated_functional_groups", []),
                    "hazard_flags": _json_value(r.get("hazard_flags"), []),
                    "safer_alternatives": _json_value(r.get("safer_alternatives"), []),
                    "temp_c": None if (r.get("temp_c") is None or np.isnan(r["temp_c"])) else round(float(r["temp_c"]), 1),
                    "atmosphere": r.get("atmosphere", "UNSPECIFIED"),
                    "reference": r.get("reference") or r.get("citation") or "ORD",
                    "citation": r.get("citation", ""),
                    "mapping_confidence": r.get("mapping_confidence"),
                    "procedure_details": r.get("procedure_details", ""),
                    "doi": r.get("doi") or (
                        r.get("reference", "")[5:]
                        if (r.get("reference") or "").startswith("DOI: ")
                        else ""
                    ),
                    "patent_id": r.get("patent_id") or (
                        r.get("reference", "")[8:]
                        if (r.get("reference") or "").startswith("Patent: ")
                        else ""
                    ),
                    "reagent_stoichiometry": json.loads(r.get("condition_stoichiometry") or "[]"),
                    "duration_h": r.get("duration_h"),
                }
                for r in analog_records[:top_k]
            ],
        }

    def score_route(self, forward_smiles_steps: List[str], *, step_reports=None) -> dict:
        """
        Computes the Route-Level ORD Empirical Confidence Score across all steps:
          S_ORD_route = prod_{i=1}^N max(0.05, S_step(i))
          plus geometric step mean and bottleneck step identification.
        """
        if not forward_smiles_steps:
            return {
                "ord_empirical_confidence": 0.0,
                "ord_geometric_confidence": 0.0,
                "ord_confidence_100": 0,
                "bottleneck_step_index": 1,
                "ord_mean_yield_pct": None,
                "ord_max_hte_fail_pct": 0.0,
                "step_validations": [],
            }
        if step_reports is None:
            step_reports = [self.validate_step(smi) for smi in forward_smiles_steps]
        confs = [max(0.05, float(r["step_ord_confidence"])) for r in step_reports]

        prod_score = float(np.prod(confs))
        geom_score = float(np.exp(np.mean(np.log(confs))))

        yields = [r["mean_analog_yield_pct"] for r in step_reports if r.get("mean_analog_yield_pct") is not None]
        fails = [r.get("hte_failure_rate_pct", 0.0) for r in step_reports]
        bottleneck_idx = int(np.argmin(confs)) + 1

        return {
            "ord_empirical_confidence": round(prod_score, 4),
            "ord_geometric_confidence": round(geom_score, 4),
            "ord_confidence_100": int(round(geom_score * 100)),
            "bottleneck_step_index": bottleneck_idx,
            "ord_mean_yield_pct": round(float(np.mean(yields)), 1) if yields else None,
            "ord_max_hte_fail_pct": round(float(max(fails)) if fails else 0.0, 1),
            "step_validations": step_reports,
        }


@functools.lru_cache(maxsize=1)
def get_ord_validator() -> ORDRetrosynthesisValidator:
    """Create the default ORD index on first use, not at import time."""
    return ORDRetrosynthesisValidator(ORD_DB_PATH)
