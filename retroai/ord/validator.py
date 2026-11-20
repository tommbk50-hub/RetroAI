"""Index ORD precedents in DuckDB and validate route steps."""

import base64
import functools
import json
import logging
import os
import time
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
    compute_folded_rxn_fp,
    compute_rdkit_internal_delta_fg,
    compute_rxn_center_diff_fp,
    matches_reaction_smarts_mols,
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
            return []
        try:
            payload = sub_resp.json()
            task_id = payload.get("task_id", payload) if isinstance(payload, dict) else str(payload)
        except Exception:
            task_id = sub_resp.text.strip().strip('"')
        if not task_id:
            return []

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
                    return []
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
                break
    except Exception:
        pass
    return []


def _fetch_step_ord_records_worker(
    forward_smiles: str,
    limit: int = 20,
    similarity_threshold: float = 0.40,
    per_step_timeout: float = 6.0,
    core_product: str = "",
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

    t0 = time.time()
    tier1_params = [
        ("component", f"{p_smi};output;similarity"),
        ("similarity", str(similarity_threshold)),
        ("use_stereochemistry", "false"),
        ("limit", str(limit)),
    ]
    if r_list:
        tier1_params.append(("component", f"{r_list[0]};input;similarity"))

    raw_protos = _submit_and_poll_ord_session(
        tier1_params,
        timeout_sec=per_step_timeout,
        poll_interval=0.25,
    )

    elapsed = time.time() - t0
    if not raw_protos and core_product and elapsed < (per_step_timeout - 1.0):
        tier2_params = [
            ("component", f"{core_product};output;similarity"),
            ("similarity", str(max(0.32, similarity_threshold - 0.08))),
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
    Validates proposed retrosynthesis steps against ORD Parquet shards & live online ORD queries:
      1. SMARTS/ΔFG admission, then structural/difference fingerprint ranking and yield tie-breaks
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
        con.close()

    def _refresh_index(self):
        with duckdb.connect(self.db_path, read_only=True) as con:
            self._index_size = con.execute("SELECT count(*) FROM ord_reactions").fetchone()[0]
        self.index_revision += 1
        self._validate_cached.cache_clear()

    def get_index_size(self) -> int:
        return self._index_size

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
                self._online_queried_steps.add(smi)

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
        core_products = {smi: _rdkit_extract_reacting_cores(smi)["core_product"] for smi in steps_to_fetch}

        executor = ThreadPoolExecutor(max_workers=n_workers)
        future_to_smi = {
            executor.submit(
                _fetch_step_ord_records_worker,
                smi,
                limit_per_step,
                0.40,
                per_step_budget,
                core_products[smi],
            ): smi
            for smi in steps_to_fetch
        }

        try:
            for fut in as_completed(future_to_smi, timeout=wall_timeout_sec):
                try:
                    pb_list = fut.result(timeout=0.05)
                    if pb_list:
                        raw_proto_batches.extend(pb_list)
                except Exception:
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
        """Index bounded ORD shards, filtering to the active steps when available."""
        gates = [(build_progressive_reaction_smarts(smi, template, mapped), compute_rdkit_internal_delta_fg(smi))
                 for smi, template, mapped in _ord_gate_specs(forward_smiles_list, route_steps) if _parse_reaction_mols(smi)[0]]
        custom_smarts = tuple(sorted(set(custom_smarts_list or [])))
        custom_queries = [q for sm in custom_smarts if (q := _compile_rxn_smarts_queries(sm))]
        # A filtered scan only covers its own gate. A different target may need the same shard.
        gate_key = (frozenset((tuple(t['smarts'] for t in tiers), delta[:2]) for tiers, delta in gates),
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
            parquet_files = [f for f in all_files if f.endswith((".parquet", ".pb.gz", ".pb"))]
            # Prioritize diverse benchmark datasets (including ord_dataset-00005539a1e04c809a9a78647bea649c) first
            priority = [f for f in parquet_files if "00005539a1e04c809a9a78647bea649c" in f]
            others = [f for f in parquet_files if f not in priority]
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
            except Exception:
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
        """Admit via SMARTS/ΔFG first; only then use fingerprints and yield to rank."""
        _, products = _parse_reaction_mols(s)
        prod_conn = Chem.MolToInchiKey(products[0]).split("-")[0] if products else ""
        tiers = build_progressive_reaction_smarts(s, template, mapped) if products else []
        delta = compute_rdkit_internal_delta_fg(s)
        matched_smarts_label, best_gate_base, exact_hits = "No verified match", 0.0, 0
        analog_records = []
        if tiers or (delta[0] and delta[1]):
            q_bits = int.from_bytes(compute_folded_rxn_fp(s) or b"", "big")
            q_diff = compute_rxn_center_diff_fp(s)
            with duckdb.connect(self.db_path, read_only=True) as con:
                cursor = con.execute("SELECT * FROM ord_reactions")
                columns = [col[0] for col in cursor.description]
                while rows := cursor.fetchmany(256):
                    for row in rows:
                        rec = dict(zip(columns, row))
                        rxn = rec["rxn_smiles"] or ""
                        gate = _match_ord_gate(*_parse_reaction_mols(rxn), compute_rdkit_internal_delta_fg(rxn), tiers, delta)
                        if not gate:
                            continue
                        label, base = gate
                        if base > best_gate_base or (base == best_gate_base and matched_smarts_label == "RDKit ΔFG match"):
                            matched_smarts_label, best_gate_base = label, base
                        exact_hits += bool(prod_conn and rec["prod_connectivity"] == prod_conn)
                        c_bits = int.from_bytes(rec["fp_bytes"] or compute_folded_rxn_fp(rxn) or b"", "big")
                        union = (q_bits | c_bits).bit_count()
                        struct_sim = (q_bits & c_bits).bit_count() / union if union else 0.0
                        c_diff = compute_rxn_center_diff_fp(rxn) if q_diff is not None else None
                        diff_sim = DataStructs.TanimotoSimilarity(q_diff, c_diff) if c_diff is not None else 0.0
                        rec["similarity"] = 0.45 * base + 0.55 * (0.50 * struct_sim + 0.50 * diff_sim)
                        rec["matched_smarts_label"] = label
                        del rec["fp_bytes"]
                        analog_records.append(rec)

        analog_records.sort(key=lambda x: (-x["similarity"], -(x["yield_pct"] if x["yield_pct"] is not None else -1.0), x["ord_id"]))

        reported_yields = [float(r["yield_pct"]) for r in analog_records if r.get("yield_pct") is not None and not np.isnan(r["yield_pct"])]
        top_k_yields = [float(r["yield_pct"]) for r in analog_records[: max(3, top_k)] if r.get("yield_pct") is not None and not np.isnan(r["yield_pct"])]

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
                f"ORD Negative-Data Alert: {hte_fail_rate:.0f}% of close ORD/HTE analogs reported <15% yield "
                f"(median yield {median_yield or 0:.0f}%). Inspect catalyst/base pairing carefully."
            )
        elif mean_yield is not None and mean_yield < 45.0:
            neg_data_warn = f"Low Empirical ORD Yield: Top analogs average {mean_yield:.1f}% yield in ORD."

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
            grounded_ref = f"{grounded_source['ord_id']} ({grounded_source.get('yield_pct', 'N/A')}% yield, {grounded_source.get('reference', '')})"

        max_sim = float(analog_records[0]["similarity"]) if analog_records else 0.0
        norm_yield = (mean_yield / 100.0) if mean_yield is not None else 0.60
        fail_penalty_factor = 1.0 - 0.35 * ((hte_fail_rate / 100.0) if hte_fail_rate else 0.0)
        raw_step_conf = (0.60 * (max_sim if analog_records else 0.35) + 0.40 * norm_yield) * fail_penalty_factor
        step_ord_confidence = round(float(min(0.99, max(0.10, raw_step_conf))), 3)

        return {
            "exact_product_hits": exact_hits,
            "smarts_matched_count": len(analog_records),
            "matched_smarts_label": matched_smarts_label,
            "rdkit_delta_fg_signature": dict(zip(("consumed_fgs", "formed_fgs", "reactant_fgs"), map(sorted, delta))),
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
                    "rxn_smiles": r.get("rxn_smiles", ""),
                    "similarity": round(r["similarity"], 3),
                    "matched_smarts_label": r["matched_smarts_label"],
                    "yield_pct": None if (r.get("yield_pct") is None or np.isnan(r["yield_pct"])) else round(float(r["yield_pct"]), 1),
                    "catalysts": r.get("catalysts", "None"),
                    "reagents": r.get("reagents", "None"),
                    "solvents": r.get("solvents", "Unspecified"),
                    "temp_c": None if (r.get("temp_c") is None or np.isnan(r["temp_c"])) else round(float(r["temp_c"]), 1),
                    "atmosphere": r.get("atmosphere", "UNSPECIFIED"),
                    "reference": r.get("reference", "ORD"),
                    "procedure_details": r.get("procedure_details", ""),
                    "doi": r.get("doi") or (r["reference"][5:] if (r.get("reference") or "").startswith("DOI: ") else ""),
                    "patent_id": r.get("patent_id") or (r["reference"][8:] if (r.get("reference") or "").startswith("Patent: ") else ""),
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
