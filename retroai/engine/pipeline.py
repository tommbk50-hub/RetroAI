"""Orchestrate and score retrosynthesis routes."""

from typing import List, Optional

from rdkit import Chem

from retroai.ai.classifier import _classify_route_steps, _route_step_key, classify_step_strategy_role
from retroai.core.chem_utils import _canonical_molecule_smiles, _extract_primary_smiles, rdkit_reaction_complexity
from retroai.core.config import ACTIVE_GENAI_SETTINGS, get_process_ram_mb
from retroai.core.state import WorkbenchState
from retroai.ord.validator import ORDRetrosynthesisValidator, get_ord_validator

from .aizynth import (
    _flatten_reaction_tree,
    _harvest_uspto_templates_from_finder,
    cleanup_aizynth_memory,
    configure_finder_runtime_stock,
    get_finder,
    is_common_lab_reagent,
)

KEY_STEP_SASCORE_THRESHOLD = 0.658
KEY_STEP_SPS_THRESHOLD = 160.0


def _longest_linear_step_indices(steps: list[dict]) -> list[int]:
    by_index = {
        int(step.get("step_index")): step
        for step in steps
        if step.get("step_index") is not None
    }
    if not by_index:
        return []
    child_indices = {
        int(child)
        for step in by_index.values()
        for child in step.get("precursor_steps") or []
        if int(child) in by_index
    }
    if not child_indices and len(by_index) > 1:
        return [int(step["step_index"]) for step in steps if step.get("step_index") is not None]
    roots = [index for index in by_index if index not in child_indices] or list(by_index)
    memo = {}

    def longest_from(index):
        if index in memo:
            return memo[index]
        children = [
            int(child) for child in by_index[index].get("precursor_steps") or []
            if int(child) in by_index
        ]
        path = [index]
        if children:
            child_paths = [longest_from(child) for child in children]
            path += min(
                child_paths,
                key=lambda candidate: (-len(candidate), candidate),
            )
        memo[index] = path
        return path

    candidates = [longest_from(root) for root in roots]
    return min(candidates, key=lambda candidate: (-len(candidate), candidate)) if candidates else []


def _organic_reaction_endpoints(forward_smiles: str) -> tuple[str, str]:
    parts = (forward_smiles or "").split(">>")
    if len(parts) != 2:
        return "", ""

    def primary(side):
        molecules = []
        for token in side.split("."):
            mol = Chem.MolFromSmiles(token.strip())
            if mol is not None and any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
                molecules.append(mol)
        if not molecules:
            return ""
        return _canonical_molecule_smiles(max(molecules, key=lambda mol: mol.GetNumHeavyAtoms()))

    return primary(parts[0]), primary(parts[1])


def detect_unproductive_loops(steps: list[dict]) -> list[dict]:
    """Detect adjacent reversible protection or esterification/hydrolysis pairs."""
    by_index = {int(step["step_index"]): step for step in steps if step.get("step_index") is not None}
    pairs = [
        (int(step["step_index"]), int(child))
        for step in steps
        if step.get("step_index") is not None
        for child in step.get("precursor_steps") or []
        if int(child) in by_index
    ]
    if not pairs:
        lls = _longest_linear_step_indices(steps)
        pairs = list(zip(lls, lls[1:]))
    loops = []
    seen_pairs = set()
    for left_index, right_index in pairs:
        pair = (left_index, right_index)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        left, right = by_index[left_index], by_index[right_index]
        left_role, _ = classify_step_strategy_role(
            left.get("reaction_name", ""),
            left.get("typical_conditions", ""),
            left.get("forward_smiles", ""),
        )
        right_role, _ = classify_step_strategy_role(
            right.get("reaction_name", ""),
            right.get("typical_conditions", ""),
            right.get("forward_smiles", ""),
        )
        left_text = f"{left.get('reaction_name', '')} {left.get('typical_conditions', '')}".casefold()
        right_text = f"{right.get('reaction_name', '')} {right.get('typical_conditions', '')}".casefold()
        left_reactant, left_product = _organic_reaction_endpoints(left.get("forward_smiles", ""))
        right_reactant, right_product = _organic_reaction_endpoints(right.get("forward_smiles", ""))
        returns_to_intermediate = bool(
            left_reactant
            and left_product
            and right_reactant
            and right_product
            and left_reactant == right_product
            and left_product == right_reactant
        )
        protection_loop = (
            "deprotection" in {left_role, right_role}
            and bool({left_role, right_role} & {"protection", "protecting_acetal"})
            and returns_to_intermediate
        )
        ester_loop = (
            (("esterification" in left_text or "ester formation" in left_text) and "hydrolysis" in right_text)
            or (("esterification" in right_text or "ester formation" in right_text) and "hydrolysis" in left_text)
        )
        if protection_loop or (ester_loop and returns_to_intermediate):
            loops.append({
                "type": "protection_deprotection_loop" if protection_loop else "esterification_hydrolysis_loop",
                "step_indices": [left_index, right_index],
                "returns_to_intermediate": returns_to_intermediate,
            })
    return loops


def annotate_route_strategy(route: dict) -> dict:
    """Attach LLS complexity jumps, key-step tags, and loop penalties to a route."""
    steps = route.get("steps", [])
    lls_indices = _longest_linear_step_indices(steps)
    lls_set = set(lls_indices)
    profile = []
    for step in steps:
        role, exclusion = classify_step_strategy_role(
            step.get("reaction_name", step.get("strategy_role", "")),
            step.get("typical_conditions", ""),
            step.get("forward_smiles", ""),
        )
        metrics = rdkit_reaction_complexity(step.get("forward_smiles", ""))
        delta_sascore = metrics.get("delta_sascore")
        delta_sps = metrics.get("delta_sps")
        eligible = not exclusion and step.get("key_step_eligible", True)
        is_key = bool(
            int(step.get("step_index")) in lls_set
            and eligible
            and (
                (delta_sascore is not None and delta_sascore >= KEY_STEP_SASCORE_THRESHOLD)
                or (delta_sps is not None and delta_sps >= KEY_STEP_SPS_THRESHOLD)
            )
        )
        complexity_entry = {
            "step_index": int(step.get("step_index", 0)),
            "is_on_lls": int(step.get("step_index", 0)) in lls_set,
            "strategy_role": role,
            "key_step_eligible": eligible,
            "key_step_exclusion": exclusion,
            "is_strategic_key_step": is_key,
            **metrics,
        }
        step.update({
            "strategy_role": role,
            "key_step_eligible": eligible,
            "key_step_exclusion": exclusion,
            "is_strategic_key_step": is_key,
            "complexity": metrics,
        })
        profile.append(complexity_entry)
    profile.sort(key=lambda item: item["step_index"])
    loops = detect_unproductive_loops(steps)
    key_steps = [entry for entry in profile if entry["is_strategic_key_step"]]
    lls_profile = [entry for entry in profile if entry["is_on_lls"]]
    strengths = []
    for entry in key_steps:
        strengths.append(max(
            max(0.0, (entry["delta_sascore"] or 0.0) / KEY_STEP_SASCORE_THRESHOLD),
            max(0.0, (entry["delta_sps"] or 0.0) / KEY_STEP_SPS_THRESHOLD),
        ))
    route.update({
        "lls_step_indices": lls_indices,
        "lls_length": len(lls_indices),
        "strategic_key_step_indices": [entry["step_index"] for entry in key_steps],
        "strategic_key_step_count": len(key_steps),
        "complexity_building_score": round(
            min(1.0, sum(min(1.0, strength) for strength in strengths) / max(1, len(lls_profile))),
            4,
        ),
        "complexity_profile": profile,
        "lls_complexity_profile": lls_profile,
        "unproductive_loops": loops,
        "unproductive_loop_count": len(loops),
    })
    return route


def deduplicate_retro_state(results_dict: dict) -> dict:
    """
    [Improvement 3] Strips heavy inline SVG strings, base64 payloads, or non-JSON objects
    from stored results so RAM stays minimal for the chat agent.
    """
    if not isinstance(results_dict, dict):
        return {}
    results_dict.pop("html", None)
    results_dict.pop("preview_svg", None)
    for route in results_dict.get("routes", []):
        route.pop("svg_html", None)
        route.pop("reaction_tree", None)
        for step in route.get("steps", []):
            step.pop("svg_html", None)
            step.pop("mol_obj", None)
    return results_dict


def compute_composite_route_score(route: dict, ranking_mode: str = "balanced") -> float:
    """
    Computes multi-objective route score blending AiZynthFinder state score, step count,
    stock ratio, safety score, and the Route-Level ORD Empirical Confidence Score.
    """
    state_s = float(route.get("state_score", 0.0))
    n_steps = max(1, int(route.get("step_count", 1)))
    step_score = max(0.0, 1.0 - (n_steps - 1) * 0.12)
    safe_s = float(route.get("avg_safety_score", 4.0)) / 5.0
    sm = route.get("starting_materials", {})
    stock_ratio = sm.get("in_stock_count", 0) / max(1, sm.get("total_leaves", 1))
    ord_geom = float(route.get("ord_geometric_confidence", route.get("ord_empirical_confidence", 0.65)))
    ord_prod = float(route.get("ord_empirical_confidence", 0.50))
    ord_blend = 0.65 * ord_geom + 0.35 * ord_prod

    risk_pen = min(0.35, int(route.get("selectivity_risk_count", 0)) * 0.07)
    hte_fail_pen = min(0.20, float(route.get("ord_max_hte_fail_pct", 0.0)) / 100.0 * 0.25)
    loop_pen = min(0.24, int(route.get("unproductive_loop_count", 0)) * 0.08)
    complexity = min(1.0, float(route.get("complexity_building_score", 0.0)))

    if ranking_mode == "medchem_fast":
        raw = 0.28 * state_s + 0.35 * step_score + 0.17 * stock_ratio + 0.15 * ord_blend + 0.05 * safe_s
    elif ranking_mode == "process_scaleup":
        raw = 0.22 * state_s + 0.13 * step_score + 0.25 * stock_ratio + 0.20 * ord_blend + 0.20 * safe_s
    elif ranking_mode == "green_safe":
        raw = 0.20 * state_s + 0.12 * step_score + 0.18 * stock_ratio + 0.15 * ord_blend + 0.35 * safe_s
    elif ranking_mode in {"strategy_complexity", "strategy", "complexity_building"}:
        raw = (
            0.12 * state_s
            + 0.10 * step_score
            + 0.08 * stock_ratio
            + 0.20 * ord_blend
            + 0.10 * safe_s
            + 0.40 * complexity
        )
    else:
        raw = 0.25 * state_s + 0.15 * step_score + 0.15 * stock_ratio + 0.35 * ord_blend + 0.10 * safe_s
    return round(max(0.0, raw - risk_pen - hte_fail_pen - loop_pen), 4)


def run_retrosynthesis_pipeline(
    target_smiles: str,
    ranking_mode: str = "balanced",
    max_transforms: int = 6,
    time_limit: int = 120,
    iteration_limit: int = 100,
    return_first: bool = False,
    cutoff_cumulative: float = 0.995,
    max_routes: int = 10,
    exclude_reaction_names: Optional[List[str]] = None,
    lock_substructure_smarts: Optional[str] = None,
    target_scale_grams: float = 1.0,
    search_algorithm: str = "retrostar",
    use_gemini: bool = True,
    render_output: bool = False,
    *,
    state: WorkbenchState | None = None,
    ord_validator: ORDRetrosynthesisValidator | None = None,
) -> dict:
    """
    Executes AiZynthFinder tree search, prefetches live online ORD precedents in parallel,
    validates every step against DuckDB, and purges tree RAM immediately.
    """
    clean_target = _extract_primary_smiles(target_smiles)
    mol = Chem.MolFromSmiles(clean_target)
    if mol is None:
        return {"error": f"Invalid target SMILES: {target_smiles}", "routes": []}
    canon_target = Chem.MolToSmiles(mol)

    if state is None:
        state = WorkbenchState()
    finder = get_finder()
    if ord_validator is None:
        ord_validator = get_ord_validator()
    cleanup_aizynth_memory(finder, trim_ipython_history=False, state=state)
    configure_finder_runtime_stock(finder, state=state)

    cfg = finder.config.search
    if "retro" in (search_algorithm or "").lower():
        cfg.algorithm = "aizynthfinder.search.retrostar.search_tree.SearchTree"
    else:
        cfg.algorithm = "mcts"
    cfg.max_transforms = int(max_transforms)
    cfg.time_limit = int(time_limit)
    cfg.iteration_limit = int(iteration_limit)
    cfg.return_first = bool(return_first)
    try:
        finder.config.expansion_policy.uspto.cutoff_cumulative = float(cutoff_cumulative)
    except Exception:
        pass

    finder.target_smiles = canon_target
    processed_routes = []
    excl_tokens = [x.strip().lower() for x in (exclude_reaction_names or []) if x.strip()]
    lock_smarts = (lock_substructure_smarts or state.canvas_lock_smarts).strip()
    lock_pat = Chem.MolFromSmarts(lock_smarts) if lock_smarts else None
    ord_prefetch_stats = {"queried_steps": 0, "added_records": 0, "elapsed_sec": 0.0}

    try:
        finder.tree_search()
        finder.build_routes()
        node_templates = {}
        all_fwd_smiles = _harvest_uspto_templates_from_finder(finder, node_templates, state=state)
        raw_routes = list(finder.routes)
        route_steps = [_flatten_reaction_tree(r["reaction_tree"], node_templates) for r in raw_routes]
        all_steps = [step for steps in route_steps for step in steps]

        # Fetch online ORD candidates in parallel, then validate on the notebook thread.
        if state.retro_search_params.get("validate_ord", True):
            ord_prefetch_stats = ord_validator.batch_prefetch_online_ord(
                all_fwd_smiles,
                max_workers=4,
                limit_per_step=20,
                wall_timeout_sec=15.0,
                route_steps=all_steps,
            )

        step_lookup = _classify_route_steps(all_steps, use_gemini, state=state, ord_validator=ord_validator)

        for r, steps in zip(raw_routes, route_steps):
            rxn_tree = r["reaction_tree"]
            state_score = float(r["score"].get("state score", 0.0))
            if not steps:
                continue
            if lock_pat is not None:
                keep = any(
                    Chem.MolFromSmiles(m.smiles) and Chem.MolFromSmiles(m.smiles).HasSubstructMatch(lock_pat)
                    for m in rxn_tree.molecules() if m.smiles != canon_target
                )
                if not keep:
                    continue
            steps_data = []
            banned_hit = False
            for step in steps:
                cls_info = dict(step_lookup[_route_step_key(step)])
                cls_info.pop("gemini_tag", None)

                rxn_label_low = (cls_info.get("reaction_name", "") + " " + cls_info.get("typical_conditions", "")).lower()
                if any(tok in rxn_label_low for tok in excl_tokens):
                    banned_hit = True
                    break
                steps_data.append({**cls_info, **step})
            if banned_hit or not steps_data:
                continue

            ord_route_eval = ord_validator.score_route([s["forward_smiles"] for s in steps_data],
                                                       step_reports=[s["ord_validation"] for s in steps_data])

            precursors = []
            in_stock_cnt = 0
            for leaf in rxn_tree.leafs():
                l_smi = leaf.smiles
                l_mol = Chem.MolFromSmiles(l_smi)
                ikey = Chem.MolToInchiKey(l_mol) if l_mol else ""
                common_reagent = (
                    bool(state.retro_search_params.get("use_common_lab_reagents", True))
                    and is_common_lab_reagent(l_smi)
                )
                if l_smi in state.custom_lab_stock_smiles or (ikey and ikey in state.custom_lab_stock_inchikeys):
                    src, tier = "Lab Stock", "$0 On-Shelf"
                    in_stock_cnt += 1
                elif common_reagent:
                    src, tier = "Common Lab Reagent", "$0 Allowlist"
                    in_stock_cnt += 1
                elif rxn_tree.in_stock(leaf):
                    src, tier = "ZINC Stock", "Commercial"
                    in_stock_cnt += 1
                else:
                    src, tier = "Not in Stock", "Custom Synthesis"
                precursors.append({"smiles": l_smi, "source": src, "cost_tier": tier})
            avg_safe = sum(s["safety_green_score"] for s in steps_data) / max(1, len(steps_data))
            risk_cnt = sum(
                1 for s in steps_data
                if s.get("selectivity_warning", "None") != "None"
                or s.get("stereochemical_control_risk", "None") != "None"
            )
            route_entry = {
                "is_solved": bool(rxn_tree.is_solved) or (in_stock_cnt == len(precursors)),
                "state_score": round(state_score, 4),
                "step_count": len(steps_data),
                "avg_safety_score": round(avg_safe, 2),
                "selectivity_risk_count": risk_cnt,
                "ord_empirical_confidence": ord_route_eval["ord_empirical_confidence"],
                "ord_geometric_confidence": ord_route_eval["ord_geometric_confidence"],
                "ord_confidence_100": ord_route_eval["ord_confidence_100"],
                "ord_bottleneck_step": ord_route_eval["bottleneck_step_index"],
                "ord_mean_yield_pct": ord_route_eval["ord_mean_yield_pct"],
                "ord_max_hte_fail_pct": ord_route_eval["ord_max_hte_fail_pct"],
                "starting_materials": {
                    "total_leaves": len(precursors),
                    "in_stock_count": in_stock_cnt,
                    "overall_cost_tier": "All In-Stock" if in_stock_cnt == len(precursors) else "Partial Stock",
                    "precursors": precursors,
                },
                "steps": steps_data,
            }
            annotate_route_strategy(route_entry)
            route_entry["composite_score"] = compute_composite_route_score(route_entry, ranking_mode)
            route_entry["score"] = route_entry["composite_score"]
            processed_routes.append(route_entry)

        del raw_routes
    finally:
        cleanup_aizynth_memory(finder, trim_ipython_history=True, state=state)

    processed_routes.sort(key=lambda x: (x["is_solved"], x["composite_score"]), reverse=True)
    processed_routes = processed_routes[: int(max_routes)]
    for i, r_i in enumerate(processed_routes):
        dominated = False
        for j, r_j in enumerate(processed_routes):
            if i == j:
                continue
            if (
                r_j["step_count"] <= r_i["step_count"]
                and r_j["avg_safety_score"] >= r_i["avg_safety_score"]
                and r_j["state_score"] >= r_i["state_score"]
                and r_j.get("ord_empirical_confidence", 0.0) >= r_i.get("ord_empirical_confidence", 0.0)
            ) and (
                r_j["step_count"] < r_i["step_count"]
                or r_j["avg_safety_score"] > r_i["avg_safety_score"]
                or r_j["state_score"] > r_i["state_score"]
                or r_j.get("ord_empirical_confidence", 0.0) > r_i.get("ord_empirical_confidence", 0.0)
            ):
                dominated = True
                break
        r_i["is_pareto_optimal"] = not dominated
        r_i["is_pareto"] = not dominated

    state.latest_retro_results = deduplicate_retro_state({
        "target_smiles": canon_target,
        "ranking_mode": ranking_mode,
        "search_algorithm": "retrostar" if "retro" in str(cfg.algorithm).lower() else "mcts",
        "is_solved": any(r["is_solved"] for r in processed_routes),
        "target_scale_grams": float(target_scale_grams),
        "locked_substructure_smarts": lock_smarts,
        "excluded_reactions": excl_tokens,
        "genai_model": ACTIVE_GENAI_SETTINGS.get("model", "gemini-3.8-flash"),
        "genai_thinking": ACTIVE_GENAI_SETTINGS.get("thinking_level", "medium"),
        "ord_index_size": ord_validator.get_index_size(),
        "ord_prefetch_stats": ord_prefetch_stats,
        "routes": processed_routes,
        "ram_mb_after_cleanup": get_process_ram_mb(),
    })
    return state.latest_retro_results
