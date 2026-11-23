"""Colab's interactive retrosynthesis workbench and callback wiring."""

import html
import json
import math
import re
import urllib.parse
from collections import defaultdict
from typing import Optional

import IPython
from google.genai import types
from IPython.display import HTML, clear_output, display
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

from retroai.ai.classifier import (
    _classify_route_steps,
    _merge_step_stoichiometry,
    _route_step_key,
    _step_rdkit_analytics,
    batch_classify_with_gemini,
    rdkit_classify_step,
)
from retroai.ai.gemini_client import call_gemini_with_config, get_gemini_client
from retroai.ai.literature import _clean_doi, _verify_doi_metadata, cbyd_precedent_literature
from retroai.core.chem_utils import (
    _canonical_molecule_smiles,
    _compute_descriptors_and_ro5,
    _extract_primary_smiles,
    mol_or_rxn_to_svg_html,
    rdkit_enumerate_byproducts,
    rdkit_predict_lcms_ions,
    rdkit_predict_workup_properties,
)
from retroai.core.config import (
    ACTIVE_GENAI_SETTINGS,
    GEMINI_MODEL_CATALOG,
    get_process_ram_mb,
    set_active_genai_settings,
)
from retroai.core.resolver import _resolve_chemical_identifier
from retroai.core.schemas import StrategySummary, _clamp_safety_score, _clamp_step_yield
from retroai.core.state import WorkbenchState
from retroai.engine.aizynth import cleanup_aizynth_memory, get_finder, register_custom_lab_stock
from retroai.engine.pipeline import (
    annotate_route_strategy,
    compute_composite_route_score,
    deduplicate_retro_state,
    run_retrosynthesis_pipeline,
)
from retroai.ord.fingerprints import _parse_reaction_mols
from retroai.ord.validator import ORDRetrosynthesisValidator, get_ord_validator

from .ketcher import _ensure_ketcher_bundle


_ELN_COLUMNS = ("Role", "Name / SMILES", "MW (g/mol)", "Eq. / loading", "mmol", "Mass (g)", "Mass (mg)", "Volume (mL)")


def _scale_route_eln(route, target_smiles, scale_g, yield_override=None):
    """Propagate demand by route edge; fall back to SMILES links for older sessions."""
    target = Chem.MolFromSmiles(_extract_primary_smiles(target_smiles))
    if target is None or not target.GetNumAtoms() or not math.isfinite(scale_g) or scale_g <= 0:
        return []
    plans, producers, step_positions = [], defaultdict(list), {}
    override = max(0.05, min(1.0, float(yield_override))) if yield_override is not None else None
    for step in route.get("steps", []):
        smi = step["forward_smiles"]
        _, products = _parse_reaction_mols(smi)
        if not products:
            continue
        product = _canonical_molecule_smiles(products[0])
        step_positions[step["step_index"]] = len(plans)
        producers[product].append(len(plans))
        step_yield = override if override is not None else _clamp_step_yield(step.get("assumed_step_yield"))
        if step_yield is None or step_yield <= 0:
            return []
        primary_provider = next(iter((step.get("ord_validation") or {}).get("top_precedents") or []), {}).get("provider")
        ingredients = _merge_step_stoichiometry(
            smi,
            step.get("reagent_stoichiometry"),
            allow_defaults=primary_provider != "CbyD",
        )
        if primary_provider == "CbyD" and any(
            row["role"] == "Reactant" and row["equivalents"] <= 0 for row in ingredients
        ):
            return []
        plans.append({"step": step, "product": product, "product_mw": Descriptors.MolWt(products[0]),
                      "ingredients": ingredients,
                      "yield": step_yield})
    final_steps = producers.get(_canonical_molecule_smiles(target))
    if not final_steps:
        return []

    # Build a dependency order from the target so convergent branches are scaled independently.
    dependencies, order, visiting = {}, [], set()

    def visit(i):
        if i in dependencies:
            return
        visiting.add(i)
        parents = defaultdict(float)
        step = plans[i]["step"]
        branches = defaultdict(list)
        for step_idx in step.get("precursor_steps", []):
            j = step_positions.get(step_idx)
            if j is not None:
                branches[plans[j]["product"]].append(j)
        for row in plans[i]["ingredients"]:
            if row["role"] != "Reactant":
                continue
            smi = row["smiles_or_formula"]
            if "precursor_steps" in step:
                choices = [branches[smi].pop(0)] if branches[smi] else []
                choices = [j for j in choices if j not in visiting]
            else:
                choices = [j for j in producers[smi] if j not in visiting]
            if choices:
                j = next((j for j in reversed(choices) if j < i), choices[0])
                visit(j)
                parents[j] += row["equivalents"]
        visiting.remove(i)
        dependencies[i] = parents
        order.append(i)

    final_idx = final_steps[-1]
    visit(final_idx)
    demand = defaultdict(float, {final_idx: 1000 * scale_g / Descriptors.MolWt(target)})
    for i in reversed(order):
        plan = plans[i]
        plan["product_mmol"] = demand[i]
        plan["limiting_mmol"] = demand[i] / plan["yield"]
        for j, eq in dependencies[i].items():
            demand[j] += plan["limiting_mmol"] * eq
    return [plans[i] for i in order]


def _eln_step_rows(plan):
    def amounts(mw, mmol):
        mg = (mw * mmol) if (mw and mmol) else None
        return (f"{mw:.2f}" if mw else "—", f"{mmol:.3f}" if mmol else "—",
                f"{mg / 1000:.4f}" if mg else "—", f"{mg:.2f}" if mg else "—")

    rows, n_reactants = [], 0
    for row in plan["ingredients"]:
        role, smi = row["role"], row["smiles_or_formula"]
        name = f"{row['name']} ({smi})" if smi and smi != row["name"] else row["name"]
        eq, molarity = row["equivalents"], row["molarity_m"]
        if role == "Solvent":
            ml = f"{plan['limiting_mmol'] / molarity:.3f}" if molarity else "—"
            name += f" ({molarity:g} M charge basis)" if molarity else " (concentration unreported)"
            rows.append(("Reaction Solvent", name, "—", "—", "—", "—", "—", ml))
            continue
        if role == "Reactant":
            role = "Limiting Reactant" if n_reactants == 0 else "Coupling Partner"
            n_reactants += 1
        load = f"{eq:.2f}" if eq >= 0.01 else (f"{eq:.3g}" if eq else "—")
        if role in ("Catalyst", "Ligand") and eq:
            load += f" ({eq * 100:g} mol%)"
        mw, mmol, g, mg = amounts(row["mw_g_mol"], plan["limiting_mmol"] * eq)
        rows.append((role, name, mw, load, mmol, g, mg, "—"))
    mw, mmol, g, mg = amounts(plan["product_mw"], plan["product_mmol"])
    rows.append(("Expected Product", plan["product"], mw, f"{plan['yield']:.0%} yield", mmol, g, mg, "—"))
    return rows


def _eln_step_text(plan, *, markdown):
    step = plan["step"]

    def text_field(text):
        text = str(text)
        return html.escape(text, quote=False).replace("|", "&#124;") if markdown else text

    title = f"Step {step['step_index']}: {text_field(step.get('reaction_name', 'Reaction step'))}"
    if markdown:
        title = f"**{title}**"
    lines = [f"{title} · {plan['yield']:.0%} assumed yield",
             text_field(step["forward_smiles"]), text_field(step.get("typical_conditions", "")), ""]
    if markdown:
        lines.extend(["| " + " | ".join(_ELN_COLUMNS) + " |", "| " + " | ".join([":---"] * len(_ELN_COLUMNS)) + " |"])
        lines.extend("| " + " | ".join(map(text_field, row)) + " |" for row in _eln_step_rows(plan))
    else:
        lines.append("\t".join(_ELN_COLUMNS))
        lines.extend("\t".join(" ".join(cell.split()) for cell in row) for row in _eln_step_rows(plan))
    if step.get("cbyd_precedents"):
        lines.extend(["", "CbyD does not report stoichiometry/time/workup. Check DOI for stoichiometry/time/workup."])
    else:
        lines.extend(["", "Planning basis: unreported reactant charges default to 1 eq. Verify analog loadings and unknown amounts before use."])
    if step.get("experimental_procedure"):
        label = "**Draft procedure:** " if markdown else "Draft procedure: "
        lines.extend(["", label + text_field(step["experimental_procedure"])])
    if step.get("source_doi_or_patent"):
        lines.append("Procedure source: " + text_field(step["source_doi_or_patent"]))
    return "\n".join(lines)


def _render_eln_step_html(plan, with_heading=True):
    step = plan["step"]
    heading = f"<b>Step {step['step_index']}: {html.escape(step.get('reaction_name', 'Reaction step'))}</b> · " if with_heading else ""
    cols = "".join(f"<th>{col}</th>" for col in _ELN_COLUMNS)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
                   for row in _eln_step_rows(plan))
    return (f"<div>{heading}{plan['yield']:.0%} assumed yield · {plan['limiting_mmol']:.3f} mmol limiting charge</div>"
            f"<div style='overflow-x:auto;'><table class='cf-eln-table'><thead><tr>{cols}</tr></thead><tbody>{rows}</tbody></table></div>"
            "<small>Unreported reactant charges default to 1 eq for planning. Verify analog loadings and unknown amounts before use.</small>")


def _generate_strategy_header(target_smiles: str) -> dict:
    client = get_gemini_client()
    fallback = {
        "iupac_or_label": target_smiles,
        "disconnection_analysis": "Strategic bond disconnections identified via USPTO neural template policy and validated against Open Reaction Database (ORD) precedents.",
        "strategy": "Convergent/linear assembly from commercial ZINC building blocks prioritizing high-confidence ORD transformations, low HTE failure rates, and minimal protecting group steps.",
    }
    if client is None:
        return fallback
    try:
        resp, _, _ = call_gemini_with_config(
            contents=f"Analyze target molecule SMILES: {target_smiles}\nProvide iupac_or_label, 1-sentence disconnection_analysis, and 2-sentence synthetic strategy.",
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=StrategySummary,
        )
        return json.loads(resp.text)
    except Exception:
        return fallback


def _enrich_route_for_chemforge(route: dict, idx: int) -> dict:
    steps = route.get("steps", [])
    n_steps = max(1, len(steps))
    sm = route.get("starting_materials", {})
    total_leaves = max(1, sm.get("total_leaves", 1))
    in_stock = sm.get("in_stock_count", 0)
    is_convergent = total_leaves >= 3 and n_steps >= 2
    lls = int(route.get("lls_length") or max(
        1, n_steps - (1 if is_convergent and n_steps >= 3 else 0)
    ))
    avg_safe = float(route.get("avg_safety_score", 4.0))
    risk_cnt = int(route.get("selectivity_risk_count", 0))
    pg_cnt = sum(1 for s in steps if s.get("protecting_group_note", "None") != "None")
    ord_conf_100 = int(route.get("ord_confidence_100", round(float(route.get("ord_geometric_confidence", 0.65)) * 100)))
    ord_emp = float(route.get("ord_empirical_confidence", 0.50))
    ord_mean_y = route.get("ord_mean_yield_pct")

    metrics = {
        "ORD CONFIDENCE": f"{ord_conf_100}% ({ord_emp:.2f})",
        "ORD MEAN YIELD": f"{ord_mean_y:.0f}%" if ord_mean_y is not None else "N/A",
        "FEASIBLE": int(min(100, max(40, round(float(route.get("state_score", 0.75)) * 100)))),
        "ROBUST": int(max(35, 100 - risk_cnt * 18 - int(float(route.get("ord_max_hte_fail_pct", 0.0)) * 0.3))),
        "SHORT": int(max(30, 100 - (n_steps - 1) * 12)),
        "BUYABLE": int(round((in_stock / total_leaves) * 100)),
        "CHEAP": 100 if sm.get("overall_cost_tier") == "All In-Stock" else 65,
        "CLEAN": int(max(35, 95 - risk_cnt * 20)),
        "MILD CONDITIONS": int(min(100, round(avg_safe * 19))),
        "NO PROTECTING GRP": int(max(30, 100 - pg_cnt * 25)),
        "SAFE REAGENTS": int(min(100, round(avg_safe * 20))),
    }
    failures = []
    for loop in route.get("unproductive_loops", []):
        failures.append(
            f"<b>Unproductive route loop (Steps {', '.join(str(x) for x in loop.get('step_indices', []))}):</b> "
            f"{html.escape(loop.get('type', '').replace('_', ' '))}; apply an {int(abs(0.08) * 100)}% route-score penalty."
        )
    for s in steps:
        st_num = s.get("step_index", 1)
        rxn_name = s.get("reaction_name", "Step")
        ov = s.get("ord_validation") or {}
        if s.get("selectivity_warning", "None") != "None":
            failures.append(f"<b>Step {st_num} ({html.escape(rxn_name)}):</b> {html.escape(s['selectivity_warning'])}")
        stereo_risk = s.get("stereochemical_control_risk", "None")
        if stereo_risk != "None":
            precedent_options = []
            for hit in s.get("stereochemical_precedents", [])[:3]:
                annotation = next(
                    (hit[field] for field in ("dr", "ee", "er") if hit.get(field)),
                    "stereoselective",
                )
                source = hit.get("doi") or hit.get("reaction_id") or "CbyD"
                precedent_options.append(
                    f"{html.escape(str(source))} ({html.escape(str(annotation))}; "
                    f"{html.escape(hit.get('catalysts') or hit.get('ligands') or 'conditions in DOI')})"
                )
            option_text = (
                " Actionable CbyD alternatives: " + "; ".join(precedent_options)
                if precedent_options
                else " Consider an enantiopure chiral-pool source or a documented Evans/Ellman/Myers auxiliary."
            )
            failures.append(
                f"<b>Step {st_num} Stereochemical Control Risk:</b> "
                f"{html.escape(stereo_risk)}{option_text}"
            )
        if ov.get("smarts_matched_count") and ov.get("ord_recorded_byproducts"):
            failures.append(
                f"<b>Step {st_num} ORD Recorded Side-Products (<code>is_desired_product=False</code>):</b> "
                f"{html.escape('; '.join(ov['ord_recorded_byproducts']))}"
            )
        analytics = _step_rdkit_analytics(s.get("forward_smiles", ""))
        byp = analytics["byproducts"]
        competing = [b["origin"] for b in byp if b["origin"] != "target product"]
        if competing:
            failures.append(f"<b>Step {st_num} RDKit Plausible Impurities:</b> {', '.join(sorted(set(competing)))}")
        wk = analytics["workup"]
        if isinstance(wk, dict) and wk.get("advice"):
            failures.append(f"<b>Step {st_num} Workup/Isolation:</b> {html.escape(wk['advice'][0])}")
    if not failures:
        failures.append("No major chemoselectivity, HTE negative-data, or protecting-group conflicts detected.")
    return {
        "route_letter": chr(64 + idx) if idx <= 26 else str(idx),
        "topology": "convergent" if is_convergent else "linear",
        "lls": lls,
        "score_100": int(round(float(route.get("composite_score", 0.0)) * 100)),
        "ord_emp_score": ord_emp,
        "ord_conf_100": ord_conf_100,
        "ord_bottleneck": route.get("ord_bottleneck_step", 1),
        "precedent_list": ", ".join(s.get("reaction_name", "Reaction") for s in steps),
        "metrics": metrics,
        "failures": failures,
    }


def _render_complexity_sparkline_html(route: dict) -> str:
    profile = route.get("lls_complexity_profile") or []
    sa_rows = [row for row in profile if row.get("product_sascore") is not None]
    sps_rows = [row for row in profile if row.get("product_sps") is not None]
    if not sa_rows or not sps_rows:
        return "<div style='font-size:10px;color:#64748b;'>Complexity profile unavailable in this RDKit build.</div>"
    sascores = [row["product_sascore"] for row in sa_rows]
    sps_values = [row["product_sps"] for row in sps_rows]
    lo_sa, hi_sa = min(sascores), max(sascores)
    lo_sps, hi_sps = min(sps_values), max(sps_values)
    width, height, padding = 320, 58, 6

    def points(rows, metric, field, low, high):
        values = [row[field] for row in rows]
        if len(values) == 1:
            xs = [width / 2]
        else:
            xs = [padding + i * (width - 2 * padding) / (len(values) - 1) for i in range(len(values))]
        ys = [
            height - padding - ((value - low) / max(high - low, 1e-9)) * (height - 2 * padding)
            for value in values
        ]
        markers = "".join(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.2' fill="
            f"'{ '#f59e0b' if rows[index].get('is_strategic_key_step') else ('#0284c7' if metric == 'SAScore' else '#0d9488') }'>"
            f"<title>Step {rows[index].get('step_index')}: {metric} {values[index]:.2f}</title></circle>"
            for index, (x, y) in enumerate(zip(xs, ys))
        )
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        return polyline, markers

    sa_points, sa_markers = points(sa_rows, "SAScore", "product_sascore", lo_sa, hi_sa)
    sps_points, sps_markers = points(sps_rows, "SPS", "product_sps", lo_sps, hi_sps)
    key_count = route.get("strategic_key_step_count", 0)
    return (
        "<div style='font-size:10px;color:#334155;margin:6px 0 3px;'><b>LLS COMPLEXITY PROFILE</b> · "
        f"<span style='color:#0284c7;'>SAScore</span> · <span style='color:#0d9488;'>SPS</span> · "
        f"{key_count} strategic key step(s)</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Route complexity profile' "
        "style='width:100%;max-width:420px;height:58px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;'>"
        f"<polyline points='{sa_points}' fill='none' stroke='#0284c7' stroke-width='2'/>{sa_markers}"
        f"<polyline points='{sps_points}' fill='none' stroke='#0d9488' stroke-width='2'/>{sps_markers}"
        "</svg>"
    )


def _render_step_source(source, ord_ids, verified_literature):
    if not source:
        return ""
    url = ""
    if source in ord_ids:
        url = "https://open-reaction-database.org/id/" + urllib.parse.quote(source, safe="")
    elif _clean_doi(source) and any(source.lower() == work["doi"].lower() for work in verified_literature):
        url = "https://doi.org/" + urllib.parse.quote(source, safe="/")
    elif re.fullmatch(r"[A-Z]{2}[\d /-]+[A-Z]?\d?", source, re.I):
        url = "https://patents.google.com/patent/" + urllib.parse.quote(re.sub(r"[ /-]", "", source), safe="")
    label = html.escape(source)
    return f"<a href='{html.escape(url)}' target='_blank' rel='noopener noreferrer'>{label}</a>" if url else label


def _render_ord_step_panel_html(ov: dict, verified_literature=()) -> str:
    """Render evidence only for records admitted by the reaction gate."""
    ov = ov or {}
    count = ov.get("smarts_matched_count")
    tier = ov.get("matched_smarts_label") or "Not verified"
    delta = ov.get("rdkit_delta_fg_signature") or {}
    lost, gained = delta.get("consumed_fgs", []), delta.get("formed_fgs", [])
    full_delta = f"Consumed: {', '.join(lost) or 'none'}; Formed: {', '.join(gained) or 'none'}"
    short_lost = ', '.join(lost[:2]) + ('…' if len(lost) > 2 else '') if lost else 'none'
    short_gained = ', '.join(gained[:2]) + ('…' if len(gained) > 2 else '') if gained else 'none'
    badges = (
        f"<span class='cf-metric-pill'>Matched SMARTS Tier: {html.escape(tier)}</span>"
        f"<span class='cf-metric-pill' title='{html.escape(full_delta)}'>RDKit ΔFG Signature: {html.escape(short_lost)} → {html.escape(short_gained)}</span>"
        f"<span class='cf-metric-pill'>SMARTS-Verified Precedent Hits: {int(count or 0)}</span>"
    )
    if not count:
        message = ("0 ORD/CbyD reactions in the local indexes matched this Reaction SMARTS / ΔFG gate — using USPTO Template + RDKit ΔFG conditions"
                   if count == 0 else "This route has no gated ORD report. Reclassify it to verify precedents.")
        return f"<div class='cf-step-section'>{badges}<div class='cf-metric-pill'>{message}</div></div>"

    step_conf = ov.get("step_ord_confidence", 0.5)
    max_sim = ov.get("max_rxn_similarity", 0.0)
    mean_y = ov.get("mean_analog_yield_pct")
    med_y = ov.get("median_analog_yield_pct")
    fail_r = ov.get("hte_failure_rate_pct", 0.0)
    exact_hits = ov.get("exact_product_hits", 0)
    grounded_cond = ov.get("grounded_conditions_consensus") or "Conditions not reported by the top analog"

    conf_color = "#0f766e" if step_conf >= 0.68 else ("#b45309" if step_conf >= 0.45 else "#be123c")
    fail_badge = (
        f"<span style='background:#ffe4e6;color:#be123c;padding:1px 6px;border-radius:4px;font-weight:700;'>"
        f"⚠️ HTE Fail Rate (&lt;15%): {fail_r:.0f}%</span>"
        if fail_r >= 25.0
        else f"<span style='background:#ccfbf1;color:#0f766e;padding:1px 6px;border-radius:4px;font-weight:600;'>"
             f"HTE Fail Rate (&lt;15%): {fail_r:.0f}%</span>"
    )

    prec_cards = []
    for p in ov.get("top_precedents", []):
        oid = p.get("ord_id", "ord")
        provenance = p.get("provenance") or p.get("provider") or "ORD"
        oid_html = _render_step_source(oid, {oid} if p.get("provider") != "CbyD" else set(), ())
        sim = p.get("similarity", 0.0)
        y_str = f"{p['yield_pct']:.0f}%" if p.get("yield_pct") is not None else "N/A"
        doi = _clean_doi(p.get("doi") or "")
        reference = doi if (p.get("provider") == "CbyD" or any(doi.lower() == work["doi"].lower() for work in verified_literature)) else p.get("patent_id")
        ref_html = _render_step_source(reference, (), verified_literature)

        cond_bits = []
        if p.get("catalysts") and p["catalysts"] != "None":
            cond_bits.append(f"Cat: {html.escape(p['catalysts'])}")
        if p.get("reagents") and p["reagents"] != "None":
            cond_bits.append(f"Reag: {html.escape(p['reagents'])}")
        if p.get("solvents") and p["solvents"] != "Unspecified":
            cond_bits.append(f"Solv: {html.escape(p['solvents'])}")
        if p.get("step_count", 1) > 1:
            cond_bits.append(f"⚠ telescoped {int(p['step_count'])}-step yield")
        if p.get("is_borsm"):
            cond_bits.append("⚠ BORSM")
        if p.get("temp_c") is not None:
            cond_bits.append(f"{p['temp_c']:.0f} °C")
        elif p.get("temperature"):
            cond_bits.append(html.escape(str(p["temperature"])))
        if p.get("hazard_flags"):
            cond_bits.append("⚠ " + html.escape("; ".join(p["hazard_flags"])))
        cond_summary = " · ".join(cond_bits) if cond_bits else "Conditions unspecified"

        rxn_smi = (p.get("rxn_smiles") or "").strip()
        rxn_svg_html = (
            f"<div style='margin-top:4px;'>{mol_or_rxn_to_svg_html(rxn_smi, width=440, height=105)}</div>"
            if rxn_smi and ">>" in rxn_smi
            else ""
        )
        procedure = p.get("procedure_details") or ""
        procedure_html = (f"<details class='cf-step-section'><summary>Verbatim ORD analog procedure</summary>"
                          f"<div class='cf-verbatim'>{html.escape(procedure)}</div></details>" if procedure else "")

        prec_cards.append(
            f"<div style='background:#ffffff;border:1px solid #bae6fd;border-radius:6px;padding:6px 8px;margin-top:5px;'>"
            f"<div style='font-size:10.5px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px;'>"
            f"<span><b>{html.escape(provenance)}</b> · {oid_html} · Sim <b>{sim:.2f}</b> · Yield <b>{y_str}</b> · {cond_summary}</span>"
            f"<span>{ref_html}</span>"
            f"</div>"
            f"{rxn_svg_html}"
            f"{procedure_html}"
            f"</div>"
        )

    yield_str = f"{mean_y:.1f}% (median {med_y:.1f}%)" if (mean_y is not None and med_y is not None) else "N/A"
    return (
        f"<div style='margin-top:6px;padding:7px 9px;background:#f0fdfa;border-left:3px solid #0d9488;border-radius:4px;font-size:11px;color:#1e293b;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;'>"
        f"<span style='font-weight:700;color:#0f766e;'>🔬 ORD + CbyD Step Validation (Verified Exact Product Hits: {exact_hits} · Gated similarity: {max_sim:.2f})</span>"
        f"<span style='display:flex;gap:6px;align-items:center;'>"
        f"{fail_badge}"
        f"<span style='background:#e0f2fe;color:{conf_color};padding:1px 7px;border-radius:4px;font-weight:700;'>Step S_ORD: {step_conf:.2f}</span>"
        f"</span></div>"
        f"<div class='cf-pill-grid'>{badges}</div>"
        f"<div style='margin-top:4px;'><b>Evidence-Grounded Forward Conditions:</b> {html.escape(grounded_cond)} · <b>Analog Mean Yield:</b> {yield_str}</div>"
        f"<div style='margin-top:4px;'>{''.join(prec_cards)}</div>"
        f"</div>"
    )


def _render_ranked_conditions_html(step):
    cards = []
    verified = step.get("verified_literature") or []
    report = step.get("ord_validation") or {}
    ord_ids = {row["ord_id"] for row in report.get("top_precedents", [])} if report.get("smarts_matched_count") else set()
    for cond in step.get("ranked_conditions") or []:
        values = [("Catalyst / ligand", "catalyst_and_ligand"), ("Base / reagent", "base_or_reagent"),
                  ("Solvent / concentration", "solvent_and_molarity"), ("Temperature / time", "temp_and_time"),
                  ("Yield", "expected_yield_range"), ("Tradeoff", "tradeoff_note")]
        parts = "".join(f"<div><b>{label}:</b> {html.escape(cond.get(key) or 'Not reported')}</div>" for label, key in values)
        source = _render_step_source(cond.get("citation_or_ord_id"), ord_ids, verified)
        cards.append(f"<div class='cf-condition-card'><b>Rank #{int(cond['rank'])}: {html.escape(cond['condition_label'])}</b>"
                     f"{parts}<div><b>Source:</b> {source or 'Not established'}</div></div>")
    hazard_flags = list(step.get("hazard_flags") or [])
    data_gaps = list(step.get("data_gap_warnings") or [])
    warning = ""
    if hazard_flags or data_gaps:
        warnings = hazard_flags + data_gaps
        warning = (
            "<div style='margin-bottom:5px;padding:5px 7px;background:#fff7ed;color:#9a3412;border-radius:4px;'>"
            f"<b>Human review:</b> {' · '.join(html.escape(item) for item in warnings)}</div>"
        )
    return (f"<details class='cf-step-section'><summary>Ranked reaction conditions ({len(cards)})</summary>"
            f"{warning}{''.join(cards)}</details>") if cards else warning


def _render_step_analytics_html(forward_smiles):
    analytics = _step_rdkit_analytics(forward_smiles)
    byproducts = [b for b in analytics["byproducts"] if b["origin"] != "target product"]
    impurity_lines = "".join(
        f"<li>{html.escape(b['origin'])}: <code>{html.escape(b['smiles'])}</code> · {html.escape(b['formula'])} · {b['exact_mw']:.4f} Da</li>"
        for b in byproducts
    )
    ion_labels = ("[M+H]+", "[M+Na]+", "[M+NH4]+", "[M-H]-")
    ion_rows = []
    for item in analytics["lcms"]:
        masses = "".join(f"<td>{item[label]:.4f}</td>" for label in ion_labels)
        ion_rows.append(f"<tr><td>{html.escape(item['role'])}</td><td><code>{html.escape(item['smiles'])}</code>"
                        f"</td>{masses}<td>{html.escape(item['isotope_note'])}</td></tr>")
    mass_headers = "".join(f"<th>{label}</th>" for label in ion_labels)
    ions = (f"<table class='cf-eln-table'><thead><tr><th>Species</th><th>SMILES</th>{mass_headers}<th>Isotope note</th></tr></thead>"
            f"<tbody>{''.join(ion_rows)}</tbody></table>") if ion_rows else "No ions predicted."
    workup = analytics["workup"]
    if product := workup.get("product"):
        properties = (f"Product cLogP {product['clogp']:.2f} · TPSA {product['tpsa']:.1f} Å² · "
                      f"ΔcLogP {workup['delta_clogp']:+.2f} · ΔTPSA {workup['delta_tpsa']:+.1f} Å²")
    else:
        properties = "Workup prediction unavailable."
    advice = "".join(f"<li>{html.escape(note)}</li>" for note in workup.get("advice", []))
    impurity_list = f"<ul>{impurity_lines}</ul>" if impurity_lines else "<div>No byproducts identified by the RDKit heuristics.</div>"
    return ("<details class='cf-step-section'><summary>RDKit analytics · Byproducts, LC-MS &amp; Workup</summary>"
            f"<b>Plausible byproducts (not observed)</b>{impurity_list}"
            f"<b>Predicted monoisotopic LC-MS ions (m/z)</b><div style='overflow-x:auto'>{ions}</div>"
            f"<b>Polarity and workup prediction</b><div>{properties}</div><ul>{advice}</ul></details>")


def _render_step_procedure_html(step):
    text = (step.get("experimental_procedure") or "").strip()
    if not text:
        return ""
    report = step.get("ord_validation") or {}
    ids = {p["ord_id"] for p in report.get("top_precedents", [])} if report.get("smarts_matched_count") else set()
    source = _render_step_source(step.get("source_doi_or_patent"), ids, step.get("verified_literature") or [])
    return (f"<details class='cf-step-section'><summary>Draft experimental procedure</summary>"
            f"<div class='cf-verbatim'>{html.escape(text)}</div><b>Source:</b> {source or 'No verified citation'}"
            "</details>")


def _render_routes_html_from_results(res: dict, ord_validator: ORDRetrosynthesisValidator) -> str:
    """Render route cards on demand without storing SVGs in the session state."""
    routes = res.get("routes", [])
    if not routes:
        return "<div style='color:#dc2626;font-weight:600;'>⚠️ No routes satisfied the current constraints.</div>"
    target = res.get("target_smiles", "")
    rk_mode = res.get("ranking_mode", "balanced")
    scale_g = float(res.get("target_scale_grams", 1.0))
    used_model = res.get("genai_model") or ACTIVE_GENAI_SETTINGS.get("model", "gemini-3.8-flash")
    used_think = res.get("genai_thinking") or ACTIVE_GENAI_SETTINGS.get("thinking_level", "medium")
    strat = _generate_strategy_header(target)
    cards_html = []
    for i, r in enumerate(routes, start=1):
        cf = _enrich_route_for_chemforge(r, i)
        eln_plans = {plan["step"]["step_index"]: plan for plan in _scale_route_eln(r, target, scale_g)}
        pills_html = "".join(f"<span class='cf-metric-pill'>{k} <b>{v}</b></span>" for k, v in cf["metrics"].items())
        fail_items = "".join(f"<li style='margin-bottom:4px;'>{f}</li>" for f in cf["failures"])
        steps_html = []
        for st in r["steps"]:
            svg = mol_or_rxn_to_svg_html(st["forward_smiles"], width=480, height=120, label=f"Step {st['step_index']}: {st['reaction_name']}")
            key_step = (
                "<span style='display:inline-block;background:#fef3c7;color:#92400e;border:1px solid #f59e0b;"
                "padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;'>⭐ STRATEGIC KEY STEP</span>"
                if st.get("is_strategic_key_step") else ""
            )
            swap = f" · <span style='color:#0d9488;font-weight:600;'>Green Swap: {html.escape(st['greener_solvent_swap'])}</span>" if st.get("greener_solvent_swap", "None") != "None" else ""
            ord_panel_html = _render_ord_step_panel_html(st.get("ord_validation"), st.get("verified_literature") or [])
            plan = eln_plans.get(st["step_index"])
            eln = _render_eln_step_html(plan, with_heading=False) if plan else "No scalable charge for this step."
            copy_attrs = f"onclick='cfCopyStepELN(this)' data-eln='{html.escape(_eln_step_text(plan, markdown=False))}'" if plan else "disabled"
            copy_button = f"<button type='button' class='cf-btn cf-copy-eln' {copy_attrs}>📋 Copy Step ELN to Clipboard</button>"
            steps_html.append(
                f"<div class='cf-step-box'>{svg}"
                f"<div style='margin:4px 0;'>{key_step}</div>"
                f"<div style='font-size:11px;color:#334155;margin-top:4px;'>"
                f"<b>Primary conditions:</b> {html.escape(st['typical_conditions'])} · <b>Safety:</b> {_clamp_safety_score(st.get('safety_green_score'))}/5{swap}</div>"
                f"{ord_panel_html}"
                f"{_render_ranked_conditions_html(st)}{_render_step_procedure_html(st)}"
                f"{_render_step_analytics_html(st['forward_smiles'])}"
                f"<details class='cf-step-section'><summary>Scaled step ELN</summary>{eln}</details>{copy_button}</div>"
            )
        sm_badges = "".join(
            f"<span class='cf-sm-badge {'cf-sm-buy' if p['source'] != 'Not in Stock' else 'cf-sm-custom'}'>"
            f"<code>{html.escape(p['smiles'])}</code> ({p['source']})</span>"
            for p in r["starting_materials"]["precursors"]
        )
        pareto_badge = "<span class='cf-badge-green' style='background:#ccfbf1;color:#0f766e;border:1px solid #5eead4;margin-left:4px;'>⭐ PARETO</span>" if r.get("is_pareto_optimal") else ""
        solved_badge = "<span class='cf-badge-green'>🟢 SOLVED</span>" if r.get("is_solved") else "<span class='cf-badge-green' style='background:#fef2f2;color:#dc2626;'>🟡 PARTIAL</span>"
        key_badge = (
            f"<span style='background:#fff7ed;color:#c2410c;border:1px solid #fdba74;padding:2px 6px;"
            f"border-radius:4px;font-size:10px;font-weight:700;margin-left:4px;'>"
            f"⭐ {r.get('strategic_key_step_count', 0)} KEY</span>"
        )
        cards_html.append(f"""
        <div class="cf-route-card">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
            <div>
              {solved_badge}{pareto_badge}{key_badge}
              <strong style="font-size:14px;margin-left:6px;color:#0f172a;">Route {cf['route_letter']} — {r['step_count']} steps, {cf['topology']}</strong>
              <span style="font-size:11px;color:#475569;margin-left:4px;">(LLS {cf['lls']} · Stock {r['starting_materials']['in_stock_count']}/{r['starting_materials']['total_leaves']} · Bottleneck: Step {cf['ord_bottleneck']})</span>
            </div>
            <div style="display:flex;gap:6px;align-items:center;">
              <div class="cf-score-pill" style="background:linear-gradient(135deg,#0d9488 0%,#0f766e 100%);" title="Multiplicative Route ORD Empirical Confidence Score (S_ORD) &amp; Geometric Step Mean">🔬 ORD S_route: {cf['ord_emp_score']:.3f} ({cf['ord_conf_100']}%)</div>
              <div class="cf-score-pill">score {cf['score_100']}</div>
            </div>
          </div>
          <div style="font-size:11px;color:#475569;margin:6px 0;"><b>Precedent:</b> {html.escape(cf['precedent_list'])}</div>
          {_render_complexity_sparkline_html(r)}
          <div class="cf-pill-grid">{pills_html}</div>
          <details class="cf-fail-details">
            <summary>⚠ WHY MIGHT THIS FAIL? ({len(cf['failures'])})</summary>
            <ul style="margin:6px 0 0 18px;padding:0;font-size:11px;color:#334155;">{fail_items}</ul>
          </details>
          <div style="margin-top:8px;font-size:11px;font-weight:700;color:#0369a1;">RETROSYNTHETIC STEPS, ORD EMPIRICAL PRECEDENTS, EXPERIMENTAL PROCEDURES &amp; STARTING MATERIALS</div>
          {''.join(steps_html)}
          <div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:6px;">{sm_badges}</div>
        </div>
        """)

    ram_now = get_process_ram_mb()
    ord_cnt = ord_validator.get_index_size()
    pf = res.get("ord_prefetch_stats") or {}
    pf_added = int(pf.get("added_records", 0))
    pf_steps = int(pf.get("queried_steps", 0))
    pf_sec = float(pf.get("elapsed_sec", 0.0))

    if pf_steps > 0:
        pf_pill_html = (
            f" · <span style='background:#ccfbf1;color:#0f766e;padding:1px 6px;border-radius:4px;border:1px solid #5eead4;'>"
            f"🌐 +{pf_added} ONLINE ORD HITS ({pf_steps} STEPS IN {pf_sec:.2f}s)</span>"
        )
    else:
        pf_pill_html = (
            " · <span style='background:#e0f2fe;color:#0369a1;padding:1px 6px;border-radius:4px;'>"
            "⚡ LOCAL ORD CACHE HIT (0.00s)</span>"
        )

    header_html = f"""
    <div class="cf-strat-box">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
        <div style="font-size:10px;font-weight:700;color:#0284c7;">TARGET · {html.escape(str(res.get('search_algorithm', 'RETROSTAR')).upper())} · MODE: {html.escape(rk_mode.upper())} ({len(routes)} ROUTES) · ORD INDEX: {ord_cnt:,} RXNS{pf_pill_html} · AI: {html.escape(used_model.upper())} ({html.escape(used_think.upper())}) · RAM: {ram_now} MB</div>
        <button class="cf-btn" style="padding:3px 9px;font-size:11px;" onclick="cfOpenAgentWithRoutes()">🤖 Discuss Routes in Chemist AI Agent</button>
      </div>
      <div style="font-size:13px;font-weight:600;margin:4px 0 6px 0;word-break:break-all;color:#0f172a;">{html.escape(strat.get('iupac_or_label', target))}</div>
      <div style="font-size:10px;font-weight:700;color:#0d9488;">DISCONNECTION ANALYSIS</div>
      <div style="font-size:12px;margin-bottom:6px;color:#1e293b;">{html.escape(strat.get('disconnection_analysis', ''))}</div>
      <div style="font-size:10px;font-weight:700;color:#0d9488;">STRATEGY</div>
      <div style="font-size:12px;color:#1e293b;">{html.escape(strat.get('strategy', ''))}</div>
    </div>
    {''.join(cards_html)}
    """
    return header_html


def _render_workbench(ord_validator: ORDRetrosynthesisValidator):
    """Renders the Unified ChemForge Workbench and frees temporary build buffers."""
    ketcher_b64_html, _ = _ensure_ketcher_bundle()
    default_smi = "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1"
    init_ram_mb = get_process_ram_mb()
    init_ord_cnt = ord_validator.get_index_size()
    catalog_json = json.dumps(GEMINI_MODEL_CATALOG)
    model_options_html = "".join(
        f'<option value="{m_id}" {"selected" if m_id == "gemini-3.8-flash" else ""}>{html.escape(spec["label"])}</option>'
        for m_id, spec in GEMINI_MODEL_CATALOG.items()
    )
    default_summary = html.escape(GEMINI_MODEL_CATALOG["gemini-3.8-flash"]["summary"])

    unified_workbench_html = f"""
<style>
  :root {{
    --cf-blue: #0284c7;
    --cf-blue-dark: #0369a1;
    --cf-teal: #0d9488;
    --cf-teal-bright: #14b8a6;
    --cf-cyan-glow: rgba(6, 182, 212, 0.22);
    --cf-grad-primary: linear-gradient(135deg, #0284c7 0%, #0d9488 100%);
    --cf-grad-action: linear-gradient(135deg, #0ea5e9 0%, #14b8a6 100%);
    --cf-grad-lock: linear-gradient(135deg, #0369a1 0%, #1e40af 100%);
    --cf-bg-tint: #f0f9ff;
    --cf-border: #bae6fd;
  }}
  .cf-workbench {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, sans-serif;
    display: grid;
    grid-template-columns: 540px 1fr;
    gap: 14px;
    max-width: 1360px;
    margin-top: 6px;
    color: #0f172a;
  }}
  .cf-pane {{
    border: 1px solid var(--cf-border);
    border-radius: 12px;
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    padding: 13px;
    box-sizing: border-box;
    box-shadow: 0 4px 16px rgba(2, 132, 199, 0.06);
  }}
  .cf-sec-hdr {{
    font-size: 11px;
    font-weight: 800;
    color: var(--cf-blue-dark);
    letter-spacing: 0.7px;
    margin-bottom: 9px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #e0f2fe;
    padding-bottom: 6px;
  }}
  .cf-model-bar {{
    background: linear-gradient(135deg, #f0f9ff 0%, #f0fdfa 100%);
    border: 1px solid #7dd3fc;
    border-radius: 9px;
    padding: 9px 11px;
    margin-bottom: 10px;
  }}
  .cf-model-row {{
    display: grid;
    grid-template-columns: 1fr 195px;
    gap: 8px;
    align-items: center;
  }}
  .cf-model-select {{
    width: 100%;
    padding: 5px 8px;
    font-size: 11.5px;
    font-weight: 600;
    color: #0f172a;
    border: 1px solid #38bdf8;
    border-radius: 6px;
    background: #fff;
  }}
  .cf-model-summary {{
    font-size: 11px;
    color: #334155;
    margin-top: 6px;
    line-height: 1.35;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }}
  .cf-input-row {{ display: flex; gap: 6px; margin-bottom: 6px; align-items: center; }}
  .cf-input {{
    flex: 1;
    padding: 6px 9px;
    font-size: 12px;
    border: 1px solid #93c5fd;
    border-radius: 6px;
    background: #fff;
    color: #0f172a;
    transition: all 0.18s ease;
  }}
  .cf-input:focus {{
    outline: none;
    border-color: var(--cf-teal-bright);
    box-shadow: 0 0 0 3px var(--cf-cyan-glow);
  }}
  .cf-btn {{
    background: var(--cf-grad-primary);
    color: #fff;
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 6px;
    cursor: pointer;
    white-space: nowrap;
    box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25);
    transition: transform 0.12s ease, box-shadow 0.15s ease, filter 0.15s ease;
  }}
  .cf-btn:hover {{
    filter: brightness(1.08);
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.35);
    transform: translateY(-1px);
  }}
  .cf-btn:active {{ transform: translateY(0); }}
  .cf-btn-green {{ background: var(--cf-grad-action); box-shadow: 0 2px 8px rgba(20, 184, 166, 0.32); }}
  .cf-btn-amber {{ background: var(--cf-grad-lock); box-shadow: 0 2px 8px rgba(3, 105, 161, 0.28); }}
  .cf-chip {{
    font-size: 11px;
    padding: 3px 9px;
    border-radius: 12px;
    background: linear-gradient(135deg, #e0f2fe 0%, #ccfbf1 100%);
    color: #0369a1;
    border: 1px solid #7dd3fc;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .cf-chip:hover {{ background: var(--cf-grad-primary); color: #fff; border-color: transparent; }}
  .cf-desc-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    background: linear-gradient(135deg, #f0f9ff 0%, #f0fdfa 100%);
    border: 1px solid #bae6fd;
    padding: 8px;
    border-radius: 8px;
    margin-top: 8px;
    font-size: 11px;
  }}
  .cf-desc-item b {{ display: block; color: #0369a1; font-size: 9px; letter-spacing: 0.4px; }}
  .cf-tabs {{ display: flex; gap: 6px; border-bottom: 1px solid #bae6fd; padding-bottom: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
  .cf-tab {{
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #bae6fd;
    border-radius: 7px;
    background: #f0f9ff;
    color: #0369a1;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .cf-tab:hover {{ background: #e0f2fe; }}
  .cf-tab.active {{
    background: var(--cf-grad-primary);
    color: #fff;
    border-color: transparent;
    box-shadow: 0 2px 8px rgba(2, 132, 199, 0.28);
  }}
  .cf-mode-pill {{
    padding: 4px 11px;
    font-size: 11px;
    border-radius: 14px;
    border: 1px solid #7dd3fc;
    background: #f0f9ff;
    color: #0369a1;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .cf-mode-pill:hover {{ background: #e0f2fe; }}
  .cf-mode-pill.active {{
    background: var(--cf-grad-action);
    color: #fff;
    border-color: transparent;
    font-weight: 700;
    box-shadow: 0 2px 6px rgba(13, 148, 136, 0.28);
  }}
  .cf-cfg-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    background: #f0f9ff;
    padding: 10px;
    border: 1px solid #bae6fd;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 11px;
  }}
  .cf-cfg-item label {{ display: block; font-weight: 600; color: #0369a1; margin-bottom: 3px; }}
  .cf-cfg-item select, .cf-cfg-item input {{
    width: 100%;
    padding: 4px 6px;
    font-size: 11px;
    border: 1px solid #7dd3fc;
    border-radius: 5px;
    box-sizing: border-box;
    background: #fff;
  }}
  .cf-strat-box {{
    background: linear-gradient(135deg, #f0f9ff 0%, #f0fdfa 100%);
    border-left: 4px solid var(--cf-teal);
    border-top: 1px solid #bae6fd;
    border-right: 1px solid #bae6fd;
    border-bottom: 1px solid #bae6fd;
    padding: 10px 12px;
    border-radius: 8px;
    margin-bottom: 12px;
  }}
  .cf-route-card {{
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
    background: #fff;
    box-shadow: 0 2px 8px rgba(2, 132, 199, 0.05);
  }}
  .cf-badge-green {{ background: #ccfbf1; color: #0f766e; font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 5px; }}
  .cf-score-pill {{ background: var(--cf-grad-primary); color: #fff; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 12px; }}
  .cf-pill-grid {{ display: flex; flex-wrap: wrap; gap: 5px; margin: 8px 0; }}
  .cf-metric-pill {{ background: #f0f9ff; color: #0369a1; border: 1px solid #bae6fd; font-size: 10px; padding: 3px 7px; border-radius: 5px; }}
  .cf-fail-details {{ background: #f0fdfa; border: 1px solid #99f6e4; border-radius: 6px; padding: 6px 10px; margin: 8px 0; font-size: 12px; }}
  .cf-fail-details summary {{ cursor: pointer; font-weight: 700; color: #0f766e; }}
  .cf-step-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; margin-top: 6px; overflow-x: auto; }}
  .cf-step-section {{ margin-top: 6px; padding: 6px 8px; border: 1px solid #e2e8f0; border-radius: 6px; background: #fff; font-size: 11px; }}
  .cf-step-section summary {{ cursor: pointer; color: #0369a1; font-weight: 700; }}
  .cf-step-section[open] > summary {{ margin-bottom: 5px; }}
  .cf-condition-card {{ padding: 6px 8px; margin-top: 5px; border-left: 3px solid #bae6fd; background: #f8fafc; }}
  .cf-verbatim {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
  .cf-copy-eln {{ margin-top: 6px; }}
  .cf-sm-badge {{ font-size: 11px; padding: 2px 7px; border-radius: 4px; }}
  .cf-sm-buy {{ background: #ccfbf1; color: #0f766e; border: 1px solid #5eead4; }}
  .cf-sm-custom {{ background: #ffe4e6; color: #be123c; border: 1px solid #fda4af; }}
  .cf-eln-table {{ width: 100%; border-collapse: collapse; font-size: 11px; text-align: left; background: #ffffff; border: 1px solid #bae6fd; border-radius: 6px; overflow: hidden; }}
  .cf-eln-table th {{ background: #e0f2fe; color: #0369a1; font-weight: 700; padding: 6px 8px; border-bottom: 1px solid #bae6fd; }}
  .cf-eln-table td {{ padding: 5px 8px; border-bottom: 1px solid #e2e8f0; color: #1e293b; }}
  .cf-progress-card {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; border: 1px solid #0ea5e9; border-radius: 10px; padding: 14px 16px; margin: 8px 0 12px 0; }}
  .cf-progress-track {{ width: 100%; height: 12px; background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 999px; overflow: hidden; margin: 9px 0; }}
  .cf-progress-fill {{ height: 100%; width: 2%; background: linear-gradient(90deg, #0284c7 0%, #06b6d4 50%, #14b8a6 100%); background-size: 200% 100%; animation: cfShimmer 2s linear infinite; border-radius: 999px; transition: width 0.4s ease; }}
  @keyframes cfShimmer {{ 0% {{ background-position: 100% 0; }} 100% {{ background-position: -100% 0; }} }}
  .cf-agent-log {{ height: 420px; overflow-y: auto; border: 1px solid #bae6fd; border-radius: 8px; padding: 10px; background: #f8fafc; margin-bottom: 8px; display: flex; flex-direction: column; gap: 8px; }}
  .cf-msg-user {{ align-self: flex-end; max-width: 82%; background: var(--cf-grad-primary); color: #fff; padding: 8px 12px; border-radius: 12px 12px 2px 12px; font-size: 12px; }}
  .cf-msg-bot {{ align-self: flex-start; max-width: 92%; background: #ffffff; color: #0f172a; border: 1px solid #bae6fd; padding: 9px 12px; border-radius: 12px 12px 12px 2px; font-size: 12px; box-shadow: 0 2px 6px rgba(2, 132, 199, 0.05); }}
</style>

<div class="cf-workbench">
  <!-- LEFT PANE: 01 KETCHER STRUCTURE & CONSTRAINTS -->
  <div class="cf-pane">
    <div class="cf-sec-hdr">
      <span>01 · STRUCTURE / REACTION (EPAM KETCHER)</span>
      <span id="cf-status" style="color:#0d9488;font-weight:700;">⏳ Loading Ketcher...</span>
    </div>
    <div class="cf-input-row">
      <input id="cf-name-a" class="cf-input" placeholder="Search Drug Name, CAS, CID, ChEMBL, or SMILES (e.g. Febuxostat, DB04868)..." />
      <button class="cf-btn" onclick="cfLoadSingle()">Load</button>
    </div>
    <div class="cf-input-row">
      <input id="cf-react-a" class="cf-input" placeholder="Reactant A: name / SMILES" />
      <input id="cf-react-b" class="cf-input" placeholder="Reactant B: name / SMILES" />
      <button class="cf-btn" onclick="cfLoadPair()">Load A + B</button>
    </div>
    <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:6px;font-size:11px;color:#0369a1;align-items:center;">
      <b>Examples:</b>
      <button class="cf-chip" onclick="cfQuick('Nilotinib')">Nilotinib</button>
      <button class="cf-chip" onclick="cfQuick('Febuxostat')">Febuxostat</button>
      <button class="cf-chip" onclick="cfQuick('Celecoxib')">Celecoxib</button>
      <button class="cf-chip" onclick="cfQuick('Imatinib')">Imatinib</button>
    </div>
    <iframe id="cf-ketcher-frame" style="width:100%;height:430px;border:1px solid #7dd3fc;border-radius:8px;background:#fff;"></iframe>
    <div class="cf-input-row" style="margin-top:6px;">
      <input id="cf-smiles-box" class="cf-input" style="font-family:monospace;font-size:11px;" value="{default_smi}" />
      <button class="cf-btn cf-btn-green" onclick="cfSyncCanvas(false)">🔄 Sync Canvas</button>
      <button class="cf-btn cf-btn-amber" onclick="cfSyncCanvas(true)">🔒 Lock Selected</button>
    </div>
    <div id="cf-desc-bar" class="cf-desc-grid"></div>
    <details style="margin-top:8px;font-size:11px;border:1px solid #bae6fd;border-radius:6px;padding:6px 10px;background:#f0f9ff;">
      <summary style="cursor:pointer;font-weight:700;color:#0369a1;">🔒 Atom-Indexed Preview, Substructure Lock &amp; Lab Inventory Stock</summary>
      <div id="cf-atom-svg" style="background:#fff;border:1px solid #bae6fd;border-radius:6px;margin:6px 0;text-align:center;"></div>
      <div class="cf-input-row">
        <input id="cf-lock-smarts" class="cf-input" style="font-family:monospace;" placeholder="Frozen Substructure SMARTS (auto-filled when locking atoms or type e.g. c1ncsc1)" />
      </div>
      <textarea id="cf-stock-input" rows="2" class="cf-input" style="width:100%;box-sizing:border-box;font-family:monospace;margin-top:4px;" placeholder="Paste in-house Lab Stock SMILES (one per line)..."></textarea>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
        <span id="cf-stock-msg" style="color:#0d9488;font-weight:600;"></span>
        <button class="cf-btn" style="padding:4px 10px;font-size:11px;" onclick="cfRegisterStock()">📦 Register Lab Stock</button>
      </div>
    </details>
  </div>

  <!-- RIGHT PANE: 02 CONFIGURE, ANALYZE & ORGANIC CHEMIST AI AGENT -->
  <div class="cf-pane">
    <div class="cf-sec-hdr">
      <span>02 · CONFIGURE, ONLINE ORD VALIDATION &amp; AI AGENT</span>
      <span style="display:flex;align-items:center;gap:6px;">
        <span id="cf-ord-badge" style="font-size:10px;background:#ccfbf1;color:#0f766e;padding:2px 7px;border-radius:10px;font-weight:700;">ORD Index: {init_ord_cnt} rxns</span>
        <span id="cf-ram-badge" style="font-size:10px;background:#e0f2fe;color:#0369a1;padding:2px 7px;border-radius:10px;font-weight:700;">RAM: {init_ram_mb} MB</span>
        <button class="cf-chip" style="padding:2px 7px;font-size:10px;" onclick="cfFreeMemory()" title="Purge search tree, trim C++ heap &amp; run GC">🧹 Free RAM</button>
      </span>
    </div>

    <!-- GLOBAL GOOGLE GENAI MODEL & THINKING LEVEL SELECTOR -->
    <div class="cf-model-bar">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;font-size:10.5px;font-weight:800;color:#0369a1;">
        <span>🧠 GLOBAL GOOGLE GENAI MODEL &amp; THINKING LEVEL (APPLIES TO ALL TABS)</span>
        <span id="cf-active-model-pill" style="background:#ccfbf1;color:#0f766e;padding:1px 7px;border-radius:10px;font-size:10px;">gemini-3.8-flash · medium</span>
      </div>
      <div class="cf-model-row">
        <div>
          <select id="cf-global-model" class="cf-model-select" onchange="cfOnModelChange(true)">
            {model_options_html}
          </select>
        </div>
        <div>
          <select id="cf-global-thinking" class="cf-model-select" onchange="cfOnThinkingChange()">
            <option value="low">Thinking: Low (Fast)</option>
            <option value="medium" selected>Thinking: Medium (Balanced)</option>
            <option value="high">Thinking: High (Deep)</option>
          </select>
        </div>
      </div>
      <div class="cf-model-summary">
        <span id="cf-model-summary-text">ℹ️ {default_summary}</span>
      </div>
    </div>

    <div class="cf-tabs">
      <button id="cf-tab-btn-synthesis" class="cf-tab active" onclick="cfSwitchTab('synthesis', this)">Synthesis</button>
      <button id="cf-tab-btn-agent" class="cf-tab" onclick="cfSwitchTab('agent', this)">🤖 Chemist AI Agent</button>
      <button id="cf-tab-btn-reaction" class="cf-tab" onclick="cfSwitchTab('reaction', this)">Reaction</button>
      <button id="cf-tab-btn-qa" class="cf-tab" onclick="cfSwitchTab('qa', this)">Structure Q&amp;A</button>
      <button id="cf-tab-btn-analytical" class="cf-tab" onclick="cfSwitchTab('analytical', this)">Analytical</button>
    </div>

    <!-- TAB 1: SYNTHESIS + INLINE SEARCH CONFIGURATION -->
    <div id="cf-panel-synthesis">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
        <div style="display:flex;gap:5px;align-items:center;font-size:11px;flex-wrap:wrap;">
          <b style="color:#0369a1;">RANK BY:</b>
          <button class="cf-mode-pill active" onclick="cfSetRankMode('cro', this)">CRO-Practical</button>
          <button class="cf-mode-pill" onclick="cfSetRankMode('discovery', this)">Discovery (Fast)</button>
          <button class="cf-mode-pill" onclick="cfSetRankMode('evidence', this)" title="Weights Route-Level ORD Empirical Confidence Score (S_ORD) highest">Evidence-Backed (ORD)</button>
          <button class="cf-mode-pill" onclick="cfSetRankMode('green', this)">Green / Safe</button>
          <button class="cf-mode-pill" onclick="cfSetRankMode('strategy', this)" title="Prioritizes SAScore/SPS complexity-building key steps and penalizes unproductive loops">Strategy / Complexity-Building</button>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button id="cf-plan-btn" class="cf-btn cf-btn-green" style="padding:7px 14px;font-size:12.5px;" onclick="cfRunSynthesis()">🚀 Plan Synthesis Route</button>
          <button id="cf-reclass-btn" class="cf-btn cf-btn-amber" style="padding:7px 12px;font-size:12px;" onclick="cfReclassifyRoutes()" title="Re-validate ORD precedents, re-classify existing routes &amp; rewrite experimental procedures using the selected model without re-running AiZynthFinder">🧠 Re-classify &amp; Re-score (ORD + AI)</button>
        </div>
      </div>
      <details style="margin-bottom:10px;">
        <summary style="cursor:pointer;font-size:11px;font-weight:700;color:#0284c7;margin-bottom:6px;">⚙️ Search, ORD Parquet Shard Sync &amp; Scoring Settings</summary>
        <div class="cf-cfg-grid">
          <div class="cf-cfg-item">
            <label>Search Algorithm</label>
            <select id="cfg-algo">
              <option value="retrostar" selected>Retro* (AND-OR Best-First)</option>
              <option value="mcts">MCTS (Monte Carlo Tree Search)</option>
            </select>
          </div>
          <div class="cf-cfg-item">
            <label>Max Synthetic Steps</label>
            <select id="cfg-steps">
              <option value="3">3 steps</option><option value="4">4 steps</option><option value="5">5 steps</option>
              <option value="6" selected>6 steps</option><option value="8">8 steps</option><option value="10">10 steps</option>
            </select>
          </div>
          <div class="cf-cfg-item">
            <label>Search Time Limit (sec)</label>
            <select id="cfg-time">
              <option value="30">30s</option><option value="60">60s</option><option value="120" selected>120s</option><option value="300">300s</option>
            </select>
          </div>
          <div class="cf-cfg-item">
            <label>Iteration Limit</label>
            <select id="cfg-iter">
              <option value="50">50</option><option value="100" selected>100</option><option value="200">200</option><option value="500">500</option>
            </select>
          </div>
          <div class="cf-cfg-item">
            <label>Template Cutoff</label>
            <select id="cfg-cutoff">
              <option value="0.90">0.90 (Conservative)</option><option value="0.95">0.95 (Standard)</option><option value="0.995" selected>0.995 (Broad)</option>
            </select>
          </div>
          <div class="cf-cfg-item">
            <label>Target Scale (g) &amp; Max Routes</label>
            <div style="display:flex;gap:4px;">
              <input id="cfg-scale" type="number" step="0.5" value="1.0" />
              <input id="cfg-max-routes" type="number" value="8" />
            </div>
          </div>
          <div class="cf-cfg-item">
            <label>Exclude Reaction Types (comma-separated)</label>
            <input id="cfg-exclude" placeholder="e.g. Stille, Azide, Ozonolysis, Grignard" />
          </div>
          <div class="cf-cfg-item">
            <label>ORD Parquet Shards (`ord-data` HF)</label>
            <div style="display:flex;gap:4px;align-items:center;">
              <button id="cf-ord-sync-btn" class="cf-btn" style="padding:4px 8px;font-size:10.5px;width:100%;" onclick="cfSyncOrdShards(1)" title="Stream &amp; index an ORD .parquet shard from Hugging Face into /dev/shm DuckDB">📥 Sync +1 HF ORD Parquet Shard</button>
            </div>
          </div>
          <div class="cf-cfg-item" style="display:flex;flex-direction:column;justify-content:center;gap:4px;">
            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" id="cfg-first" style="width:auto;" /> Stop at First Solved</label>
            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" id="cfg-gemini" checked style="width:auto;" /> Gemini + ORD Grounding</label>
            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox" id="cfg-common-stock" checked style="width:auto;" /> Treat Common Lab Reagents as On-Shelf</label>
          </div>
        </div>
      </details>
      <div id="cf-synthesis-out" style="font-size:12px;color:#475569;">
        Draw or load your target molecule on the left, adjust any search settings above, and click <b>🚀 Plan Synthesis Route</b>. Every unique step is prefetched in parallel against the live Open Reaction Database API and validated in DuckDB for reaction-center similarity, HTE negative-data failure rates, grounded forward conditions, and a Route-Level <b>ORD Empirical Confidence Score</b>.
      </div>
    </div>

    <!-- TAB 2: ORGANIC CHEMIST AI AGENT -->
    <div id="cf-panel-agent" style="display:none;">
      <div style="display:flex;justify-content:space-between;align-items:center;background:#f0f9ff;border:1px solid #bae6fd;border-radius:7px;padding:7px 10px;margin-bottom:8px;font-size:11px;">
        <span id="cf-agent-ctx-bar" style="color:#0369a1;font-weight:700;">🤖 Session: Ready — run synthesis or ask a question below</span>
        <button class="cf-chip" style="padding:2px 8px;font-size:10px;" onclick="cfResetAgentChat()">🗑️ Clear Chat</button>
      </div>
      <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px;">
        <button class="cf-chip" onclick="cfAskAgentQuick('Compare the top retrosynthesis routes by ORD Empirical Confidence Score, bottleneck steps, and HTE failure risks.')">🔬 Compare by ORD Confidence</button>
        <button class="cf-chip" onclick="cfAskAgentQuick('Generate a 5.0 g scaled ELN stoichiometry table for Route 1.')">📋 5g ELN Table (Route A)</button>
        <button class="cf-chip" onclick="cfAskAgentQuick('Which steps have ORD negative-data alerts or competing side-products, and how can we mitigate them?')">⚠️ Audit ORD Negative Data</button>
        <button class="cf-chip" onclick="cfAskAgentQuick('Re-run the retrosynthesis in balanced (Evidence-Backed) mode with max 5 steps.')">🔄 Re-run (Evidence Mode, ≤5 Steps)</button>
      </div>
      <div id="cf-agent-log" class="cf-agent-log">
        <div class="cf-msg-bot">
          <b>👨‍🔬 Organic Chemist AI Agent:</b> Welcome! Once you run a synthesis (or right now), ask me to compare routes by their <b>ORD Empirical Confidence Score</b>, audit HTE negative-data failure rates, generate scaled ELN stoichiometry tables, or re-run AiZynthFinder with new reaction constraints.
        </div>
      </div>
      <div class="cf-input-row">
        <input id="cf-agent-input" class="cf-input" placeholder="Chat with the Organic Chemist AI Agent about ORD precedents, routes, conditions, ELN scale-up, or re-running..." onkeydown="if(event.key==='Enter') cfSendAgentMsg();" />
        <button id="cf-agent-send-btn" class="cf-btn cf-btn-green" onclick="cfSendAgentMsg()">Send</button>
      </div>
    </div>

    <!-- TAB 3: REACTION ANALYSIS -->
    <div id="cf-panel-reaction" style="display:none;">
      <div style="margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:12px;color:#475569;">Draw a full reaction (<code>Reactants &gt;&gt; Product</code>) in Ketcher or paste in the SMILES box.</span>
        <button class="cf-btn" onclick="cfRunReaction()">Analyze Reaction + ORD Precedent</button>
      </div>
      <div id="cf-reaction-out"></div>
    </div>

    <!-- TAB 4: STRUCTURE Q&A & LITERATURE CHAT -->
    <div id="cf-panel-qa" style="display:none;">
      <div class="cf-input-row">
        <input id="cf-qa-input" class="cf-input" placeholder="Ask about route comparison, ORD yields, literature conditions, or side-products..." />
        <button class="cf-btn" onclick="cfRunQA()">Ask Gemini</button>
      </div>
      <div id="cf-qa-out"></div>
    </div>

    <!-- TAB 5: ANALYTICAL LC-MS & REPORT READER -->
    <div id="cf-panel-analytical" style="display:none;">
      <textarea id="cf-analytical-input" rows="3" class="cf-input" style="width:100%;box-sizing:border-box;margin-bottom:6px;font-family:monospace;" placeholder="Optional: Paste raw LC-MS / NMR / HPLC peak text here, or leave blank to compute expected monoisotopic adduct ions..."></textarea>
      <button class="cf-btn" onclick="cfRunAnalytical()">Compute Ions / Read Report</button>
      <div id="cf-analytical-out" style="margin-top:8px;"></div>
    </div>
  </div>
</div>

<script>
window.cfRankMode = 'cro';
window.cfModelCatalog = {catalog_json};

const THINKING_LABELS = {{
  minimal: "Thinking: Minimal (Fastest)",
  low: "Thinking: Low (Fast)",
  medium: "Thinking: Medium (Balanced)",
  high: "Thinking: High (Deep Reasoning)",
  xhigh: "Thinking: XHigh (Max Frontier)"
}};

async function cfOnModelChange(resetToDefaultThinking) {{
  const mSel = document.getElementById("cf-global-model");
  const tSel = document.getElementById("cf-global-thinking");
  const sumEl = document.getElementById("cf-model-summary-text");
  const pillEl = document.getElementById("cf-active-model-pill");
  const modelId = mSel.value;
  const spec = window.cfModelCatalog[modelId] || window.cfModelCatalog["gemini-3.8-flash"];
  const prevThinking = tSel.value;
  const supported = spec.supported_thinking || ["low", "medium", "high"];
  const targetThinking = resetToDefaultThinking
    ? spec.default_thinking
    : (supported.includes(prevThinking) ? prevThinking : spec.default_thinking);

  tSel.innerHTML = supported.map(lvl => {{
    const isDef = (lvl === spec.default_thinking) ? " · Default" : "";
    const selAttr = (lvl === targetThinking) ? "selected" : "";
    return `<option value="${{lvl}}" ${{selAttr}}>${{THINKING_LABELS[lvl] || lvl}}${{isDef}}</option>`;
  }}).join("");

  if (sumEl) sumEl.innerText = "ℹ️ " + spec.summary;
  if (pillEl) pillEl.innerText = `${{modelId}} · ${{tSel.value}}`;
  try {{
    await google.colab.kernel.invokeFunction("chemforge.set_model", [modelId, tSel.value], {{}});
  }} catch (e) {{}}
}}

async function cfOnThinkingChange() {{
  const mSel = document.getElementById("cf-global-model");
  const tSel = document.getElementById("cf-global-thinking");
  const pillEl = document.getElementById("cf-active-model-pill");
  if (pillEl) pillEl.innerText = `${{mSel.value}} · ${{tSel.value}}`;
  try {{
    await google.colab.kernel.invokeFunction("chemforge.set_model", [mSel.value, tSel.value], {{}});
  }} catch (e) {{}}
}}

(async function initKetcherFrame() {{
  const frame = document.getElementById("cf-ketcher-frame");
  const statusEl = document.getElementById("cf-status");
  let b64 = "{ketcher_b64_html}";
  try {{
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    frame.srcdoc = new TextDecoder("utf-8").decode(bytes);
    b64 = null;
  }} catch (err) {{
    console.error("Ketcher srcdoc init error:", err);
  }}
  let attempts = 0;
  const timer = setInterval(async () => {{
    attempts++;
    try {{
      const cw = frame.contentWindow;
      if (cw && cw.ketcher && typeof cw.ketcher.setMolecule === "function") {{
        clearInterval(timer);
        window.cfKetcher = cw.ketcher;
        statusEl.innerText = "✅ Ketcher Ready";
        const initSmi = document.getElementById("cf-smiles-box").value.trim();
        if (initSmi) {{
          await cw.ketcher.setMolecule(initSmi);
          cfSyncCanvas(false);
        }}
        return;
      }}
    }} catch (e) {{}}
    if (attempts >= 60) {{
      clearInterval(timer);
    }}
  }}, 350);
  cfOnModelChange(true);
  cfSyncCanvas(false);
}})();

function cfUpdateRamBadge(ramMb) {{
  if (!ramMb) return;
  const el = document.getElementById("cf-ram-badge");
  if (el) el.innerText = `RAM: ${{ramMb}} MB`;
}}

function cfUpdateOrdBadge(ordCount, prefetch) {{
  if (!ordCount) return;
  const el = document.getElementById("cf-ord-badge");
  const statusEl = document.getElementById("cf-status");
  if (prefetch && prefetch.queried_steps > 0) {{
    const addCnt = prefetch.added_records || 0;
    const sec = Number(prefetch.elapsed_sec || 0).toFixed(2);
    if (el) {{
      el.innerText = `ORD Index: ${{ordCount}} rxns (+${{addCnt}} live · ${{sec}}s)`;
      el.title = `Queried ${{prefetch.queried_steps}} unique steps against live open-reaction-database.org API in ${{sec}}s`;
    }}
    if (statusEl) {{
      statusEl.innerText = `🌐 +${{addCnt}} live ORD hits (${{prefetch.queried_steps}} steps in ${{sec}}s)`;
    }}
  }} else {{
    if (el) el.innerText = `ORD Index: ${{ordCount}} rxns`;
  }}
}}

async function cfSyncOrdShards(nShards) {{
  const btn = document.getElementById("cf-ord-sync-btn");
  if (btn) {{
    btn.disabled = true;
    btn.innerText = "⏳ Streaming HF Parquet Shard...";
  }}
  try {{
    const res = await google.colab.kernel.invokeFunction("chemforge.sync_ord_shards", [nShards || 1], {{}});
    const data = res.data["application/json"];
    cfUpdateOrdBadge(data.indexed_count);
    cfUpdateRamBadge(data.ram_mb);
    if (btn) {{
      btn.innerText = data.error ? "⚠️ Sync Error (Retry)" : `✅ Indexed (${{data.indexed_count}} rxns) · +1 Shard`;
    }}
  }} catch (err) {{
    if (btn) btn.innerText = "⚠️ Sync Failed";
  }} finally {{
    if (btn) btn.disabled = false;
  }}
}}

async function cfFreeMemory() {{
  const res = await google.colab.kernel.invokeFunction("chemforge.free_memory", [], {{}});
  const data = res.data["application/json"];
  cfUpdateRamBadge(data.ram_mb);
  document.getElementById("cf-status").innerText = data.message;
}}

function cfEscapeHtml(text) {{
  const span = document.createElement("span");
  span.textContent = String(text ?? "");
  return span.innerHTML;
}}

function cfRenderDescriptors(d) {{
  const el = document.getElementById("cf-desc-bar");
  if (!d || d.error) {{
    el.innerHTML = `<div style="grid-column:span 4;color:#dc2626;">⚠️ ${{cfEscapeHtml(d ? d.error : 'Invalid structure')}}</div>`;
    return;
  }}
  const ro5 = d.ro5_pass
    ? `<span style="color:#0d9488;font-weight:700;">PASS</span>`
    : `<span style="color:#dc2626;font-weight:700;">FAIL (${{d.ro5_flags}})</span>`;
  el.innerHTML = `
    <div class="cf-desc-item"><b>FORMULA</b>${{d.formula}}</div>
    <div class="cf-desc-item"><b>MOL. WEIGHT</b>${{d.mw}}</div>
    <div class="cf-desc-item"><b>EXACT MASS</b>${{d.exact_mass}}</div>
    <div class="cf-desc-item"><b>CLOGP</b>${{d.clogp}}</div>
    <div class="cf-desc-item"><b>TPSA</b>${{d.tpsa}} Å²</div>
    <div class="cf-desc-item"><b>HBD / HBA</b>${{d.hbd}} / ${{d.hba}}</div>
    <div class="cf-desc-item"><b>ROT / RINGS</b>${{d.rotatable}} / ${{d.rings}}</div>
    <div class="cf-desc-item"><b>LIPINSKI RO5</b>${{ro5}}</div>
  `;
}}

async function cfSyncCanvas(lockSelected) {{
  let structStr = document.getElementById("cf-smiles-box").value.trim();
  let selAtoms = [];
  if (window.cfKetcher) {{
    try {{
      const kSmi = await window.cfKetcher.getSmiles();
      if (kSmi && kSmi.trim()) structStr = kSmi.trim();
      if (lockSelected) {{
        structStr = await window.cfKetcher.getMolfile();
        const sel = window.cfKetcher.editor ? window.cfKetcher.editor.selection() : null;
        if (sel && Array.isArray(sel.atoms)) selAtoms = sel.atoms;
      }}
    }} catch (e) {{}}
  }}
  const manualSmarts = lockSelected ? "" : document.getElementById("cf-lock-smarts").value.trim();
  const res = await google.colab.kernel.invokeFunction("chemforge.sync_canvas", [structStr, JSON.stringify(selAtoms), manualSmarts], {{}});
  const data = res.data["application/json"];
  if (!data.error) {{
    document.getElementById("cf-smiles-box").value = data.smiles;
    document.getElementById("cf-lock-smarts").value = data.locked_smarts || "";
    document.getElementById("cf-atom-svg").innerHTML = data.preview_svg || "";
    cfRenderDescriptors(data.descriptors);
    cfUpdateRamBadge(data.ram_mb);
    document.getElementById("cf-status").innerText = data.locked_atoms && data.locked_atoms.length
      ? `🔒 Locked ${{data.locked_atoms.length}} atoms`
      : "✅ Synced with Ketcher";
  }}
}}

async function cfLoadSingle() {{
  const q = document.getElementById("cf-name-a").value.trim();
  if (!q) return;
  document.getElementById("cf-status").innerText = "⏳ Resolving...";
  const res = await google.colab.kernel.invokeFunction("chemforge.load_id", [q, ""], {{}});
  const data = res.data["application/json"];
  if (data.error) {{
    document.getElementById("cf-status").innerText = "⚠️ " + data.error;
  }} else {{
    document.getElementById("cf-smiles-box").value = data.smiles;
    if (window.cfKetcher) await window.cfKetcher.setMolecule(data.smiles);
    cfRenderDescriptors(data.descriptors);
    document.getElementById("cf-status").innerText = "✅ " + data.source;
  }}
}}

function cfQuick(name) {{
  document.getElementById("cf-name-a").value = name;
  cfLoadSingle();
}}

async function cfLoadPair() {{
  const qa = document.getElementById("cf-react-a").value.trim();
  const qb = document.getElementById("cf-react-b").value.trim();
  if (!qa || !qb) return;
  document.getElementById("cf-status").innerText = "⏳ Loading A + B...";
  const res = await google.colab.kernel.invokeFunction("chemforge.load_id", [qa, qb], {{}});
  const data = res.data["application/json"];
  if (!data.error) {{
    document.getElementById("cf-smiles-box").value = data.smiles;
    if (window.cfKetcher) await window.cfKetcher.setMolecule(data.smiles);
    cfRenderDescriptors(data.descriptors);
    document.getElementById("cf-status").innerText = "✅ " + data.source;
  }}
}}

function cfSwitchTab(tab, btn) {{
  ["synthesis", "agent", "reaction", "qa", "analytical"].forEach(t => {{
    document.getElementById("cf-panel-" + t).style.display = (t === tab) ? "block" : "none";
  }});
  btn.parentElement.querySelectorAll(".cf-tab").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  if (tab === "agent") cfRefreshAgentStatus();
}}

function cfOpenAgentWithRoutes() {{
  const btn = document.getElementById("cf-tab-btn-agent");
  if (btn) cfSwitchTab("agent", btn);
}}

async function cfCopyStepELN(button) {{
  const text = button.dataset.eln;
  const label = button.textContent;
  let copied = false;
  button.disabled = true;
  try {{
    await navigator.clipboard.writeText(text);
    copied = true;
  }} catch (_) {{
    const field = document.createElement("textarea");
    field.value = text;
    field.style.position = "fixed";
    field.style.opacity = "0";
    document.body.appendChild(field);
    field.select();
    try {{ copied = document.execCommand("copy"); }} catch (_) {{ /* Clipboard blocked in this frame. */ }}
    field.remove();
  }}
  button.textContent = copied ? "✓ Copied step ELN" : "Clipboard unavailable";
  setTimeout(() => {{ button.textContent = label; button.disabled = false; }}, 1600);
}}

function cfSetRankMode(mode, btn) {{
  window.cfRankMode = mode;
  btn.parentElement.querySelectorAll(".cf-mode-pill").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
}}

async function cfRefreshAgentStatus() {{
  const res = await google.colab.kernel.invokeFunction("chemforge.agent_status", [], {{}});
  const d = res.data["application/json"];
  cfUpdateRamBadge(d.ram_mb);
  cfUpdateOrdBadge(d.ord_index_size);
  const bar = document.getElementById("cf-agent-ctx-bar");
  if (bar) {{
    bar.innerText = d.route_count > 0
      ? `✅ Active Session: ${{d.route_count}} ORD-validated routes (${{d.ranking_mode}}) · Model: ${{d.model}} (${{d.thinking_level}}) | Target: ${{d.target_smiles.slice(0, 30)}}...`
      : `🤖 Active Model: ${{d.model}} (${{d.thinking_level}}) | ORD Index: ${{d.ord_index_size}} rxns | Target: ${{d.target_smiles.slice(0, 30)}}`;
  }}
}}

async function cfRunSynthesis() {{
  await cfSyncCanvas(false);
  const smi = document.getElementById("cf-smiles-box").value.trim();
  const timeLimitSec = parseInt(document.getElementById("cfg-time").value, 10) || 120;
  const maxIter = parseInt(document.getElementById("cfg-iter").value, 10) || 100;
  const algoLabel = document.getElementById("cfg-algo").value === "mcts" ? "MCTS" : "Retro*";
  const useGem = document.getElementById("cfg-gemini").checked;
  const selModel = document.getElementById("cf-global-model").value;
  const selThinking = document.getElementById("cf-global-thinking").value;
  const estTotalSec = Math.min(timeLimitSec + (useGem ? 25 : 15), Math.max(30, Math.round(timeLimitSec * 0.65) + (useGem ? 20 : 15)));
  const cfg = {{
    rank_mode: window.cfRankMode,
    algorithm: document.getElementById("cfg-algo").value,
    max_steps: parseInt(document.getElementById("cfg-steps").value, 10),
    time_limit: timeLimitSec,
    iteration_limit: maxIter,
    cutoff: parseFloat(document.getElementById("cfg-cutoff").value),
    scale_grams: parseFloat(document.getElementById("cfg-scale").value || "1.0"),
    max_routes: parseInt(document.getElementById("cfg-max-routes").value || "8", 10),
    exclude_reactions: document.getElementById("cfg-exclude").value,
    return_first: document.getElementById("cfg-first").checked,
    use_gemini: useGem,
    use_common_lab_reagents: document.getElementById("cfg-common-stock").checked,
    genai_model: selModel,
    genai_thinking: selThinking,
    lock_smarts: document.getElementById("cf-lock-smarts").value.trim()
  }};
  const out = document.getElementById("cf-synthesis-out");
  const planBtn = document.getElementById("cf-plan-btn");
  const reclassBtn = document.getElementById("cf-reclass-btn");
  if (planBtn) planBtn.disabled = true;
  if (reclassBtn) reclassBtn.disabled = true;
  out.innerHTML = `
    <div class="cf-progress-card">
      <div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;font-weight:700;">
        <span id="cf-prog-stage" style="color:#38bdf8;">⚡ Running ${{algoLabel}} Tree Search (USPTO + ZINC Stock + Online ORD Validator)...</span>
        <span id="cf-prog-pct" style="color:#2dd4bf;font-family:monospace;font-size:13px;">0%</span>
      </div>
      <div class="cf-progress-track"><div id="cf-prog-fill" class="cf-progress-fill" style="width:2%;"></div></div>
      <div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#94a3b8;font-family:monospace;">
        <span id="cf-prog-elapsed">Elapsed: 0s</span>
        <span id="cf-prog-eta">Est. remaining: ~${{estTotalSec}}s (Max cap: ${{timeLimitSec}}s)</span>
      </div>
    </div>
  `;
  const startMs = Date.now();
  const progTimer = setInterval(() => {{
    const elapsedSec = Math.floor((Date.now() - startMs) / 1000);
    const fillEl = document.getElementById("cf-prog-fill");
    const pctEl = document.getElementById("cf-prog-pct");
    const elapEl = document.getElementById("cf-prog-elapsed");
    const stageEl = document.getElementById("cf-prog-stage");
    if (!fillEl) return;
    let ratio = elapsedSec / estTotalSec;
    let pct = ratio < 0.85 ? Math.min(85, Math.round(ratio * 100)) : Math.min(96, 85 + Math.round((1 - Math.exp(-(elapsedSec - estTotalSec * 0.85) / 15)) * 11));
    fillEl.style.width = pct + "%";
    pctEl.innerText = pct + "%";
    elapEl.innerText = `Elapsed: ${{elapsedSec}}s`;
    if (elapsedSec < estTotalSec * 0.65) {{
      stageEl.innerText = `⚡ Exploring ${{algoLabel}} disconnection tree (up to ${{maxIter}} iterations)...`;
    }} else if (elapsedSec < estTotalSec * 0.65 + 4) {{
      stageEl.innerText = `🌐 Parallel 16-thread online ORD API prefetch (matching reactants & products across 2M+ rxns)...`;
    }} else {{
      stageEl.innerText = useGem
        ? `🔬 Grounding experimental procedures in live ORD precedents with ${{selModel}}...`
        : `🔬 Computing ORD Empirical Confidence Scores & purging tree memory...`;
    }}
  }}, 500);
  try {{
    const res = await google.colab.kernel.invokeFunction("chemforge.run_synthesis", [smi, JSON.stringify(cfg)], {{}});
    clearInterval(progTimer);
    const data = res.data["application/json"];
    out.innerHTML = data.error ? `<div style="color:#dc2626;font-weight:600;">⚠️ ${{cfEscapeHtml(data.error)}}</div>` : data.html;
    cfUpdateRamBadge(data.ram_mb);
    cfUpdateOrdBadge(data.ord_index_size, data.ord_prefetch);
    cfRefreshAgentStatus();
  }} catch (err) {{
    clearInterval(progTimer);
    out.innerHTML = `<div style="color:#dc2626;font-weight:600;">⚠️ Kernel error: ${{cfEscapeHtml(err)}}</div>`;
  }} finally {{
    if (planBtn) planBtn.disabled = false;
    if (reclassBtn) reclassBtn.disabled = false;
  }}
}}

async function cfReclassifyRoutes() {{
  const selModel = document.getElementById("cf-global-model").value;
  const selThinking = document.getElementById("cf-global-thinking").value;
  const out = document.getElementById("cf-synthesis-out");
  const reclassBtn = document.getElementById("cf-reclass-btn");
  const planBtn = document.getElementById("cf-plan-btn");
  if (reclassBtn) reclassBtn.disabled = true;
  if (planBtn) planBtn.disabled = true;
  document.getElementById("cfg-gemini").checked = true;
  const oldBanner = document.getElementById("cf-reclass-banner");
  if (oldBanner) oldBanner.remove();
  out.insertAdjacentHTML("afterbegin", `
    <div id="cf-reclass-banner" class="cf-progress-card" style="margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;font-weight:700;">
        <span style="color:#38bdf8;">🌐 Prefetching uncached online ORD steps (16 threads) &amp; rewriting procedures with ${{selModel}} (${{selThinking}})...</span>
        <span style="color:#2dd4bf;font-family:monospace;font-size:11px;">Zero tree-search overhead</span>
      </div>
      <div class="cf-progress-track"><div class="cf-progress-fill" style="width:100%;"></div></div>
    </div>
  `);
  try {{
    const res = await google.colab.kernel.invokeFunction("chemforge.reclassify_routes", [selModel, selThinking, window.cfRankMode], {{}});
    const data = res.data["application/json"];
    const banner = document.getElementById("cf-reclass-banner");
    if (banner) banner.remove();
    if (data.error) {{
      out.insertAdjacentHTML("afterbegin", `<div style="color:#dc2626;font-weight:600;margin-bottom:8px;">⚠️ ${{cfEscapeHtml(data.error)}}</div>`);
    }} else {{
      out.innerHTML = data.html;
    }}
    cfUpdateRamBadge(data.ram_mb);
    cfUpdateOrdBadge(data.ord_index_size, data.ord_prefetch);
    cfRefreshAgentStatus();
  }} catch (err) {{
    const banner = document.getElementById("cf-reclass-banner");
    if (banner) banner.remove();
    out.insertAdjacentHTML("afterbegin", `<div style="color:#dc2626;font-weight:600;margin-bottom:8px;">⚠️ Kernel error: ${{cfEscapeHtml(err)}}</div>`);
  }} finally {{
    if (reclassBtn) reclassBtn.disabled = false;
    if (planBtn) planBtn.disabled = false;
  }}
}}

async function cfSendAgentMsg() {{
  const inp = document.getElementById("cf-agent-input");
  const log = document.getElementById("cf-agent-log");
  const sendBtn = document.getElementById("cf-agent-send-btn");
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = "";
  const smi = document.getElementById("cf-smiles-box").value.trim();
  const selModel = document.getElementById("cf-global-model").value;
  const selThinking = document.getElementById("cf-global-thinking").value;
  log.insertAdjacentHTML("beforeend", `<div class="cf-msg-user">${{cfEscapeHtml(msg)}}</div>`);
  const waitId = "cf-wait-" + Date.now();
  log.insertAdjacentHTML("beforeend", `<div id="${{waitId}}" class="cf-msg-bot">⏳ <b>Chemist AI Agent</b> (<code>${{selModel}}</code> · <code>${{selThinking}}</code>) is analyzing ORD-validated routes &amp; running tools...</div>`);
  log.scrollTop = log.scrollHeight;
  if (sendBtn) sendBtn.disabled = true;
  try {{
    const res = await google.colab.kernel.invokeFunction("chemforge.agent_chat", [msg, smi, selModel, selThinking], {{}});
    const data = res.data["application/json"];
    const waitEl = document.getElementById(waitId);
    if (waitEl) waitEl.outerHTML = `<div class="cf-msg-bot">${{data.reply_html}}</div>`;
    if (data.updated_synthesis_html) {{
      document.getElementById("cf-synthesis-out").innerHTML = data.updated_synthesis_html;
    }}
    cfUpdateRamBadge(data.ram_mb);
    cfRefreshAgentStatus();
  }} catch (err) {{
    const waitEl = document.getElementById(waitId);
    if (waitEl) waitEl.outerHTML = `<div class="cf-msg-bot" style="color:#dc2626;">⚠️ Error: ${{cfEscapeHtml(err)}}</div>`;
  }} finally {{
    if (sendBtn) sendBtn.disabled = false;
    log.scrollTop = log.scrollHeight;
  }}
}}

function cfAskAgentQuick(promptText) {{
  document.getElementById("cf-agent-input").value = promptText;
  cfSendAgentMsg();
}}

async function cfResetAgentChat() {{
  const res = await google.colab.kernel.invokeFunction("chemforge.agent_reset", [], {{}});
  const data = res.data["application/json"];
  cfUpdateRamBadge(data.ram_mb);
  document.getElementById("cf-agent-log").innerHTML = `<div class="cf-msg-bot">🧹 Chat history and cached objects cleared. Ask a new question about your ORD-validated routes!</div>`;
}}

async function cfRunReaction() {{
  if (window.cfKetcher) {{
    try {{
      const kSmi = await window.cfKetcher.getSmiles();
      if (kSmi) document.getElementById("cf-smiles-box").value = kSmi;
    }} catch (e) {{}}
  }}
  const smi = document.getElementById("cf-smiles-box").value.trim();
  const selModel = document.getElementById("cf-global-model").value;
  const selThinking = document.getElementById("cf-global-thinking").value;
  const out = document.getElementById("cf-reaction-out");
  out.innerHTML = `⏳ <b style='color:#0284c7;'>Prefetching live online ORD precedents &amp; analyzing with ${{selModel}} (${{selThinking}})...</b>`;
  const res = await google.colab.kernel.invokeFunction("chemforge.analyze_reaction", [smi, selModel, selThinking], {{}});
  const data = res.data["application/json"];
  out.innerHTML = data.error ? `<div style="color:#dc2626;">⚠️ ${{cfEscapeHtml(data.error)}}</div>` : data.html;
}}

async function cfRunQA() {{
  const smi = document.getElementById("cf-smiles-box").value.trim();
  const q = document.getElementById("cf-qa-input").value.trim();
  const selModel = document.getElementById("cf-global-model").value;
  const selThinking = document.getElementById("cf-global-thinking").value;
  const out = document.getElementById("cf-qa-out");
  out.innerHTML = `⏳ <b style='color:#0284c7;'>Querying ${{selModel}} (${{selThinking}}) with active structure &amp; ORD-validated routes...</b>`;
  const res = await google.colab.kernel.invokeFunction("chemforge.structure_qa", [smi, q, selModel, selThinking], {{}});
  out.innerHTML = res.data["application/json"].html;
}}

async function cfRunAnalytical() {{
  const smi = document.getElementById("cf-smiles-box").value.trim();
  const rep = document.getElementById("cf-analytical-input").value.trim();
  const selModel = document.getElementById("cf-global-model").value;
  const selThinking = document.getElementById("cf-global-thinking").value;
  const out = document.getElementById("cf-analytical-out");
  out.innerHTML = `⏳ <b style='color:#0284c7;'>Computing exact monoisotopic LC-MS adducts &amp; running ${{selModel}} (${{selThinking}})...</b>`;
  const res = await google.colab.kernel.invokeFunction("chemforge.analytical", [smi, rep, selModel, selThinking], {{}});
  out.innerHTML = res.data["application/json"].html;
}}

async function cfRegisterStock() {{
  const raw = document.getElementById("cf-stock-input").value;
  const res = await google.colab.kernel.invokeFunction("chemforge.register_stock", [raw], {{}});
  document.getElementById("cf-stock-msg").innerText = res.data["application/json"].message;
}}
</script>
"""
    clear_output(wait=True)
    display(HTML(unified_workbench_html))

    del ketcher_b64_html
    del unified_workbench_html


def _format_agent_reply(reply_text: str) -> str:
    """Keep the requested formatting but do not execute model-generated markup."""
    formatted = html.escape(reply_text)
    for tag in ("b", "ul", "li", "code", "p"):
        formatted = formatted.replace(f"&lt;{tag}&gt;", f"<{tag}>")
        formatted = formatted.replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    return formatted


class _Workbench:
    """Keep state shared across callbacks for one workbench launch."""

    def __init__(self, state: WorkbenchState, finder, ord_validator: ORDRetrosynthesisValidator):
        self.state = state
        self.finder = finder
        self.ord_validator = ord_validator

    def _eln_route_plans(self, route_idx, target_scale_grams, assumed_step_yield, step_filter):
        res = self.state.latest_retro_results
        routes = res.get("routes", [])
        if not 1 <= int(route_idx) <= len(routes):
            return [], "Route index out of range."
        route = routes[int(route_idx) - 1]
        plans = _scale_route_eln(route, res.get("target_smiles", ""), float(target_scale_grams), assumed_step_yield)
        if not plans and any(
            next(iter((step.get("ord_validation") or {}).get("top_precedents") or []), {}).get("provider") == "CbyD"
            and any(
                row.get("role") == "Reactant" and not row.get("equivalents")
                for row in step.get("reagent_stoichiometry") or []
            )
            for step in route.get("steps", [])
        ):
            return [], "Check DOI for stoichiometry/time/workup before generating a numeric CbyD ELN charge."
        if not plans and assumed_step_yield is None and any(
            _clamp_step_yield(step.get("assumed_step_yield")) in (None, 0.0)
            for step in route.get("steps", [])
        ):
            return [], "No positive empirical yield is available for every step; provide an explicit scenario yield to build the ELN."
        if step_filter is not None:
            plans = [plan for plan in plans if plan["step"]["step_index"] == step_filter]
        return plans, "No scalable step for this target and scale." if not plans else ""


    def generate_eln_markdown(self, route_idx: int = 1, target_scale_grams: float = 1.0, assumed_step_yield: Optional[float] = None, step_filter: Optional[int] = None) -> str:
        """Use per-step evidence unless the caller explicitly supplies a scenario yield."""
        plans, error = self._eln_route_plans(route_idx, target_scale_grams, assumed_step_yield, step_filter)
        if error:
            return error
        heading = f"### 📋 Scaled ELN Stoichiometry — Route #{int(route_idx)} ({float(target_scale_grams):.2f} g target)"
        return "\n\n".join([heading] + [_eln_step_text(plan, markdown=True) for plan in plans])


    def generate_eln_html_table(self, route_idx: int = 1, target_scale_grams: float = 1.0, assumed_step_yield: Optional[float] = None, step_filter: Optional[int] = None) -> str:
        plans, error = self._eln_route_plans(route_idx, target_scale_grams, assumed_step_yield, step_filter)
        if error:
            return f"<div style='color:#dc2626;'>{html.escape(error)}</div>"
        heading = f"<b>📋 Scaled ELN Stoichiometry — Route #{int(route_idx)} ({float(target_scale_grams):.2f} g target)</b>" if step_filter is None else ""
        return heading + "".join(_render_eln_step_html(plan, with_heading=step_filter is None) for plan in plans)

    def _build_retrosynthesis_context(self) -> str:
        """Build a compact route and ORD precedent context for the chat agent."""
        res = deduplicate_retro_state(self.state.latest_retro_results)
        target = res.get("target_smiles") or self.state.canvas_target_smiles or "None"
        routes = res.get("routes", [])
        cbyd_size = getattr(self.ord_validator, "get_cbyd_index_size", lambda: 0)()
        if not routes:
            return (
                f"Active Target SMILES: {target}\nORD DuckDB Index Size: "
                f"{self.ord_validator.get_index_size()} reactions | CbyD Index Size: "
                f"{cbyd_size} reactions\nNo retrosynthesis routes have been generated yet."
            )
        lines = [
            f"Active Target SMILES: {target}",
            f"Search Algorithm: {res.get('search_algorithm', 'retrostar')} | Ranking Profile: {res.get('ranking_mode', 'balanced')} | Target Scale: {res.get('target_scale_grams', 1.0)} g",
            f"ORD DuckDB Index Size: {self.ord_validator.get_index_size()} reactions | CbyD Index Size: {cbyd_size} reactions | Locked Substructure SMARTS: {res.get('locked_substructure_smarts') or 'None'} | Excluded Reactions: {', '.join(res.get('excluded_reactions') or []) or 'None'}",
            f"Total Extracted Routes: {len(routes)}\n",
        ]
        for idx, r in enumerate(routes, start=1):
            letter = chr(64 + idx) if idx <= 26 else str(idx)
            sm = r.get("starting_materials", {})
            precs = ", ".join(f"{p['smiles']} ({p['source']})" for p in sm.get("precursors", []))
            lines.append(
                f"--- Route {letter} (Route #{idx}) | Solved: {r.get('is_solved')} | Pareto: {r.get('is_pareto_optimal')} | "
                f"Composite Score: {r.get('composite_score')} | ORD Empirical Confidence: {r.get('ord_empirical_confidence')} "
                f"(Geom: {r.get('ord_geometric_confidence')}, Bottleneck: Step {r.get('ord_bottleneck_step')}, Mean ORD Yield: {r.get('ord_mean_yield_pct')}%) | "
                f"State Score: {r.get('state_score')} | Steps: {r.get('step_count')} | Avg Safety: {r.get('avg_safety_score')}/5 ---"
            )
            lines.append(f"  Starting Materials ({sm.get('in_stock_count', 0)}/{sm.get('total_leaves', 0)} in stock): {precs}")
            for s in r.get("steps", []):
                ov = s.get("ord_validation") or {}
                lines.append(
                    f"  Step {s['step_index']} [{s['reaction_name']}]: {s['forward_smiles']} | "
                    f"ORD Step Conf: {ov.get('step_ord_confidence')} (Max Sim: {ov.get('max_rxn_similarity')}, "
                    f"Mean Yield: {ov.get('mean_analog_yield_pct')}%, HTE Fail <15%: {ov.get('hte_failure_rate_pct')}%) | "
                    f"Conditions: {s['typical_conditions']} | Safety: {s['safety_green_score']}/5 | "
                    f"Selectivity/ORD Warning: {s['selectivity_warning']} | "
                    f"Stereochemical Control Risk: {s.get('stereochemical_control_risk', 'None')} | "
                    f"Strategic Key Step: {bool(s.get('is_strategic_key_step'))}"
                )
                cbyd_hits = [
                    {
                        "reaction_id": hit.get("precedent_id"),
                        "similarity": hit.get("similarity"),
                        "doi": hit.get("doi"),
                        "citation": hit.get("citation"),
                        "tolerated_functional_groups": hit.get("tolerated_functional_groups"),
                        "conditions": {
                            "catalysts": hit.get("catalysts"),
                            "ligands": hit.get("ligands"),
                            "solvents": hit.get("solvents"),
                            "temperature": hit.get("temperature"),
                            "yield": hit.get("yield_raw") or hit.get("yield_pct"),
                            "step_count": hit.get("step_count"),
                            "dr": hit.get("dr"),
                            "ee": hit.get("ee"),
                        },
                        "hazard_flags": hit.get("hazard_flags"),
                    }
                    for hit in (ov.get("top_precedents") or [])
                    if hit.get("provider") == "CbyD"
                ][:3]
                lines.append(
                    "    Functional-group audit: "
                    + json.dumps(ov.get("functional_group_audit") or {}, sort_keys=True)
                    + " | CbyD precedents: "
                    + json.dumps(cbyd_hits, ensure_ascii=False, sort_keys=True)
                )
        return "\n".join(lines)

    def _cb_set_genai_model(self, model_name: str, thinking_level: str = ""):
        info = set_active_genai_settings(model_name, thinking_level)
        return IPython.display.JSON(info)


    def _cb_sync_ord_shards(self, num_shards: int = 1):
        """Streams ORD protobuf shards from Hugging Face into the DuckDB index and updates UI badge."""
        info = self.ord_validator.sync_huggingface_parquet_shards(
            max_shards=int(num_shards), route_steps=self.state.active_route_steps()
        )
        info["ram_mb"] = get_process_ram_mb()
        return IPython.display.JSON(info)


    def _cb_load_identifier(self, query_a: str, query_b: str = ""):
        qa = (query_a or "").strip()
        qb = (query_b or "").strip()
        smi_a, src_a = _resolve_chemical_identifier(qa)
        if not smi_a:
            return IPython.display.JSON({"error": src_a})
        if qb:
            smi_b, src_b = _resolve_chemical_identifier(qb)
            if not smi_b:
                return IPython.display.JSON({"error": src_b})
            rxn_smi = f"{smi_a}.{smi_b}>>"
            info = _compute_descriptors_and_ro5(smi_a)
            return IPython.display.JSON({"smiles": rxn_smi, "source": f"{src_a} + {src_b}", "descriptors": info})
        self.state.canvas_target_smiles = _extract_primary_smiles(smi_a)
        info = _compute_descriptors_and_ro5(smi_a)
        return IPython.display.JSON({"smiles": smi_a, "source": src_a, "descriptors": info})


    def _cb_sync_ketcher_canvas(self, molblock_or_smiles: str, selected_atoms_json: str = "[]", manual_smarts: str = ""):
        raw = (molblock_or_smiles or "").strip()
        if not raw:
            return IPython.display.JSON({"error": "Ketcher canvas is empty."})
        mol = None
        if "M  END" in raw or "V2000" in raw or "V3000" in raw:
            mol = Chem.MolFromMolBlock(raw, sanitize=True)
        if mol is None:
            mol = Chem.MolFromSmiles(_extract_primary_smiles(raw))
        if mol is None:
            return IPython.display.JSON({"error": "Invalid chemical structure on canvas."})
        try:
            sel_atoms = [int(i) for i in json.loads(selected_atoms_json or "[]") if 0 <= int(i) < mol.GetNumAtoms()]
        except Exception:
            sel_atoms = []
        canon_smi = Chem.MolToSmiles(mol, canonical=True)
        canon_mol = Chem.MolFromSmiles(canon_smi)
        locked_atoms = []
        if sel_atoms and canon_mol:
            match = canon_mol.GetSubstructMatch(mol)
            if match and len(match) == mol.GetNumAtoms():
                locked_atoms = sorted({match[i] for i in sel_atoms})
        locked_smarts = (manual_smarts or "").strip()
        if locked_atoms and not locked_smarts:
            try:
                locked_smarts = Chem.MolFragmentToSmarts(canon_mol, atomsToUse=locked_atoms)
            except Exception:
                locked_smarts = ""
        self.state.canvas_target_smiles = canon_smi
        self.state.canvas_lock_smarts = locked_smarts
        drawer = rdMolDraw2D.MolDraw2DSVG(380, 175)
        drawer.drawOptions().addAtomIndices = True
        drawer.DrawMolecule(canon_mol, highlightAtoms=locked_atoms, highlightAtomColors={i: (0.05, 0.72, 0.65) for i in locked_atoms})
        drawer.FinishDrawing()
        return IPython.display.JSON({
            "smiles": canon_smi,
            "locked_atoms": locked_atoms,
            "locked_smarts": locked_smarts,
            "preview_svg": drawer.GetDrawingText(),
            "descriptors": _compute_descriptors_and_ro5(canon_smi),
            "ram_mb": get_process_ram_mb(),
        })


    def _cb_run_synthesis(self, smiles_str: str, config_json: str = "{}"):
        target = _extract_primary_smiles(smiles_str)
        if not target:
            return IPython.display.JSON({"error": "Load or draw a target molecule first."})
        try:
            ui_cfg = json.loads(config_json or "{}")
        except Exception:
            ui_cfg = {}
        if ui_cfg.get("genai_model") or ui_cfg.get("genai_thinking"):
            set_active_genai_settings(ui_cfg.get("genai_model"), ui_cfg.get("genai_thinking"))
        mode_map = {
            "cro": "process_scaleup",
            "discovery": "medchem_fast",
            "evidence": "balanced",
            "green": "green_safe",
            "strategy": "strategy_complexity",
        }
        rk_mode = mode_map.get(ui_cfg.get("rank_mode", "cro"), "balanced")
        excl_raw = ui_cfg.get("exclude_reactions", "")
        excl_list = [x.strip() for x in excl_raw.split(",") if x.strip()] if isinstance(excl_raw, str) else list(excl_raw)
        params = self.state.retro_search_params
        params.update({
            "search_algorithm": ui_cfg.get("algorithm", "retrostar"),
            "ranking_mode": rk_mode,
            "max_transforms": int(ui_cfg.get("max_steps", 6)),
            "time_limit": int(ui_cfg.get("time_limit", 120)),
            "iteration_limit": int(ui_cfg.get("iteration_limit", 100)),
            "return_first": bool(ui_cfg.get("return_first", False)),
            "cutoff_cumulative": float(ui_cfg.get("cutoff", 0.995)),
            "max_routes": int(ui_cfg.get("max_routes", 10)),
            "target_scale_grams": float(ui_cfg.get("scale_grams", 1.0)),
            "exclude_reaction_names": excl_list,
            "use_gemini": bool(ui_cfg.get("use_gemini", True)),
            "use_common_lab_reagents": bool(ui_cfg.get("use_common_lab_reagents", True)),
        })
        res = run_retrosynthesis_pipeline(
            target_smiles=target,
            ranking_mode=rk_mode,
            max_transforms=params["max_transforms"],
            time_limit=params["time_limit"],
            iteration_limit=params["iteration_limit"],
            return_first=params["return_first"],
            cutoff_cumulative=params["cutoff_cumulative"],
            max_routes=params["max_routes"],
            exclude_reaction_names=excl_list,
            lock_substructure_smarts=ui_cfg.get("lock_smarts", ""),
            target_scale_grams=params["target_scale_grams"],
            search_algorithm=params["search_algorithm"],
            use_gemini=params["use_gemini"],
            state=self.state,
            ord_validator=self.ord_validator,
        )
        routes = res.get("routes", [])
        if not routes:
            return IPython.display.JSON({"error": "No routes satisfied the current constraints. Try increasing Max Steps / Iteration Limit or relaxing exclusions."})
        header_html = _render_routes_html_from_results(res, self.ord_validator)
        deduplicate_retro_state(self.state.latest_retro_results)
        ram_mb = cleanup_aizynth_memory(self.finder, trim_ipython_history=True, state=self.state)
        return IPython.display.JSON({
            "html": header_html,
            "route_count": len(routes),
            "target_smiles": res.get("target_smiles", target),
            "ord_index_size": self.ord_validator.get_index_size(),
            "ord_prefetch": res.get("ord_prefetch_stats", {}),
            "ram_mb": ram_mb,
        })


    def _cb_reclassify_routes(self, model_name: str = "", thinking_level: str = "", rank_mode_ui: str = ""):
        """Refresh ORD evidence and Gemini output without rerunning AiZynthFinder."""
        if model_name or thinking_level:
            set_active_genai_settings(model_name, thinking_level)
        res = self.state.latest_retro_results
        routes = res.get("routes", [])
        if not routes:
            return IPython.display.JSON({
                "error": "No existing synthesis routes in memory. Click '🚀 Plan Synthesis Route' first to generate routes."
            })

        mode_map = {
            "cro": "process_scaleup",
            "discovery": "medchem_fast",
            "evidence": "balanced",
            "green": "green_safe",
            "strategy": "strategy_complexity",
        }
        rk_mode = mode_map.get(rank_mode_ui, res.get("ranking_mode", "balanced")) if rank_mode_ui else res.get("ranking_mode", "balanced")
        self.state.retro_search_params["use_gemini"] = True
        self.state.retro_search_params["ranking_mode"] = rk_mode

        all_steps = [st for r in routes for st in r.get("steps", []) if st.get("forward_smiles")]

        ord_prefetch = self.ord_validator.batch_prefetch_online_ord(
            [st["forward_smiles"] for st in all_steps],
            max_workers=4,
            limit_per_step=20,
            wall_timeout_sec=15.0,
            route_steps=all_steps,
        )
        res["ord_prefetch_stats"] = ord_prefetch

        step_lookup = _classify_route_steps(all_steps, use_gemini=True, state=self.state, ord_validator=self.ord_validator)

        for r in routes:
            steps_data = r.get("steps", [])
            for st in steps_data:
                cls_info = dict(step_lookup[_route_step_key(st)])
                cls_info.pop("gemini_tag", None)
                st.update(cls_info)

            ord_route_eval = self.ord_validator.score_route([s["forward_smiles"] for s in steps_data],
                                                       step_reports=[s["ord_validation"] for s in steps_data])
            r["ord_empirical_confidence"] = ord_route_eval["ord_empirical_confidence"]
            r["ord_geometric_confidence"] = ord_route_eval["ord_geometric_confidence"]
            r["ord_confidence_100"] = ord_route_eval["ord_confidence_100"]
            r["ord_bottleneck_step"] = ord_route_eval["bottleneck_step_index"]
            r["ord_mean_yield_pct"] = ord_route_eval["ord_mean_yield_pct"]
            r["ord_max_hte_fail_pct"] = ord_route_eval["ord_max_hte_fail_pct"]

            avg_safe = sum(s["safety_green_score"] for s in steps_data) / max(1, len(steps_data))
            risk_cnt = sum(
                1 for s in steps_data
                if s.get("selectivity_warning", "None") != "None"
                or s.get("stereochemical_control_risk", "None") != "None"
            )
            r["avg_safety_score"] = round(avg_safe, 2)
            r["selectivity_risk_count"] = risk_cnt
            annotate_route_strategy(r)
            r["composite_score"] = compute_composite_route_score(r, rk_mode)
            r["score"] = r["composite_score"]

        routes.sort(key=lambda x: (x["is_solved"], x["composite_score"]), reverse=True)
        for i, r_i in enumerate(routes):
            dominated = False
            for j, r_j in enumerate(routes):
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

        res["ranking_mode"] = rk_mode
        res["genai_model"] = ACTIVE_GENAI_SETTINGS.get("model", "gemini-3.8-flash")
        res["genai_thinking"] = ACTIVE_GENAI_SETTINGS.get("thinking_level", "medium")
        res["ord_index_size"] = self.ord_validator.get_index_size()
        header_html = _render_routes_html_from_results(res, self.ord_validator)
        self.state.latest_retro_results = deduplicate_retro_state(res)
        ram_mb = cleanup_aizynth_memory(self.finder, trim_ipython_history=True, state=self.state)
        return IPython.display.JSON({
            "html": header_html,
            "route_count": len(routes),
            "target_smiles": res.get("target_smiles", ""),
            "ord_index_size": self.ord_validator.get_index_size(),
            "ord_prefetch": ord_prefetch,
            "ram_mb": ram_mb,
        })


    def _prefetch_reaction(self, forward_smiles: str) -> dict:
        s = (forward_smiles or "").strip()
        template = self.state.step_templates.get(s, {})
        self.ord_validator.batch_prefetch_online_ord(
            [s], max_workers=2, wall_timeout_sec=3.5,
            route_steps=[dict(template, forward_smiles=s)],
        )
        return template

    def _cb_analyze_reaction(self, rxn_smiles: str, model_name: str = "", thinking_level: str = ""):
        if model_name or thinking_level:
            set_active_genai_settings(model_name, thinking_level)
        s = (rxn_smiles or "").strip()
        if ">>" not in s or not all(p.strip() for p in s.split(">>", 1)):
            return IPython.display.JSON({"error": "Draw a full reaction (Reactants >> Product) in Ketcher or paste a reaction SMILES (`A.B>>C`) first."})
        self._prefetch_reaction(s)
        cls = batch_classify_with_gemini([s], state=self.state, ord_validator=self.ord_validator)[s]
        ord_panel = _render_ord_step_panel_html(cls.get("ord_validation"), cls.get("verified_literature") or [])
        svg = mol_or_rxn_to_svg_html(s, width=480, height=130, label=cls.get("reaction_name", "Reaction"))
        used_m = cls.get("step_genai_model") or "RDKit"
        stereo_risk = cls.get("stereochemical_control_risk", "None")
        stereo_html = (
            f"<div style='margin-top:6px;padding:6px 8px;border:1px solid #f59e0b;"
            f"background:#fffbeb;border-radius:5px;color:#92400e;'>"
            f"<b>⚠ {html.escape(stereo_risk)}</b></div>"
            if stereo_risk != "None" else ""
        )
        return IPython.display.JSON({"html": f"""
    <div class="cf-route-card">
      <div style="font-size:10px;font-weight:700;color:#0284c7;margin-bottom:4px;">REACTION ANALYSIS + ORD/CBYD VALIDATION · {html.escape(used_m.upper())}</div>
      <div class="cf-step-box">{svg}</div>
      {ord_panel}
      <div style="margin-top:8px;font-size:12px;">
        <b>Conditions:</b> {html.escape(cls.get('typical_conditions', ''))} · <b>Safety:</b> {cls.get('safety_green_score', 4)}/5 · <b>Green Swap:</b> {html.escape(cls.get('greener_solvent_swap', 'None'))}
      </div>
      {stereo_html}
      {_render_ranked_conditions_html(cls)}
      {_render_step_procedure_html(cls)}
      {_render_step_analytics_html(s)}
    </div>"""})


    def _cb_structure_qa(self, smiles_str: str, question: str, model_name: str = "", thinking_level: str = ""):
        if model_name or thinking_level:
            set_active_genai_settings(model_name, thinking_level)
        s = (smiles_str or "").strip()
        q = (question or "Compare the top retrosynthesis routes by ORD Empirical Confidence Score, highlight key HTE negative-data/selectivity risks, and suggest grounded ORD reaction conditions.").strip()
        desc = _compute_descriptors_and_ro5(s)
        routes_ctx = json.dumps(self.state.latest_retro_results.get("routes", [])[:3])
        client = get_gemini_client()
        if client is None:
            return IPython.display.JSON({"html": f"<div class='cf-route-card'>Add <code>KDD_GEMINI_API_KEY</code> in Colab Secrets to enable Gemini Q&amp;A. RDKit descriptors: <code>{html.escape(json.dumps(desc))}</code></div>"})
        try:
            resp, used_m, used_t = call_gemini_with_config(
                contents=f"Active SMILES: {s}\nDescriptors: {json.dumps(desc)}\nTop AiZynthFinder + ORD Validated Routes: {routes_ctx}\nQuestion: {q}",
                temperature=0.2,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )
            ans_text = resp.text
        except Exception as e:
            used_m, used_t = ACTIVE_GENAI_SETTINGS["model"], ACTIVE_GENAI_SETTINGS["thinking_level"]
            ans_text = f"Error querying Gemini: {e}"
        return IPython.display.JSON({
            "html": (
                f"<div class='cf-route-card'>"
                f"<div style='font-size:10px;font-weight:700;color:#0284c7;margin-bottom:6px;'>STRUCTURE Q&amp;A · MODEL: {html.escape(used_m.upper())} (THINKING: {html.escape(used_t.upper())})</div>"
                f"<div style='font-size:12px;line-height:1.5;white-space:pre-wrap;'>{html.escape(ans_text)}</div></div>"
            )
        })


    def _cb_analytical_report(self, smiles_str: str, raw_report: str = "", model_name: str = "", thinking_level: str = ""):
        if model_name or thinking_level:
            set_active_genai_settings(model_name, thinking_level)
        s = (smiles_str or "").strip()
        rxn_smi = s if (">>" in s and all(p.strip() for p in s.split(">>", 1))) else f"{_extract_primary_smiles(s)}>>{_extract_primary_smiles(s)}"
        ions = rdkit_predict_lcms_ions(rxn_smi)
        rows = "".join(
            f"<tr><td style='padding:4px;'>{html.escape(i['role'])}</td><td><code>{html.escape(i['formula'])}</code></td>"
            f"<td>{i['exact_mw']:.4f}</td><td><b>{i['[M+H]+']:.4f}</b></td><td>{i['[M+Na]+']:.4f}</td><td>{i['[M-H]-']:.4f}</td><td>{html.escape(i['isotope_note'])}</td></tr>"
            for i in ions
        )
        used_m = ACTIVE_GENAI_SETTINGS["model"]
        used_t = ACTIVE_GENAI_SETTINGS["thinking_level"]
        out = f"""
    <div class="cf-route-card">
      <div style="font-size:11px;font-weight:700;color:#0284c7;margin-bottom:6px;">EXPECTED MONOISOTOPIC LC-MS IONS (RDKit) · AI: {html.escape(used_m.upper())} ({html.escape(used_t.upper())})</div>
      <table class="cf-eln-table">
        <thead><tr>
          <th>Role</th><th>Formula</th><th>Exact Mass</th><th>[M+H]+</th><th>[M+Na]+</th><th>[M-H]-</th><th>Isotope</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>"""
        if (raw_report or "").strip() and get_gemini_client():
            try:
                resp, used_m, used_t = call_gemini_with_config(
                    contents=f"Structure: {s}\nExpected Ions: {json.dumps(ions)}\nReport:\n{raw_report}\nAssign peaks, check isotope patterns, and flag plausible byproducts/impurities.",
                    temperature=0.2,
                )
                out += f"<div style='margin-top:8px;border-top:1px solid #e0f2fe;padding-top:8px;font-size:12px;white-space:pre-wrap;'>{html.escape(resp.text)}</div>"
            except Exception as e:
                out += f"<div style='margin-top:8px;color:#dc2626;font-size:12px;'>{html.escape(str(e))}</div>"
        return IPython.display.JSON({"html": out + "</div>"})


    def _cb_register_stock(self, raw_text: str):
        tokens = [line.split(",")[0].strip() for line in (raw_text or "").splitlines() if line.strip()]
        cnt = register_custom_lab_stock(tokens, state=self.state)
        return IPython.display.JSON({"message": f"✅ Registered {cnt} building blocks into Lab Stock."})


    def _cb_chemist_agent_status(self):
        res = self.state.latest_retro_results
        routes = res.get("routes", [])
        target = res.get("target_smiles") or self.state.canvas_target_smiles or "None"
        ram_mb = get_process_ram_mb()
        return IPython.display.JSON({
            "target_smiles": target,
            "route_count": len(routes),
            "is_solved": bool(res.get("is_solved", False)),
            "ranking_mode": res.get("ranking_mode", "balanced"),
            "model": ACTIVE_GENAI_SETTINGS.get("model", "gemini-3.8-flash"),
            "thinking_level": ACTIVE_GENAI_SETTINGS.get("thinking_level", "medium"),
            "ord_index_size": self.ord_validator.get_index_size(),
            "ram_mb": ram_mb,
        })


    def _cb_free_memory(self):
        deduplicate_retro_state(self.state.latest_retro_results)
        ram_mb = cleanup_aizynth_memory(self.finder, trim_ipython_history=True, state=self.state)
        return IPython.display.JSON({"ram_mb": ram_mb, "message": f"🧹 Freed search tree & C++ heap (RAM: {ram_mb} MB)"})


    def _cb_chemist_agent_reset(self):
        self.state.chemist_chat_history.clear()
        ram_mb = cleanup_aizynth_memory(self.finder, trim_ipython_history=True, state=self.state)
        return IPython.display.JSON({"status": "cleared", "ram_mb": ram_mb})


    def _cb_chemist_agent_chat(self, user_msg: str, active_smiles: str = "", model_name: str = "", thinking_level: str = ""):
        """Chat about routes, validate steps, and generate scaled ELN stoichiometry."""
        if model_name or thinking_level:
            set_active_genai_settings(model_name, thinking_level)
        msg = (user_msg or "").strip()
        if not msg:
            return IPython.display.JSON({"error": "Empty message."})

        client = get_gemini_client()
        if client is None:
            return IPython.display.JSON({
                "reply_html": "<div style='color:#dc2626;'>⚠️ Please add <code>KDD_GEMINI_API_KEY</code> to Colab Secrets to use the Organic Chemist AI Agent.</div>",
                "ram_mb": get_process_ram_mb(),
            })

        tool_artifacts_html = []
        updated_synthesis_html = None

        def rerun_retrosynthesis_with_constraints(
            ranking_mode: str = "balanced",
            max_steps: int = 6,
            exclude_reactions_csv: str = "",
            lock_substructure_smarts: str = "",
            target_scale_grams: float = 1.0,
            search_algorithm: str = "retrostar",
        ) -> str:
            """Re-runs the AiZynthFinder retrosynthesis search + parallel ORD validation with updated constraints."""
            nonlocal updated_synthesis_html
            target_smi = (
                self.state.latest_retro_results.get("target_smiles")
                or _extract_primary_smiles(active_smiles)
                or self.state.canvas_target_smiles
            )
            excl = [x.strip() for x in (exclude_reactions_csv or "").split(",") if x.strip()]
            params = self.state.retro_search_params
            res = run_retrosynthesis_pipeline(
                target_smiles=target_smi,
                ranking_mode=ranking_mode,
                max_transforms=int(max_steps),
                time_limit=int(params.get("time_limit", 120)),
                iteration_limit=int(params.get("iteration_limit", 100)),
                exclude_reaction_names=excl,
                lock_substructure_smarts=lock_substructure_smarts,
                target_scale_grams=float(target_scale_grams),
                search_algorithm=search_algorithm,
                use_gemini=True,
                state=self.state,
                ord_validator=self.ord_validator,
            )
            updated_synthesis_html = _render_routes_html_from_results(res, self.ord_validator)
            deduplicate_retro_state(self.state.latest_retro_results)
            cleanup_aizynth_memory(self.finder, trim_ipython_history=True, state=self.state)
            return self._build_retrosynthesis_context()

        def build_scaled_eln_table(route_number: int = 1, scale_grams: float = 1.0, assumed_yield: Optional[float] = None) -> str:
            """Generate scaled ELN stoichiometry using each step's yield unless a scenario yield is requested."""
            tbl_html = self.generate_eln_html_table(int(route_number), float(scale_grams), assumed_yield)
            tool_artifacts_html.append(f"<div style='margin-top:8px;padding:8px;background:#f8fafc;border:1px solid #bae6fd;border-radius:6px;'>{tbl_html}</div>")
            return self.generate_eln_markdown(int(route_number), float(scale_grams), assumed_yield)

        def inspect_reaction_step_details(forward_reaction_smiles: str) -> str:
            """Prefetches online ORD hits and predicts byproducts, exact LC-MS adduct ions, and workup ΔTPSA/ΔcLogP."""
            template = self._prefetch_reaction(forward_reaction_smiles)
            svg_card = mol_or_rxn_to_svg_html(forward_reaction_smiles, width=440, height=115, label="Inspected Reaction Step")
            ord_rep = self.ord_validator.validate_step(forward_reaction_smiles, template_info=template)
            hits = ord_rep.get("top_precedents", []) if ord_rep.get("smarts_matched_count") else []
            verified_literature = [
                work for hit in hits
                if (work := (
                    cbyd_precedent_literature(hit)
                    if hit.get("provider") == "CbyD"
                    else _verify_doi_metadata(hit.get("doi") or "")
                ))
            ]
            ord_card = _render_ord_step_panel_html(ord_rep, verified_literature)
            tool_artifacts_html.append(f"<div style='margin-top:6px;'>{svg_card}{ord_card}</div>")
            return json.dumps({
                "ord_validation": ord_rep,
                "verified_literature": verified_literature,
                "classification": rdkit_classify_step(
                    forward_reaction_smiles, template_info=template, ord_validator=self.ord_validator
                ),
                "workup": rdkit_predict_workup_properties(forward_reaction_smiles),
                "byproducts": rdkit_enumerate_byproducts(forward_reaction_smiles),
                "lcms_ions": rdkit_predict_lcms_ions(forward_reaction_smiles),
            })

        retro_ctx = self._build_retrosynthesis_context()
        history = self.state.chemist_chat_history
        history_text = "\n".join(f"{turn['role'].upper()}: {turn['text']}" for turn in history[-6:])
        system_instruction = (
            "You are an expert Medicinal & Process Organic Chemist AI Agent integrated into the ChemForge Workbench.\n"
            "You have full access to the active AiZynthFinder retrosynthesis routes, Route-Level ORD Empirical Confidence Scores (S_ORD), "
            "step-level ORD precedent matches, HTE negative-data failure rates (<15% yield), grounded ORD conditions/DOIs, "
            "and optional Chemistry by Design (CbyD) peer-reviewed precedents, functional-group tolerance, and hazard flags, "
            "starting material stock status, and RDKit tools.\n"
            "When the user asks to re-run a search with new constraints (e.g., excluding Pd/Stille/azides, fewer steps, green or evidence mode), "
            "call `rerun_retrosynthesis_with_constraints`. When asked for stoichiometry or scale-up quantities, call `build_scaled_eln_table`. "
            "When asked about ORD precedents, impurities, LC-MS ions, or workup for a specific reaction step, call `inspect_reaction_step_details`.\n"
            "For CbyD precedents, never invent stoichiometry, time, or workup: state 'Check DOI for stoichiometry/time/workup'. "
            "Surface telescoped/multi-step and BORSM yields as non-single-step evidence.\n"
            "Format your response in clean HTML using <b>, <ul>, <li>, <code>, and concise paragraphs."
        )
        prompt = f"CURRENT RETROSYNTHESIS + ORD VALIDATION SESSION CONTEXT:\n{retro_ctx}\n\nRECENT CONVERSATION HISTORY:\n{history_text}\n\nUSER QUESTION:\n{msg}"

        used_m = ACTIVE_GENAI_SETTINGS["model"]
        used_t = ACTIVE_GENAI_SETTINGS["thinking_level"]
        try:
            resp, used_m, used_t = call_gemini_with_config(
                contents=prompt,
                temperature=0.2,
                system_instruction=system_instruction,
                tools=[rerun_retrosynthesis_with_constraints, build_scaled_eln_table, inspect_reaction_step_details],
            )
            reply_text = resp.text or "Completed tool execution."
        except Exception as e:
            reply_text = f"Agent error: {e}"

        history.append({"role": "user", "text": msg})
        history.append({"role": "assistant", "text": reply_text[:1500]})
        if len(history) > 12:
            del history[:-12]

        ram_mb = cleanup_aizynth_memory(self.finder, trim_ipython_history=True, state=self.state)
        badge_html = f"<div style='font-size:10px;font-weight:700;color:#0284c7;margin-bottom:4px;'>🤖 {html.escape(used_m)} · Thinking: {html.escape(used_t.upper())}</div>"
        combined_html = f"<div style='line-height:1.5;font-size:12px;'>{badge_html}{_format_agent_reply(reply_text)}{''.join(tool_artifacts_html)}</div>"
        routes_now = self.state.latest_retro_results.get("routes", [])
        return IPython.display.JSON({
            "reply_html": combined_html,
            "updated_synthesis_html": updated_synthesis_html,
            "route_count": len(routes_now),
            "target_smiles": self.state.latest_retro_results.get("target_smiles", active_smiles),
            "ram_mb": ram_mb,
        })


def launch_workbench() -> None:
    """Initialize the workbench, register its Colab callbacks, and display it."""
    try:
        from google.colab import output
    except ImportError as exc:
        raise RuntimeError("The interactive workbench requires Google Colab.") from exc

    workbench = _Workbench(WorkbenchState(), get_finder(), get_ord_validator())
    output.register_callback("chemforge.set_model", workbench._cb_set_genai_model)
    output.register_callback("chemforge.sync_ord_shards", workbench._cb_sync_ord_shards)
    output.register_callback("chemforge.load_id", workbench._cb_load_identifier)
    output.register_callback("chemforge.sync_canvas", workbench._cb_sync_ketcher_canvas)
    output.register_callback("chemforge.run_synthesis", workbench._cb_run_synthesis)
    output.register_callback("chemforge.reclassify_routes", workbench._cb_reclassify_routes)
    output.register_callback("chemforge.analyze_reaction", workbench._cb_analyze_reaction)
    output.register_callback("chemforge.structure_qa", workbench._cb_structure_qa)
    output.register_callback("chemforge.analytical", workbench._cb_analytical_report)
    output.register_callback("chemforge.register_stock", workbench._cb_register_stock)
    output.register_callback("chemforge.agent_chat", workbench._cb_chemist_agent_chat)
    output.register_callback("chemforge.agent_reset", workbench._cb_chemist_agent_reset)
    output.register_callback("chemforge.agent_status", workbench._cb_chemist_agent_status)
    output.register_callback("chemforge.free_memory", workbench._cb_free_memory)
    _render_workbench(workbench.ord_validator)
    cleanup_aizynth_memory(workbench.finder, trim_ipython_history=True, state=workbench.state)
