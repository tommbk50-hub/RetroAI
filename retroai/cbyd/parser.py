"""Parse Chemistry by Design route JSON steps into normalized DuckDB rows."""

import json
import re
from collections.abc import Mapping

from rdkit import Chem

from retroai.core.chem_utils import _extract_primary_smiles
from retroai.ord.fingerprints import (
    compute_drfp_bytes,
    compute_folded_rxn_fp,
    compute_rdkit_internal_delta_fg,
    compute_rxn_center_diff_fp,
    compute_spectator_functional_groups,
)
from retroai.ord.parser import _standardize_primary_smiles


_HAZARDOUS_ELEMENTS = {
    "Cr": "Chromium chemistry: review toxic Cr(VI) exposure and waste; screen a lower-toxicity oxidation.",
    "Os": "Osmium chemistry: review toxicity and metal waste; screen a non-osmium alternative where feasible.",
}
_RESTRICTED_SOLVENTS = {
    "dcm": ("CH2Cl2/DCM", "Screen 2-MeTHF, EtOAc, or another process-compatible replacement."),
    "dichloromethane": ("CH2Cl2/DCM", "Screen 2-MeTHF, EtOAc, or another process-compatible replacement."),
    "ch2cl2": ("CH2Cl2/DCM", "Screen 2-MeTHF, EtOAc, or another process-compatible replacement."),
    "dmf": ("DMF", "Screen NBP, DMSO, MeCN, or a lower-hazard solvent after solubility checks."),
    "diethyl ether": ("Et2O", "Screen 2-MeTHF or MTBE, including peroxide and flammability controls."),
    "et2o": ("Et2O", "Screen 2-MeTHF or MTBE, including peroxide and flammability controls."),
    "pyridine": ("pyridine", "Screen a lower-hazard non-nucleophilic base/solvent combination."),
}


def _as_list(value) -> list:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    return list(value) if isinstance(value, (list, tuple, set)) else []


def _condition_text(value) -> str:
    return "; ".join(dict.fromkeys(str(item).strip() for item in _as_list(value) if str(item).strip()))


def parse_yield_fields(value) -> dict:
    """Parse the reported CbyD yield without erasing multi-step/BORSM semantics."""
    texts = _as_list(value)
    raw = "; ".join(str(item).strip() for item in texts if str(item).strip())
    pct_matches = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%", raw)
    step_matches = re.findall(r"(\d+)\s*steps?", raw, flags=re.I)
    yield_pct = float(pct_matches[0]) if pct_matches else None
    return {
        "yield_raw": raw,
        "yield_pct": yield_pct,
        "step_count": int(step_matches[0]) if step_matches else 1,
        "is_borsm": bool(re.search(r"\bBORSM\b", raw, flags=re.I)),
        "dr": next((m.group(0).strip() for m in re.finditer(r"\b\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\s*(?:dr|d\.?r\.?)\b", raw, flags=re.I)), ""),
        "ee": next((m.group(0).strip() for m in re.finditer(r"\b\d+(?:\.\d+)?\s*%\s*ee\b", raw, flags=re.I)), ""),
        "er": next((m.group(0).strip() for m in re.finditer(r"\b\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\s*(?:er|e\.?r\.?)\b", raw, flags=re.I)), ""),
    }


def _standardize_reaction(mapped_reaction: str) -> tuple[str, str]:
    parts = mapped_reaction.split(">")
    if len(parts) != 3:
        return "", ""
    sides = []
    for side in (parts[0], parts[-1]):
        standardized = sorted(
            dict.fromkeys(
                canonical
                for component in side.split(".")
                if (canonical := _standardize_primary_smiles(component.strip()))
            )
        )
        if not standardized:
            return "", ""
        sides.append(".".join(standardized))
    return ".".join(parts[0].split(".")), f"{sides[0]}>>{sides[1]}"


def assess_condition_hazards(catalysts=(), reagents=(), solvents=(), *, smiles=()) -> dict:
    """Return deterministic hazard flags and possible process-safety swaps."""
    names = " ".join(str(v) for group in (catalysts, reagents, solvents, smiles) for v in _as_list(group))
    normalized = " ".join(names.casefold().split())
    flags, alternatives = [], []
    compact = re.sub(r"[\s,_-]", "", normalized)
    for metal, message in _HAZARDOUS_ELEMENTS.items():
        if re.search(fr"\b{metal}\b", names) or re.search(fr"\[{metal}(?:[+0-9-]*)\]", names):
            flags.append(message)
    for token, (label, alternative) in _RESTRICTED_SOLVENTS.items():
        if token in normalized or (token in {"dcm", "dmf"} and token in compact):
            flags.append(f"Restricted/undesirable solvent: {label}.")
            alternatives.append(alternative)
    return {"hazard_flags": list(dict.fromkeys(flags)), "safer_alternatives": list(dict.fromkeys(alternatives))}


def parse_cbyd_step(
    step: Mapping,
    background: Mapping,
    *,
    route_file: str = "",
    release_version: str = "",
    min_mapping_confidence: float = 0.5,
) -> dict | None:
    """Normalize one route step, or return ``None`` when it is not indexable."""
    if any(bool(step.get(flag)) for flag in ("invalid", "warning", "orphan")):
        return None
    try:
        confidence = float(step.get("mapping_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < min_mapping_confidence:
        return None
    mapped = str(step.get("reaction_smiles_mapped") or "").strip()
    if not mapped:
        return None
    mapped_reactants, canonical = _standardize_reaction(mapped)
    if not canonical:
        return None
    raw_parts = mapped.split(">")
    normalized_mapped = f"{mapped_reactants}>>{raw_parts[-1]}"
    if not Chem.MolFromSmiles(_extract_primary_smiles(canonical)):
        return None
    fp = compute_folded_rxn_fp(canonical)
    if fp is None:
        return None
    product = Chem.MolFromSmiles(_extract_primary_smiles(canonical))
    prod_connectivity = Chem.MolToInchiKey(product).split("-")[0]
    consumed, formed, _ = compute_rdkit_internal_delta_fg(canonical)
    spectator = compute_spectator_functional_groups(canonical)
    conditions = step.get("conditions") or {}
    yields = parse_yield_fields(conditions.get("YIELD"))
    condition_fields = {
        "general_reagents": _condition_text(conditions.get("GENERAL_REAGENTS")),
        "catalysts": _condition_text(conditions.get("CATALYSTS")),
        "ligands": _condition_text(conditions.get("LIGANDS")),
        "solvents": _condition_text(conditions.get("SOLVENTS")),
        "modifiers": _condition_text(conditions.get("MODIFIERS")),
        "temperature": _condition_text(conditions.get("TEMPERATURE")),
    }
    smiles_fields = {
        "general_reagents_smiles": _condition_text(conditions.get("GENERAL_REAGENTS_SMILES")),
        "catalysts_smiles": _condition_text(conditions.get("CATALYSTS_SMILES")),
        "ligands_smiles": _condition_text(conditions.get("LIGANDS_SMILES")),
        "solvents_smiles": _condition_text(conditions.get("SOLVENTS_SMILES")),
        "modifiers_smiles": _condition_text(conditions.get("MODIFIERS_SMILES")),
    }
    hazards = assess_condition_hazards(
        condition_fields["catalysts"],
        condition_fields["general_reagents"],
        condition_fields["solvents"],
        smiles=tuple(smiles_fields.values()),
    )
    diff_fp = compute_rxn_center_diff_fp(canonical)
    return {
        "reaction_id": str(step.get("reaction_id") or f"{route_file}:{len(canonical)}"),
        "canonical_reaction": canonical,
        "rxn_smiles": normalized_mapped,
        "prod_connectivity": prod_connectivity,
        "mapping_confidence": confidence,
        "doi": str(background.get("doi") or "").strip(),
        "citation": str(background.get("citation") or "").strip(),
        "drug_name": str(background.get("drug_name") or background.get("name") or "").strip(),
        "route_file": route_file,
        "release_version": release_version,
        **condition_fields,
        **smiles_fields,
        **yields,
        "raw_description": str(conditions.get("raw_description") or ""),
        "fp_bytes": fp,
        "drfp_bytes": compute_drfp_bytes(canonical),
        "diff_fp_bytes": diff_fp.ToBinary() if diff_fp is not None else None,
        "consumed_fgs": json.dumps(sorted(consumed.elements())),
        "formed_fgs": json.dumps(sorted(formed.elements())),
        "reactant_fgs": json.dumps(spectator["reactant"]),
        "product_fgs": json.dumps(spectator["product"]),
        "survived_fgs": json.dumps(spectator["survived"]),
        "hazard_flags": json.dumps(hazards["hazard_flags"]),
        "safer_alternatives": json.dumps(hazards["safer_alternatives"]),
        "route_count": 1,
    }
