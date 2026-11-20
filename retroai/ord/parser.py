"""Parse ORD reaction messages and normalize condition/stoichiometry data."""

import functools
import json
import math
import re
import urllib.parse
from typing import Optional

import requests
from ord_schema import message_helpers as ord_message_helpers
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from retroai.core.chem_utils import _canonical_molecule_smiles, _extract_primary_smiles

from .fingerprints import (
    compute_drfp_bytes,
    compute_folded_rxn_fp,
    compute_rdkit_internal_delta_fg,
    compute_rxn_center_diff_fp,
)


def _ord_enum_name(msg, field_name: str) -> str:
    """
    Version-safe ORD enum lookup: resolves an enum field to its symbolic name via the protobuf
    descriptor (e.g. 'SMILES', 'CATALYST', 'YIELD', 'KELVIN', 'NITROGEN') and never raises.
    """
    try:
        num = int(getattr(msg, field_name, 0))
        fd = msg.DESCRIPTOR.fields_by_name.get(field_name)
        if fd is not None and fd.enum_type is not None:
            ev = fd.enum_type.values_by_number.get(num)
            if ev is not None:
                return ev.name
    except Exception:
        pass
    return "UNSPECIFIED"


def _safe_compound_smiles(comp) -> str:
    """Safely extracts canonical SMILES from an ORD Compound or ProductCompound message."""
    try:
        smi = ord_message_helpers.smiles_from_compound(comp)
        if smi:
            return smi.strip().split()[0]
    except Exception:
        pass
    for ident in getattr(comp, "identifiers", []):
        if _ord_enum_name(ident, "type") in ("SMILES", "CXSMILES") and getattr(ident, "value", ""):
            return ident.value.strip().split()[0]
    return ""


# Canonical RDKit SMILES -> English chemical name lookup for common HTE catalysts,
# Buchwald/Suzuki ligands, bases, coupling reagents and solvents found in ORD records.
_RAW_CHEM_NAME_PAIRS = [
    # Solvents
    ("C1COCCO1", "1,4-dioxane"),
    ("COCCOC", "DME (1,2-dimethoxyethane)"),
    ("C1CCOC1", "THF"),
    ("CC1CCCO1", "2-MeTHF"),
    ("CN(C)C=O", "DMF"),
    ("CC(=O)N(C)C", "DMAc"),
    ("CN1CCCC1=O", "NMP"),
    ("CS(C)=O", "DMSO"),
    ("ClCCl", "DCM"),
    ("ClCCCl", "DCE"),
    ("Cc1ccccc1", "toluene"),
    ("CC#N", "MeCN"),
    ("CCOC(C)=O", "EtOAc"),
    ("CC(C)OC(C)=O", "i-PrOAc"),
    ("CC(C)O", "i-PrOH"),
    ("CCCCO", "n-BuOH"),
    ("CC(C)(C)O", "t-BuOH"),
    ("CCC(C)(C)O", "t-AmOH"),
    ("CO", "MeOH"),
    ("CCO", "EtOH"),
    ("O", "water"),
    ("C1CCCCC1", "cyclohexane"),
    ("CCCCCCC", "heptane"),
    ("COc1ccccc1", "anisole"),
    ("COC1CCCC1", "CPME"),
    # Bases & Common Reagents
    ("O=C([O-])[O-].[Cs+].[Cs+]", "Cs2CO3"),
    ("O=C([O-])[O-].[K+].[K+]", "K2CO3"),
    ("O=C([O-])[O-].[Na+].[Na+]", "Na2CO3"),
    ("[O-]P(=O)([O-])[O-].[K+].[K+].[K+]", "K3PO4"),
    ("CC(C)(C)[O-].[Na+]", "NaOtBu"),
    ("CC(C)(C)[O-].[K+]", "KOtBu"),
    ("CCC(C)(C)[O-].[Na+]", "NaOtAm (sodium tert-pentoxide)"),
    ("CCC(C)(C)[O-].[K+]", "KOtAm"),
    ("CCN(CC)CC", "Et3N"),
    ("CCN(C(C)C)C(C)C", "DIPEA"),
    ("C1CCN2CCCCC2=C1", "DBU"),
    ("CN1CCN(C)C1=N[P+](N(C)C)(N(C)C)N(C)C.F[P-](F)(F)(F)(F)F", "HATU"),
    ("CCN=C=NCCCN(C)C", "EDC"),
    ("CCN=C=NCCCN(C)C.Cl", "EDC·HCl"),
    ("On1nnc2ccccc21", "HOBt"),
    ("CC(=O)O[BH-](OC(C)=O)OC(C)=O.[Na+]", "NaBH(OAc)3"),
    ("[OH-].[Na+]", "NaOH"),
    ("[OH-].[K+]", "KOH"),
    ("[OH-].[Li+]", "LiOH"),
    ("CC(=O)O", "AcOH"),
    ("O=C(O)C(F)(F)F", "TFA"),
    # Catalysts & Phosphine Ligands
    ("CC(=O)O.CC(=O)O.[Pd]", "Pd(OAc)2"),
    ("CC(=O)[O-].CC(=O)[O-].[Pd+2]", "Pd(OAc)2"),
    ("O=C(/C=C/c1ccccc1)/C=C/c1ccccc1.[Pd]", "Pd2(dba)3"),
    ("[Cu]I", "CuI"),
    ("[Cu].[I-]", "CuI"),
    ("CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21", "Xantphos"),
    ("c1ccc(-c2ccccc2P(C2CCCCC2)C2CCCCC2)cc1", "CyJohnPhos"),
    ("CC(C)(C)P(c1ccccc1-c2ccccc2)C(C)(C)C", "JohnPhos"),
    ("COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1", "SPhos"),
    ("CC(C)Oc1cccc(OC(C)C)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1", "RuPhos"),
    ("CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1", "XPhos"),
    ("COc1ccc(OC)c(P(C2CCCCC2)C2CCCCC2)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C", "BrettPhos"),
    ("c1ccc(P(c2ccccc2)c2ccccc2)cc1", "PPh3"),
    ("c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1", "BINAP"),
]

_CANON_SMILES_TO_NAME = {}
for _raw_s, _eng_name in _RAW_CHEM_NAME_PAIRS:
    try:
        _m = Chem.MolFromSmiles(_raw_s)
        if _m is not None:
            _CANON_SMILES_TO_NAME[Chem.MolToSmiles(_m, canonical=True)] = _eng_name
        _CANON_SMILES_TO_NAME[_raw_s] = _eng_name
    except Exception:
        _CANON_SMILES_TO_NAME[_raw_s] = _eng_name


_SOLVENT_NAMES = frozenset({
    "1,4-dioxane", "DME (1,2-dimethoxyethane)", "THF", "2-MeTHF", "DMF", "DMAc", "NMP",
    "DMSO", "DCM", "DCE", "toluene", "MeCN", "EtOAc", "i-PrOAc", "i-PrOH", "n-BuOH",
    "t-BuOH", "t-AmOH", "MeOH", "EtOH", "water", "cyclohexane", "heptane", "anisole",
    "CPME",
})
_BASE_REAGENT_NAMES = frozenset({
    "Cs2CO3", "K2CO3", "Na2CO3", "K3PO4", "NaOtBu", "KOtBu",
    "NaOtAm (sodium tert-pentoxide)", "KOtAm", "Et3N", "DIPEA", "DBU",
    "NaOH", "KOH", "LiOH",
})
_LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


def _canonical_component_smiles(smi: str) -> str:
    """Canonicalize one possibly multi-fragment component without selecting a fragment."""
    mol = Chem.MolFromSmiles((smi or "").strip())
    return _canonical_molecule_smiles(mol) if mol is not None else ""


def _standardize_primary_smiles(smi: str) -> str:
    """Canonicalize and desalt a molecule using RDKit's largest-fragment chooser."""
    mol = Chem.MolFromSmiles((smi or "").strip())
    if mol is None:
        return ""
    try:
        mol = _LARGEST_FRAGMENT_CHOOSER.choose(mol)
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError):
        return ""
    return _canonical_molecule_smiles(mol)


def _known_spectator_role(smi: str) -> Optional[str]:
    canonical = _canonical_component_smiles(smi)
    primary = _standardize_primary_smiles(smi)
    for candidate in (canonical, primary):
        name = _CANON_SMILES_TO_NAME.get(candidate)
        if name in _SOLVENT_NAMES:
            return "SOLVENT"
        if name in _BASE_REAGENT_NAMES:
            return "REAGENT"
    return None


@functools.lru_cache(maxsize=512)
def _smiles_to_english_name(smi_or_name: str, allow_network: bool = False) -> str:
    """
    Translates a raw SMILES string (for a catalyst, ligand, reagent, or solvent)
    into a readable English chemical name using the canonical lookup table or PubChem.
    Strings that are already English names (i.e. not parseable as SMILES) are returned unchanged.
    """
    s = (smi_or_name or "").strip()
    if not s or s in ("None", "Unspecified"):
        return s
    if s in _CANON_SMILES_TO_NAME:
        return _CANON_SMILES_TO_NAME[s]
    # Whitespace never occurs inside a SMILES token, so treat it as an existing English name
    if re.search(r"\s", s):
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        # Not a valid SMILES -> already a human-readable name (e.g. "Xantphos", "Pd(OAc)2", "DMF")
        return s
    canon = Chem.MolToSmiles(mol, canonical=True)
    if canon in _CANON_SMILES_TO_NAME:
        return _CANON_SMILES_TO_NAME[canon]
    # Multi-component salt/catalyst mixture (e.g. ligand + Pd(OAc)2 in one SMILES)
    if "." in canon:
        remaining = [p.strip() for p in canon.split(".") if p.strip()]
        mapped_parts = []
        # Greedily consume known multi-fragment entries (salts / Pd(OAc)2) before single fragments
        for key in sorted((k for k in _CANON_SMILES_TO_NAME if "." in k), key=lambda k: -k.count(".")):
            key_frags = key.split(".")
            while all(remaining.count(fr) >= key_frags.count(fr) for fr in set(key_frags)):
                for fr in key_frags:
                    remaining.remove(fr)
                mapped_parts.append(_CANON_SMILES_TO_NAME[key])
        mapped_parts.extend(_CANON_SMILES_TO_NAME.get(p) for p in remaining)
        if mapped_parts and all(mapped_parts):
            return " + ".join(dict.fromkeys(mapped_parts))
    if not allow_network:
        return s
    # Query PubChem Title for unknown SMILES (cached via lru_cache)
    try:
        enc = urllib.parse.quote(canon, safe="")
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{enc}/property/Title/JSON",
            timeout=1.2,
        )
        if r.status_code == 200:
            title = r.json()["PropertyTable"]["Properties"][0].get("Title")
            if title:
                return title
    except Exception:
        pass
    return s


def _humanize_condition_list(raw_semi_str: str, allow_network: bool = False) -> str:
    """Converts a semicolon-separated string of SMILES/names into English names."""
    if not isinstance(raw_semi_str, str) or not raw_semi_str or raw_semi_str in ("None", "Unspecified"):
        return raw_semi_str
    parts = [p.strip() for p in raw_semi_str.split(";") if p.strip()]
    eng_parts = list(dict.fromkeys(_smiles_to_english_name(p, allow_network=allow_network) for p in parts))
    return "; ".join(eng_parts)


def _safe_compound_name(comp, fallback_smi: str = "") -> str:
    """Read ORD names without network calls during shard indexing."""
    for wanted in ("NAME", "IUPAC_NAME"):
        for ident in getattr(comp, "identifiers", []):
            if _ord_enum_name(ident, "type") == wanted and getattr(ident, "value", ""):
                return ident.value.strip()
    mol = Chem.MolFromSmiles(fallback_smi) if fallback_smi else None
    smi = _canonical_molecule_smiles(mol) if mol is not None else fallback_smi
    return _CANON_SMILES_TO_NAME.get(smi, smi)


def _ord_quantity(component, field, factors):
    value = getattr(component, field, None)
    if value is None:
        return 0.0
    amount = float(value.value) * factors.get(_ord_enum_name(value, "units"), 0.0)
    return amount if math.isfinite(amount) and amount > 0 else 0.0


def _ord_stoich_components(components):
    """Normalize recorded amounts to the ORD limiting charge; leave unknown amounts unset (0)."""
    rows = []
    for role, comp, smi, name in components:
        mol = Chem.MolFromSmiles(smi) if smi else None
        solvent = role == "Solvent"
        mw = Descriptors.MolWt(mol) if mol is not None and not solvent else 0.0
        mmol = _ord_quantity(comp.amount, "moles", {"MOLE": 1e3, "MILLIMOLE": 1, "MICROMOLE": 1e-3, "NANOMOLE": 1e-6})
        if not mmol and mw:
            mmol = _ord_quantity(comp.amount, "mass", {"KILOGRAM": 1e6, "GRAM": 1e3, "MILLIGRAM": 1, "MICROGRAM": 1e-3}) / mw
        volume_ml = _ord_quantity(comp.amount, "volume", {"LITER": 1e3, "MILLILITER": 1, "MICROLITER": 1e-3, "NANOLITER": 1e-6}) if solvent and not comp.amount.volume_includes_solutes else 0.0
        rows.append(({
            "name": name or smi, "role": role,
            "smiles_or_formula": _canonical_molecule_smiles(mol) if mol is not None else smi,
            "mw_g_mol": mw, "equivalents": 0.0, "molarity_m": 0.0,
        }, mmol, volume_ml, comp.is_limiting))
    reactants = [i for i, (row, _, _, _) in enumerate(rows) if row["role"] == "Reactant"]
    limiting = next((i for i in reactants if rows[i][3]), None)
    if limiting is None and reactants and all(rows[i][1] > 0 for i in reactants):
        limiting = min(reactants, key=lambda i: rows[i][1])
    limiting_mmol = rows[limiting][1] if limiting is not None else 0.0
    if limiting is not None:
        rows.insert(0, rows.pop(limiting))
    if limiting_mmol:
        for row, mmol, ml, _ in rows:
            if row["role"] == "Solvent":
                row["molarity_m"] = limiting_mmol / ml if ml else 0.0
            else:
                row["equivalents"] = mmol / limiting_mmol
    return [row for row, *_ in rows if row["name"]]


def parse_ord_reaction_protobuf(rxn, admit=None) -> Optional[dict]:
    """
    Extracts experimental metadata, yields, side-products, and grounded conditions
    from a real ord_schema.proto.reaction_pb2.Reaction protobuf.
    """
    if rxn is None or not rxn.reaction_id:
        return None

    # 1. Extract reactant and product SMILES directly from ORD components (and/or 3-part reaction SMILES)
    reactant_smis = []
    condition_compounds, reactant_compounds = [], []
    catalysts, reagents, solvents = [], [], []
    for _, inp in rxn.inputs.items():
        for comp in inp.components:
            raw_smi = _safe_compound_smiles(comp)
            smi = _standardize_primary_smiles(raw_smi)
            role = _ord_enum_name(comp, "reaction_role")
            if role == "UNSPECIFIED" and (spectator_role := _known_spectator_role(raw_smi)):
                role = spectator_role
            if role in ("CATALYST", "REAGENT", "SOLVENT"):
                condition_smi = _canonical_component_smiles(raw_smi) or smi
                condition_compounds.append((role, comp, condition_smi))
            elif role in ("REACTANT", "UNSPECIFIED"):
                if smi and smi not in reactant_smis:
                    reactant_smis.append(smi)
                    reactant_compounds.append(("Reactant", comp, smi, _CANON_SMILES_TO_NAME.get(smi, smi)))

    product_smis = []
    best_yield = None
    best_outcome = None
    undesired, undesired_byproducts = [], []
    for outcome in rxn.outcomes:
        prods = list(outcome.products)
        any_marked_desired = any(getattr(p, "is_desired_product", False) for p in prods)
        yielded_products = {
            id(prod)
            for prod in prods
            if any(_ord_enum_name(meas, "type") == "YIELD" for meas in getattr(prod, "measurements", []))
        }
        for prod in prods:
            is_desired = (
                getattr(prod, "is_desired_product", False)
                or (not any_marked_desired and id(prod) in yielded_products)
                or (not any_marked_desired and not yielded_products and len(prods) == 1)
            )
            p_smi = _standardize_primary_smiles(_safe_compound_smiles(prod))
            if is_desired:
                if p_smi and p_smi not in product_smis:
                    product_smis.append(p_smi)
                for meas in getattr(prod, "measurements", []):
                    if _ord_enum_name(meas, "type") == "YIELD":
                        try:
                            if meas.HasField("percentage"):
                                val = float(meas.percentage.value)
                            elif meas.HasField("float_value"):
                                val = float(meas.float_value.value)
                            else:
                                continue
                            if 0.0 <= val <= 100.0 and (best_yield is None or val > best_yield):
                                best_yield = round(val, 1)
                                best_outcome = outcome
                        except Exception:
                            continue
            else:
                undesired.append((prod, p_smi))

    # Fallback to get_reaction_smiles if component lists were empty
    if not reactant_smis or not product_smis:
        try:
            raw_rxn_smi = ord_message_helpers.get_reaction_smiles(rxn, generate_if_missing=True)
            if raw_rxn_smi and ">" in raw_rxn_smi:
                clean_part = raw_rxn_smi.strip().split()[0]
                parts = clean_part.split(">")
                if not reactant_smis and parts[0].strip():
                    reactant_smis = list(dict.fromkeys(
                        canonical
                        for x in parts[0].split(".")
                        if (canonical := _standardize_primary_smiles(x.strip()))
                    ))
                if not product_smis and parts[-1].strip():
                    product_smis = list(dict.fromkeys(
                        canonical
                        for x in parts[-1].split(".")
                        if (canonical := _standardize_primary_smiles(x.strip()))
                    ))
        except Exception:
            pass

    if not reactant_smis or not product_smis:
        return None

    reactants_smi = ".".join(reactant_smis)
    products_smi = ".".join(product_smis)
    clean_rxn_smi = f"{reactants_smi}>>{products_smi}"
    if admit is not None and not admit(clean_rxn_smi):
        return None
    stoich_compounds = list(reactant_compounds)
    for role, comp, smi in condition_compounds:
        name = _safe_compound_name(comp, smi)
        names = {"CATALYST": catalysts, "REAGENT": reagents, "SOLVENT": solvents}[role]
        if name and name not in names:
            names.append(name)
        stoich_compounds.append(({"CATALYST": "Catalyst", "REAGENT": "Base/Reagent", "SOLVENT": "Solvent"}[role], comp, smi, name))
    for comp, smi in undesired:
        name = _safe_compound_name(comp, smi)
        if name and name not in undesired_byproducts:
            undesired_byproducts.append(name)
    prod_mol = Chem.MolFromSmiles(_extract_primary_smiles(products_smi))
    if prod_mol is None:
        return None
    prod_inchikey = Chem.MolToInchiKey(prod_mol)
    prod_connectivity = prod_inchikey.split("-")[0]

    temp_c = None
    try:
        if rxn.conditions.HasField("temperature") and rxn.conditions.temperature.HasField("setpoint"):
            sp = rxn.conditions.temperature.setpoint
            u = _ord_enum_name(sp, "units")
            if u == "CELSIUS":
                temp_c = round(float(sp.value), 1)
            elif u == "KELVIN":
                temp_c = round(float(sp.value) - 273.15, 1)
            elif u == "FAHRENHEIT":
                temp_c = round((float(sp.value) - 32.0) * 5.0 / 9.0, 1)
    except Exception:
        pass

    atmosphere = "UNSPECIFIED"
    try:
        # Gas atmosphere lives in conditions.pressure.atmosphere (AIR, NITROGEN, ARGON, ...)
        if rxn.conditions.HasField("pressure") and rxn.conditions.pressure.HasField("atmosphere"):
            atm_name = _ord_enum_name(rxn.conditions.pressure.atmosphere, "type")
            if atm_name not in ("UNSPECIFIED", "CUSTOM"):
                atmosphere = atm_name
        if atmosphere == "UNSPECIFIED" and rxn.setup.HasField("environment"):
            env_name = _ord_enum_name(rxn.setup.environment, "type")
            if env_name in ("GLOVE_BOX", "GLOVE_BAG"):
                atmosphere = "INERT"
    except Exception:
        pass

    doi, patent, doc_url = "", "", ""
    try:
        if rxn.HasField("provenance"):
            doi = (getattr(rxn.provenance, "doi", "") or "").strip()
            patent = (getattr(rxn.provenance, "patent", "") or "").strip()
            doc_url = (getattr(rxn.provenance, "publication_url", "") or "").strip()
    except Exception:
        pass

    if doi:
        ref = f"DOI: {doi}"
    elif patent:
        ref = f"Patent: {patent}"
    elif doc_url:
        ref = doc_url
    else:
        ref = f"ORD ({rxn.reaction_id})" if getattr(rxn, "reaction_id", "") else "ORD Record"

    fp_bytes = compute_folded_rxn_fp(clean_rxn_smi)
    if fp_bytes is None:
        return None
    drfp_bytes = compute_drfp_bytes(clean_rxn_smi)
    diff_fp = compute_rxn_center_diff_fp(clean_rxn_smi)
    diff_fp_bytes = diff_fp.ToBinary() if diff_fp is not None else None
    consumed, formed, _ = compute_rdkit_internal_delta_fg(clean_rxn_smi)
    consumed_fgs = sorted(consumed.elements()) if hasattr(consumed, "elements") else sorted(consumed)
    formed_fgs = sorted(formed.elements()) if hasattr(formed, "elements") else sorted(formed)
    time_outcome = best_outcome if best_outcome is not None else next(iter(rxn.outcomes), None)
    duration_h = _ord_quantity(time_outcome, "reaction_time", {"DAY": 24, "HOUR": 1, "MINUTE": 1 / 60, "SECOND": 1 / 3600})

    return {
        "ord_id": rxn.reaction_id,
        "rxn_smiles": clean_rxn_smi,
        "prod_inchikey": prod_inchikey,
        "prod_connectivity": prod_connectivity,
        "yield_pct": best_yield,
        "catalysts": "; ".join(catalysts[:4]) or "None",
        "reagents": "; ".join(reagents[:5]) or "None",
        "solvents": "; ".join(solvents[:3]) or "Unspecified",
        "temp_c": temp_c,
        "atmosphere": atmosphere,
        "undesired_byproducts": "; ".join(undesired_byproducts[:3]) or "None",
        "reference": ref,
        "fp_bytes": fp_bytes,
        "drfp_bytes": drfp_bytes,
        "diff_fp_bytes": diff_fp_bytes,
        "consumed_fgs": json.dumps(consumed_fgs),
        "formed_fgs": json.dumps(formed_fgs),
        "procedure_details": rxn.notes.procedure_details,
        "doi": doi,
        "patent_id": patent,
        "condition_stoichiometry": json.dumps(_ord_stoich_components(stoich_compounds)),
        "duration_h": duration_h or None,
    }
