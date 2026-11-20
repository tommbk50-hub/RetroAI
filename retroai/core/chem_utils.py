"""RDKit descriptors, drawings, and reaction analytics."""

import html

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdChemReactions, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


def _extract_primary_smiles(smiles_or_rxn: str) -> str:
    """Safely extracts the primary molecule SMILES even if a dangling '>>' arrow is present."""
    s = (smiles_or_rxn or "").strip()
    if not s:
        return ""
    if ">>" in s:
        left, right = [p.strip() for p in s.split(">>", 1)]
        target_part = right if right else left
        return target_part.split(".")[0].strip()
    return s.split(".")[0].strip()


def _canonical_molecule_smiles(mol):
    if any(a.GetAtomMapNum() for a in mol.GetAtoms()):
        mol = Chem.Mol(mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _compute_descriptors_and_ro5(smiles_or_rxn: str) -> dict:
    """Computes RDKit descriptors and Lipinski Rule-of-5 flags for the active canvas."""
    s = (smiles_or_rxn or "").strip()
    if not s:
        return {"error": "Empty input"}
    target_smi = _extract_primary_smiles(s)
    mol = Chem.MolFromSmiles(target_smi)
    if mol is None:
        return {"error": f"Invalid SMILES: {target_smi}"}
    canon = Chem.MolToSmiles(mol)
    mw = Descriptors.MolWt(mol)
    exact_mw = Descriptors.ExactMolWt(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    clogp = Descriptors.MolLogP(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    flags = []
    if mw > 500:
        flags.append("MW > 500")
    if clogp > 5:
        flags.append("cLogP > 5")
    if hbd > 5:
        flags.append("HBD > 5")
    if hba > 10:
        flags.append("HBA > 10")
    has_valid_rxn = ">>" in s and all(p.strip() for p in s.split(">>", 1))
    return {
        "is_reaction": has_valid_rxn,
        "full_smiles": s,
        "canonical_smiles": canon,
        "formula": formula,
        "mw": round(mw, 2),
        "exact_mass": round(exact_mw, 4),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 1),
        "rotatable": rot,
        "rings": rings,
        "clogp": round(clogp, 2),
        "ro5_pass": len(flags) <= 1,
        "ro5_flags": " · ".join(flags) if flags else "All criteria met",
    }


def mol_or_rxn_to_svg_html(smiles_or_rxn: str, width: int = 480, height: int = 125, highlight_smarts: str = "", label: str = "") -> str:
    """Renders a molecule or reaction SMILES into an inline 2D RDKit SVG card without storing in global RAM."""
    s = (smiles_or_rxn or "").strip()
    try:
        if ">>" in s and all(p.strip() for p in s.split(">>", 1)):
            rxn = AllChem.ReactionFromSmarts(s, useSmiles=True)
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.drawOptions().padding = 0.06
            drawer.DrawReaction(rxn)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
        else:
            mol = Chem.MolFromSmiles(_extract_primary_smiles(s))
            if mol is None:
                return f"<code>{html.escape(s)}</code>"
            hl_atoms = []
            if highlight_smarts:
                pat = Chem.MolFromSmarts(highlight_smarts)
                if pat and mol.HasSubstructMatch(pat):
                    hl_atoms = list(mol.GetSubstructMatch(pat))
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.drawOptions().padding = 0.08
            drawer.DrawMolecule(mol, highlightAtoms=hl_atoms)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
        hdr = f"<div style='font-size:11px;font-weight:700;color:#0284c7;margin-bottom:3px;'>{html.escape(label)}</div>" if label else ""
        return f"<div class='chem-card' style='background:#fff;border:1px solid #bae6fd;border-radius:6px;padding:6px;text-align:center;'>{hdr}{svg}</div>"
    except Exception:
        return f"<code>{html.escape(s)}</code>"


def rdkit_enumerate_byproducts(forward_smiles: str) -> list:
    """Enumerates target product and plausible side-products (protodeboronation, hydrodehalogenation, hydrolysis)."""
    s = (forward_smiles or "").strip()
    if ">>" not in s:
        return []
    react_s, prod_s = s.split(">>", 1)
    r_mols = [Chem.MolFromSmiles(x.strip()) for x in react_s.split(".") if x.strip()]
    p_mols = [Chem.MolFromSmiles(x.strip()) for x in prod_s.split(".") if x.strip()]
    out = []
    seen = set()

    def _add(mol, origin):
        if mol is None:
            return
        try:
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
            if smi in seen:
                return
            seen.add(smi)
            out.append({
                "origin": origin,
                "smiles": smi,
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "exact_mw": round(Descriptors.ExactMolWt(mol), 4),
            })
        except Exception:
            pass

    for pm in p_mols:
        _add(pm, "target product")
    side_transforms = [
        ("protodeboronation impurity", "[#6:1][BX3]([OX2])[OX2]>>[#6:1]"),
        ("hydrodehalogenation impurity", "[c:1][Cl,Br,I]>>[c:1]"),
        ("ester hydrolysis byproduct", "[CX3:1](=[O:2])[OX2][#6]>>[CX3:1](=[O:2])O"),
    ]
    for label, smarts in side_transforms:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        for rm in r_mols:
            if rm is None:
                continue
            for prod_tuple in rxn.RunReactants((rm,), maxProducts=2):
                _add(prod_tuple[0], label)
    return out


def rdkit_predict_lcms_ions(forward_smiles: str) -> list:
    """Predicts exact monoisotopic LC-MS adduct ions ([M+H]+, [M+Na]+, [M-H]-, isotope pattern)."""
    s = (forward_smiles or "").strip()
    react_s, prod_s = s.split(">>", 1) if ">>" in s else (s, s)
    entries = []
    seen = set()
    items = [("Reactant", x) for x in react_s.split(".") if x.strip()] + [("Product", x) for x in prod_s.split(".") if x.strip()]
    for b in rdkit_enumerate_byproducts(s):
        if b["origin"] != "target product":
            items.append((f"Byproduct ({b['origin']})", b["smiles"]))
    for role, smi_raw in items:
        mol = Chem.MolFromSmiles(smi_raw.strip())
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        if (role, smi) in seen:
            continue
        seen.add((role, smi))
        emw = Descriptors.ExactMolWt(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        cl_cnt = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "Cl")
        br_cnt = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "Br")
        iso = "Standard M+1"
        if br_cnt:
            iso = f"{br_cnt}x Br (1:1 M/M+2 doublet)"
        elif cl_cnt:
            iso = f"{cl_cnt}x Cl (3:1 M/M+2 isotope)"
        entries.append({
            "role": role,
            "smiles": smi,
            "formula": formula,
            "exact_mw": round(emw, 4),
            "[M+H]+": round(emw + 1.007276, 4),
            "[M+Na]+": round(emw + 22.989221, 4),
            "[M+NH4]+": round(emw + 18.033826, 4),
            "[M-H]-": round(emw - 1.007276, 4),
            "isotope_note": iso,
        })
    return entries


def rdkit_predict_workup_properties(forward_smiles: str) -> dict:
    """Computes ΔcLogP / ΔTPSA between reactants and product and recommends workup conditions."""
    s = (forward_smiles or "").strip()
    if ">>" not in s:
        return {"error": "Not a reaction SMILES"}
    react_s, prod_s = s.split(">>", 1)
    r_mols = [Chem.MolFromSmiles(x.strip()) for x in react_s.split(".") if x.strip()]
    p_mol = Chem.MolFromSmiles(_extract_primary_smiles(prod_s))
    r_mols = [m for m in r_mols if m is not None]
    if p_mol is None or not r_mols:
        return {"error": "Invalid reaction species"}
    p_clogp = Descriptors.MolLogP(p_mol)
    p_tpsa = Descriptors.TPSA(p_mol)
    r_clogp = sum(Descriptors.MolLogP(m) for m in r_mols) / len(r_mols)
    r_tpsa = sum(Descriptors.TPSA(m) for m in r_mols) / len(r_mols)
    d_clogp = p_clogp - r_clogp
    d_tpsa = p_tpsa - r_tpsa
    advice = []
    if p_clogp > 1.5:
        advice.append(f"Lipophilic product (cLogP {p_clogp:.2f}): partitions cleanly into EtOAc or i-PrOAc over aq. NaHCO3/brine.")
    else:
        advice.append(f"Polar product (cLogP {p_clogp:.2f}, TPSA {p_tpsa:.1f} Å²): avoid excessive aqueous washes; extract with 3:1 CHCl3/i-PrOH or 2-MeTHF.")
    if abs(d_tpsa) > 18:
        advice.append(f"Large polarity shift (ΔTPSA {d_tpsa:+.1f} Å²): high normal-phase silica TLC resolution expected.")
    else:
        advice.append(f"Small polarity shift (ΔTPSA {d_tpsa:+.1f} Å²): use gradient flash chromatography or C18 reverse-phase prep-HPLC.")
    return {
        "product": {
            "smiles": Chem.MolToSmiles(p_mol),
            "clogp": round(p_clogp, 2),
            "tpsa": round(p_tpsa, 1),
            "hbd": rdMolDescriptors.CalcNumHBD(p_mol),
            "hba": rdMolDescriptors.CalcNumHBA(p_mol),
        },
        "delta_clogp": round(d_clogp, 2),
        "delta_tpsa": round(d_tpsa, 1),
        "advice": advice,
    }
