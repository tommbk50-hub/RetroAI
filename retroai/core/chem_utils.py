"""RDKit descriptors, drawings, and reaction analytics."""

import html
import importlib
import importlib.util
import os
from collections import Counter
from functools import lru_cache

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


@lru_cache(maxsize=1)
def _sascore_calculator():
    """Load RDKit Contrib's SA_Score lazily, without slowing package imports."""
    try:
        module = importlib.import_module("rdkit.Contrib.SA_Score.sascorer")
    except ImportError:
        from rdkit.Chem import RDConfig

        path = os.path.join(RDConfig.RDDataDir, "..", "Contrib", "SA_Score", "sascorer.py")
        path = os.path.normpath(path)
        if not os.path.exists(path):
            return None
        spec = importlib.util.spec_from_file_location("retroai_rdkit_sascorer", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return getattr(module, "calculateScore", getattr(module, "calculate_score", None))


@lru_cache(maxsize=1)
def _spacial_score_calculator():
    """Resolve the spelling-variant SpacialScore APIs in current RDKit builds."""
    try:
        module = importlib.import_module("rdkit.Chem.SpacialScore")
    except ImportError:
        module = None
    if module is not None:
        direct = getattr(module, "CalcSpacialScore", None)
        if callable(direct):
            return lambda mol, calculator=direct: calculator(mol, normalize=False)
        for name in ("SPS", "SpacialScore", "spacial_score"):
            calculator = getattr(module, name, None)
            if callable(calculator):
                method = getattr(calculator, "CalcSpacialScore", None) or getattr(
                    calculator, "calculateScore", None
                )
                if callable(method):
                    calculator = method
                return lambda mol, calculator=calculator: calculator(mol, normalize=False)
    for name in ("CalcSPS", "CalcSpacialScore"):
        calculator = getattr(rdMolDescriptors, name, None)
        if callable(calculator):
            return lambda mol, calculator=calculator: calculator(mol, normalize=False)
    return None


def _mol_from_smiles_or_mol(smiles_or_mol):
    if isinstance(smiles_or_mol, Chem.Mol):
        return Chem.Mol(smiles_or_mol)
    return Chem.MolFromSmiles((smiles_or_mol or "").strip())


def rdkit_calculate_sascore(smiles_or_mol) -> float:
    """Return the RDKit Contrib synthetic-accessibility score (1 easy, 10 hard)."""
    mol = _mol_from_smiles_or_mol(smiles_or_mol)
    if mol is None:
        raise ValueError("Invalid molecule for SAScore calculation")
    calculator = _sascore_calculator()
    if calculator is None:
        raise RuntimeError("RDKit Contrib SA_Score is unavailable")
    return float(calculator(mol))


def rdkit_calculate_spacial_score(smiles_or_mol) -> float:
    """Return RDKit's SPS (SpacialScore) descriptor for a molecule."""
    mol = _mol_from_smiles_or_mol(smiles_or_mol)
    if mol is None:
        raise ValueError("Invalid molecule for SPS calculation")
    calculator = _spacial_score_calculator()
    if calculator is None:
        raise RuntimeError("RDKit SpacialScore is unavailable")
    return float(calculator(mol))


def _reaction_side_mols(side: str) -> list:
    return [
        mol
        for token in (side or "").split(".")
        if token.strip()
        for mol in [Chem.MolFromSmiles(token.strip())]
        if mol is not None
    ]


def rdkit_reaction_complexity(forward_smiles: str) -> dict:
    """Calculate product-relative SAScore/SPS complexity jumps for one reaction."""
    parts = (forward_smiles or "").split(">")
    reaction = parts[-3] if len(parts) >= 3 else (parts[0] if parts else "")
    products = parts[-1] if len(parts) >= 2 else ""
    reactants = _reaction_side_mols(reaction)
    product_mols = _reaction_side_mols(products)
    if not reactants or not product_mols:
        return {
            "product_sascore": None,
            "product_sps": None,
            "reactant_max_sascore": None,
            "reactant_sum_sps": None,
            "delta_sascore": None,
            "delta_sps": None,
            "complexity_error": "Reaction requires valid reactants and product",
        }
    product = max(product_mols, key=lambda mol: mol.GetNumHeavyAtoms())
    try:
        reactant_sascores = [rdkit_calculate_sascore(mol) for mol in reactants]
        reactant_sps = [rdkit_calculate_spacial_score(mol) for mol in reactants]
        product_sascore = rdkit_calculate_sascore(product)
        product_sps = rdkit_calculate_spacial_score(product)
    except (RuntimeError, ValueError) as exc:
        return {
            "product_sascore": None,
            "product_sps": None,
            "reactant_max_sascore": None,
            "reactant_sum_sps": None,
            "delta_sascore": None,
            "delta_sps": None,
            "complexity_error": str(exc),
        }
    reactant_max = max(reactant_sascores)
    reactant_sum = sum(reactant_sps)
    return {
        "product_sascore": round(product_sascore, 4),
        "product_sps": round(product_sps, 4),
        "reactant_max_sascore": round(reactant_max, 4),
        "reactant_sum_sps": round(reactant_sum, 4),
        "delta_sascore": round(product_sascore - reactant_max, 4),
        "delta_sps": round(product_sps - reactant_sum, 4),
        "complexity_error": "",
    }


def _mapped_stereo_key(atom) -> int | None:
    atom_map = atom.GetAtomMapNum()
    return atom_map or None


def _atom_stereo_orientation(mol, atom):
    if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
        return None
    neighbor_keys = []
    for neighbor in atom.GetNeighbors():
        mapped = _mapped_stereo_key(neighbor)
        neighbor_keys.append(
            ("map", mapped)
            if mapped is not None
            else ("unmapped", neighbor.GetAtomicNum(), tuple(sorted(
                (n.GetAtomicNum(), str(mol.GetBondBetweenAtoms(neighbor.GetIdx(), n.GetIdx()).GetBondType()))
                for n in neighbor.GetNeighbors()
                if n.GetIdx() != atom.GetIdx()
            )))
        )
    for _ in range(max(0, 4 - len(neighbor_keys))):
        neighbor_keys.append(("hydrogen", 1, ""))
    inversions = sum(
        left > right
        for index, left in enumerate(neighbor_keys)
        for right in neighbor_keys[index + 1:]
    )
    direction = 1 if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else -1
    return direction * (-1 if inversions % 2 else 1)


def _atom_local_signature(mol, atom):
    bonds = sorted(
        (neighbor.GetAtomicNum(), str(mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()))
        for neighbor in atom.GetNeighbors()
    )
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetTotalDegree(),
        atom.GetTotalNumHs(includeNeighbors=True),
        tuple(bonds),
    )


def _tetrahedral_stereocenters(molecules):
    mapped, fallback = {}, Counter()
    for mol in molecules:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        for atom in mol.GetAtoms():
            orientation = _atom_stereo_orientation(mol, atom)
            if orientation is None:
                continue
            atom_map = _mapped_stereo_key(atom)
            if atom_map is not None:
                mapped[atom_map] = orientation
            fallback[(_atom_local_signature(mol, atom), orientation)] += 1
    return mapped, fallback


def _bond_stereo_key(mol, bond):
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    begin_map, end_map = _mapped_stereo_key(begin), _mapped_stereo_key(end)
    if begin_map is None or end_map is None:
        return None
    try:
        chosen = tuple(_mapped_stereo_key(mol.GetAtomWithIdx(index)) for index in bond.GetStereoAtoms())
    except RuntimeError:
        chosen = ()
    if len(chosen) != 2 or any(item is None for item in chosen):
        return None
    endpoint_key = tuple(sorted((begin_map, end_map)))
    orientation = 1 if bond.GetStereo() == Chem.BondStereo.STEREOZ else -1
    for endpoint in (begin, end):
        chosen_map = next(
            (
                _mapped_stereo_key(mol.GetAtomWithIdx(index))
                for index in bond.GetStereoAtoms()
                if mol.GetBondBetweenAtoms(endpoint.GetIdx(), index) is not None
            ),
            None,
        )
        neighbor_maps = sorted(
            _mapped_stereo_key(neighbor)
            for neighbor in endpoint.GetNeighbors()
            if neighbor.GetIdx() not in {begin.GetIdx(), end.GetIdx()}
            and _mapped_stereo_key(neighbor) is not None
        )
        if chosen_map in neighbor_maps and neighbor_maps.index(chosen_map) % 2:
            orientation *= -1
    return endpoint_key, orientation


def _double_bond_stereocenters(molecules):
    centers = {}
    for mol in molecules:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() == Chem.BondStereo.STEREONONE:
                continue
            item = _bond_stereo_key(mol, bond)
            if item is not None:
                centers[item[0]] = item[1]
    return centers


def _unmapped_bond_stereo_fingerprint(mol, bond):
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    endpoint_key = tuple(sorted((
        _atom_local_signature(mol, begin),
        _atom_local_signature(mol, end),
    )))
    orientation = 1 if bond.GetStereo() == Chem.BondStereo.STEREOZ else -1
    try:
        stereo_atoms = tuple(bond.GetStereoAtoms())
    except RuntimeError:
        return None
    if len(stereo_atoms) != 2:
        return None
    for endpoint in (begin, end):
        selected = next(
            (
                _atom_local_signature(mol, mol.GetAtomWithIdx(index))
                for index in stereo_atoms
                if mol.GetBondBetweenAtoms(endpoint.GetIdx(), index) is not None
            ),
            None,
        )
        neighbor_signatures = sorted(
            _atom_local_signature(mol, neighbor)
            for neighbor in endpoint.GetNeighbors()
            if neighbor.GetIdx() not in {begin.GetIdx(), end.GetIdx()}
        )
        if selected in neighbor_signatures and neighbor_signatures.index(selected) % 2:
            orientation *= -1
    return endpoint_key, orientation


def _unmapped_double_bond_stereocenters(molecules):
    centers = Counter()
    for mol in molecules:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() == Chem.BondStereo.STEREONONE:
                continue
            item = _unmapped_bond_stereo_fingerprint(mol, bond)
            if item is not None:
                centers[item] += 1
    return centers


def rdkit_detect_stereochemical_changes(
    forward_smiles: str, mapped_reaction_smiles: str = ""
) -> dict:
    """Compare tetrahedral and double-bond configurations across a reaction.

    Atom maps are preferred because they distinguish a real inversion from a CIP
    re-prioritization. For unmapped reactions, unchanged local atom signatures
    provide a conservative fallback.
    """
    source = (mapped_reaction_smiles or forward_smiles or "").strip()
    parts = source.split(">")
    if len(parts) < 3:
        parts = (forward_smiles or "").split(">>")
        if len(parts) != 2:
            return {"events": [], "mapping_available": False, "error": "Not a reaction SMILES"}
        reactant_side, product_side = parts
    else:
        reactant_side, product_side = parts[0], parts[-1]
    reactants, products = _reaction_side_mols(reactant_side), _reaction_side_mols(product_side)
    if not reactants or not products:
        if mapped_reaction_smiles:
            return rdkit_detect_stereochemical_changes(forward_smiles)
        return {"events": [], "mapping_available": False, "error": "Invalid reaction species"}

    reactant_maps = [atom.GetAtomMapNum() for mol in reactants + products for atom in mol.GetAtoms()]
    mapping_available = any(reactant_maps) and all(
        atom.GetAtomMapNum() or atom.GetAtomicNum() == 1
        for mol in reactants + products
        for atom in mol.GetAtoms()
    )
    r_tetra, r_fallback = _tetrahedral_stereocenters(reactants)
    p_tetra, p_fallback = _tetrahedral_stereocenters(products)
    r_bonds, p_bonds = _double_bond_stereocenters(reactants), _double_bond_stereocenters(products)
    r_unmapped_bonds = _unmapped_double_bond_stereocenters(reactants)
    p_unmapped_bonds = _unmapped_double_bond_stereocenters(products)
    events = []
    if mapping_available:
        all_tetra = sorted(set(r_tetra) | set(p_tetra))
        for atom_map in all_tetra:
            if atom_map not in r_tetra:
                events.append({"type": "tetrahedral_creation", "atom_map": atom_map})
            elif atom_map not in p_tetra:
                events.append({"type": "tetrahedral_destruction", "atom_map": atom_map})
            elif r_tetra[atom_map] != p_tetra[atom_map]:
                events.append({"type": "tetrahedral_inversion", "atom_map": atom_map})
        for bond_key in sorted(set(r_bonds) | set(p_bonds)):
            if bond_key not in r_bonds:
                events.append({"type": "alkene_creation", "bond_maps": list(bond_key)})
            elif bond_key not in p_bonds:
                events.append({"type": "alkene_destruction", "bond_maps": list(bond_key)})
            elif r_bonds[bond_key] != p_bonds[bond_key]:
                events.append({"type": "alkene_ez_inversion", "bond_maps": list(bond_key)})
    else:
        reactant_signatures = Counter(key[0] for key in r_fallback for _ in range(r_fallback[key]))
        product_signatures = Counter(key[0] for key in p_fallback for _ in range(p_fallback[key]))
        for signature, count in product_signatures.items():
            unmatched = max(0, count - reactant_signatures[signature])
            events.extend({"type": "tetrahedral_creation", "atom_map": None}
                          for _ in range(unmatched))
        for signature, count in reactant_signatures.items():
            unmatched = max(0, count - product_signatures[signature])
            events.extend({"type": "tetrahedral_destruction", "atom_map": None}
                          for _ in range(unmatched))
        for signature in set(reactant_signatures) & set(product_signatures):
            before_ccw = r_fallback.get((signature, 1), 0)
            before_cw = r_fallback.get((signature, -1), 0)
            after_ccw = p_fallback.get((signature, 1), 0)
            after_cw = p_fallback.get((signature, -1), 0)
            inversions = min(before_ccw, after_cw) + min(before_cw, after_ccw)
            events.extend({"type": "tetrahedral_inversion", "atom_map": None}
                          for _ in range(inversions))
        reactant_bond_signatures = Counter(
            key[0] for key in r_unmapped_bonds for _ in range(r_unmapped_bonds[key])
        )
        product_bond_signatures = Counter(
            key[0] for key in p_unmapped_bonds for _ in range(p_unmapped_bonds[key])
        )
        for signature, count in product_bond_signatures.items():
            events.extend(
                {"type": "alkene_creation", "bond_maps": []}
                for _ in range(max(0, count - reactant_bond_signatures[signature]))
            )
        for signature, count in reactant_bond_signatures.items():
            events.extend(
                {"type": "alkene_destruction", "bond_maps": []}
                for _ in range(max(0, count - product_bond_signatures[signature]))
            )
        for signature in set(reactant_bond_signatures) & set(product_bond_signatures):
            before_positive = r_unmapped_bonds.get((signature, 1), 0)
            before_negative = r_unmapped_bonds.get((signature, -1), 0)
            after_positive = p_unmapped_bonds.get((signature, 1), 0)
            after_negative = p_unmapped_bonds.get((signature, -1), 0)
            inversions = min(before_positive, after_negative) + min(before_negative, after_positive)
            events.extend(
                {"type": "alkene_ez_inversion", "bond_maps": []}
                for _ in range(inversions)
            )
    return {
        "events": events,
        "mapping_available": mapping_available,
        "reactant_stereocenter_count": sum(r_fallback.values()),
        "product_stereocenter_count": sum(p_fallback.values()),
        "creates_stereochemistry": any(event["type"] in {"tetrahedral_creation", "alkene_creation"} for event in events),
        "inverts_stereochemistry": any(event["type"] in {"tetrahedral_inversion", "alkene_ez_inversion"} for event in events),
        "destroys_stereochemistry": any(event["type"] in {"tetrahedral_destruction", "alkene_destruction"} for event in events),
        "error": "",
    }


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
