"""Reaction fingerprints, functional-group deltas, and SMARTS gates."""

import functools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, FilterCatalog, FunctionalGroups, rdChemReactions, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

from retroai.core.chem_utils import _extract_primary_smiles

DrfpEncoder = None
_DRFP_IMPORT_ATTEMPTED = False


_SMARTSRX_URL = "https://raw.githubusercontent.com/MolecularAI/smartsrx/main/smartsrx.json"
_COMPETING_ELECTROPHILES = frozenset({
    "fr_anhydride", "fr_acyl_halide", "fr_isocyan", "RDKitFG_Anhydride", "RDKitFG_AcidChloride",
})
_POPCOUNT_UINT8 = np.fromiter((value.bit_count() for value in range(256)), dtype=np.uint8, count=256)
_SPECTATOR_FG_SMARTS = (
    ("silyl_ether", "[Si][O]"),
    ("boc_protected_amine", "[NX3][C](=O)O[#6]([#6])([#6])[#6]"),
    ("cbz_protected_amine", "[NX3][C](=O)O[CH2]c"),
    ("acetal_ketal", "[#6X4]([OX2])[OX2]"),
    ("epoxide", "[C;r3]1[O;r3][C;r3]1"),
    ("alkene", "[#6;!$([#6]=O);!a]=[#6;!a]"),
    ("alkyne", "[#6]#[#6]"),
    ("indole", "[nH]1ccc2ccccc21"),
    ("ester", "[CX3](=[OX1])[OX2][#6]"),
    ("ketone", "[#6][CX3](=[OX1])[#6]"),
    ("aldehyde", "[CX3H1](=[OX1])[#6]"),
    ("unprotected_oh", "[OX2H]"),
    ("unprotected_nh", "[NX3;H1,H2]"),
)


def compute_folded_rxn_fp(rxn_smiles: str, n_bits: int = 2048) -> Optional[bytes]:
    """
    Computes a 2048-bit binary RDKit Reaction Fingerprint (packed into 256 bytes)
    using positional arguments for Boost.Python compatibility.
    """
    try:
        s = (rxn_smiles or "").strip()
        if ">>" not in s:
            return None
        rxn = rdChemReactions.ReactionFromSmarts(s, useSmiles=True)
        if not rxn or rxn.GetNumReactantTemplates() == 0 or rxn.GetNumProductTemplates() == 0:
            return None
        rdChemReactions.RemoveMappingNumbersFromReactions(rxn)

        params = rdChemReactions.ReactionFingerprintParams()
        params.fpSize = n_bits
        # CRITICAL FIX: Pass `params` positionally (Boost.Python rejects keyword ReactionFpParams)
        bv = rdChemReactions.CreateStructuralFingerprintForReaction(rxn, params)

        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return np.packbits(arr).tobytes()
    except Exception:
        return None


def compute_drfp_bytes(rxn_smiles: str, n_bits: int = 2048, radius: int = 3) -> Optional[bytes]:
    """Compute a packed DRFP reaction fingerprint, or return ``None`` when unavailable."""
    global DrfpEncoder, _DRFP_IMPORT_ATTEMPTED
    if not _DRFP_IMPORT_ATTEMPTED:
        _DRFP_IMPORT_ATTEMPTED = True
        try:
            from drfp import DrfpEncoder as encoder
        except ImportError:
            encoder = None
        DrfpEncoder = encoder
    if DrfpEncoder is None or n_bits <= 0 or n_bits % 8:
        return None
    try:
        s = (rxn_smiles or "").strip()
        if ">>" not in s:
            return None
        rxn = rdChemReactions.ReactionFromSmarts(s, useSmiles=True)
        if not rxn or rxn.GetNumReactantTemplates() == 0 or rxn.GetNumProductTemplates() == 0:
            return None
        rdChemReactions.RemoveMappingNumbersFromReactions(rxn)
        clean_rxn = rdChemReactions.ReactionToSmiles(rxn)
        fingerprint = DrfpEncoder.encode(
            [clean_rxn], n_folded_length=n_bits, radius=radius, rings=True
        )[0]
        bits = np.asarray(fingerprint, dtype=np.uint8).reshape(-1)
        if bits.size != n_bits:
            return None
        return np.packbits(bits).tobytes()
    except Exception:
        return None


def tanimoto_packed_fp_batch(query_bytes: bytes, matrix_u8: np.ndarray) -> np.ndarray:
    """Vectorized Tanimoto similarities between one packed fingerprint and a byte matrix."""
    query = np.frombuffer(query_bytes or b"", dtype=np.uint8)
    matrix = np.asarray(matrix_u8, dtype=np.uint8)
    if matrix.size == 0:
        if matrix.ndim == 1:
            matrix = matrix.reshape(0, query.size)
        return np.empty(matrix.shape[0], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != query.size:
        raise ValueError("matrix_u8 must have shape (N, len(query_bytes))")
    intersections = _POPCOUNT_UINT8[np.bitwise_and(matrix, query)].sum(axis=1, dtype=np.uint32)
    unions = _POPCOUNT_UINT8[np.bitwise_or(matrix, query)].sum(axis=1, dtype=np.uint32)
    similarities = np.zeros(matrix.shape[0], dtype=np.float64)
    np.divide(intersections, unions, out=similarities, where=unions != 0)
    return similarities


def compute_rxn_center_diff_fp(rxn_smiles: str):
    """Computes an RDKit Difference Fingerprint specifically weighting reacting bond changes."""
    try:
        rxn = rdChemReactions.ReactionFromSmarts((rxn_smiles or "").strip(), useSmiles=True)
        if not rxn or rxn.GetNumReactantTemplates() == 0 or rxn.GetNumProductTemplates() == 0:
            return None
        rdChemReactions.RemoveMappingNumbersFromReactions(rxn)
        return rdChemReactions.CreateDifferenceFingerprintForReaction(rxn)
    except Exception:
        return None


def _load_smartsrx_data():
    path = Path("data/smartsrx.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))["data"]
    except (OSError, ValueError, TypeError, KeyError):
        pass
    try:
        response = requests.get(_SMARTSRX_URL, timeout=4)
        response.raise_for_status()
        rows = response.json()["data"]
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return []  # RDKit's bundled catalogs still work offline.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
    except OSError:
        pass
    return rows


@functools.lru_cache(maxsize=1)
def _build_rdkit_internal_fg_catalog():
    """Compile RDKit's fragment/hierarchy data and MolecularAI SMARTSRX as C++ matchers."""
    patterns = {}
    path = Path(RDConfig.RDDataDir) / "FragmentDescriptors.csv"  # RDKit ships this as TSV.
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3 and fields[0].startswith("fr_"):
            patterns[fields[0]] = fields[2].strip()
    nodes = list(FunctionalGroups.BuildFuncGroupHierarchy())
    while nodes:
        node = nodes.pop()
        patterns[f"RDKitFG_{node.label}"] = node.smarts
        nodes.extend(node.children)
    for row in _load_smartsrx_data() or []:
        if isinstance(row, dict) and row.get("specific_type") and isinstance(row.get("smarts"), str):
            patterns[f"SMARTSRX_{row['specific_type']}"] = row["smarts"]
    # Ensure the requested vetoes cover cyclic anhydrides and non-chloride acyl halides,
    # which are absent from some RDKit data releases (SMARTSRX Anhydride excludes rings).
    patterns.setdefault("fr_anhydride", "[CX3](=[OX1])[OX2][CX3](=[OX1])")
    patterns.setdefault("fr_acyl_halide", "[CX3](=[OX1])[F,Cl,Br,I]")
    catalog = FilterCatalog.FilterCatalog()
    for name, smarts in patterns.items():
        # A finite maxCount asks SmartsMatcher to collect all matches, not just the first.
        matcher = FilterCatalog.SmartsMatcher(name, smarts, 1, 100000)
        if matcher.IsValid():
            catalog.AddEntry(FilterCatalog.FilterCatalogEntry(name, matcher))
    return catalog


@functools.lru_cache(maxsize=4096)
def _parse_reaction_mols(rxn_smiles):
    parts = rxn_smiles.split(">")
    if len(parts) != 3 or not (parts[0].strip() and parts[-1].strip()):
        return (), ()
    reactants, products = (tuple(Chem.MolFromSmiles(smi.strip()) for smi in side.split("."))
                           for side in (parts[0], parts[-1]))
    if any(mol is None or not mol.GetNumAtoms() for mol in reactants + products):
        return (), ()
    return reactants, products


@functools.lru_cache(maxsize=8192)
def compute_rdkit_internal_delta_fg(rxn_smiles):
    """Return multiplicity-aware depleted, increased, and starting functional groups."""
    reactants, products = _parse_reaction_mols(rxn_smiles)
    if not reactants or not products:
        return (frozenset(),) * 3
    catalog = _build_rdkit_internal_fg_catalog()

    def count(mols):
        counts = Counter()
        for mol in mols:
            seen = set()
            for match in catalog.GetFilterMatches(mol):
                name = match.filterMatch.GetName()
                if name in seen:
                    continue
                seen.add(name)
                matches = match.filterMatch.GetMatches(mol)
                counts[name] += len(matches)
        return counts

    before, after = count(reactants), count(products)
    return before - after, after - before, before.copy()


@functools.lru_cache(maxsize=4096)
def compute_spectator_functional_groups(rxn_smiles):
    """Count groups and, for mapped reactions, identify exact mapped survival."""
    reactants, products = _parse_reaction_mols(rxn_smiles)
    if not reactants or not products:
        return {"reactant": {}, "product": {}, "survived": {}}
    queries = tuple(
        (name, query)
        for name, smarts in _SPECTATOR_FG_SMARTS
        if (query := Chem.MolFromSmarts(smarts)) is not None
    )

    def occurrences(mols):
        counts, mapped = Counter(), defaultdict(list)
        for mol in mols:
            for name, query in queries:
                matches = mol.GetSubstructMatches(query, uniquify=True)
                if matches:
                    counts[name] += len(matches)
                for match in matches:
                    map_set = frozenset(
                        mol.GetAtomWithIdx(idx).GetAtomMapNum()
                        for idx in match
                        if mol.GetAtomWithIdx(idx).GetAtomMapNum()
                    )
                    if map_set and len(map_set) == len(match):
                        mapped[name].append(map_set)
        return counts, mapped

    reactant, reactant_mapped = occurrences(reactants)
    product, product_mapped = occurrences(products)
    surviving_counts = Counter()
    has_mapping = any(reactant_mapped.values()) and any(product_mapped.values())
    if has_mapping:
        for name in reactant_mapped:
            remaining = Counter(product_mapped.get(name, ()))
            for atom_maps in reactant_mapped[name]:
                if remaining[atom_maps]:
                    surviving_counts[name] += 1
                    remaining[atom_maps] -= 1
    else:
        # Unmapped user reactions cannot prove atom identity; use class overlap only as
        # a lower-confidence audit signal while mapped precedents use exact atom maps.
        surviving_counts = reactant & product
    return {
        "reactant": dict(sorted(reactant.items())),
        "product": dict(sorted(product.items())),
        "survived": dict(sorted(surviving_counts.items())),
    }


@functools.lru_cache(maxsize=128)
def _compile_rxn_smarts_queries(rxn_smarts: str):
    """
    Pre-extracts RDKit reactant and product query Mol templates from a Reaction SMARTS.
    Retains a reference to `template` in the returned tuple so RDKit C++ internal
    pointers for GetReactantTemplate/GetProductTemplate are never garbage-collected.
    """
    try:
        template = AllChem.ReactionFromSmarts(rxn_smarts)
        if not template or template.GetNumReactantTemplates() == 0 or template.GetNumProductTemplates() == 0:
            return None
        r_queries = [template.GetReactantTemplate(i) for i in range(template.GetNumReactantTemplates())]
        p_queries = [template.GetProductTemplate(i) for i in range(template.GetNumProductTemplates())]
        return (template, r_queries, p_queries)
    except Exception:
        return None


def _assign_distinct_molecules(mols, queries, use_chirality: bool = True):
    if not queries or len(queries) > len(mols):
        return False
    edges = [[idx for idx, mol in enumerate(mols) if mol.HasSubstructMatch(query, useChirality=use_chirality)]
             for query in queries]
    assigned = {}

    def augment(query_idx, seen):
        for mol_idx in edges[query_idx]:
            if mol_idx in seen:
                continue
            seen.add(mol_idx)
            if mol_idx not in assigned or augment(assigned[mol_idx], seen):
                assigned[mol_idx] = query_idx
                return True
        return False

    return all(augment(idx, set()) for idx in range(len(queries)))


def matches_reaction_smarts_mols(
    rxn_reactants: List[Chem.Mol],
    rxn_products: List[Chem.Mol],
    compiled_queries,
    use_chirality: bool = True,
) -> bool:
    """Match distinct molecules, then require mapped SMARTS to witness the same center on both sides."""
    if not compiled_queries:
        return False
    _, r_queries, p_queries = compiled_queries
    if not (
        _assign_distinct_molecules(rxn_reactants, r_queries, use_chirality)
        and _assign_distinct_molecules(rxn_products, p_queries, use_chirality)
    ):
        return False
    return _same_mapped_reaction_center(
        tuple(rxn_reactants), tuple(rxn_products), r_queries, p_queries, use_chirality
    )


def _atom_state(atom):
    return (atom.GetIsAromatic(), atom.GetFormalCharge(), atom.GetTotalNumHs(), atom.IsInRing())


def _mcs_atom_correspondence(reactants, product):
    """Map reactant atoms to unique product atoms, honoring explicit atom maps when supplied."""
    product_maps = {a.GetAtomMapNum(): a.GetIdx() for a in product.GetAtoms() if a.GetAtomMapNum()}
    mapping = {}
    remaining = set(range(product.GetNumAtoms()))
    for i, mol in enumerate(reactants):
        for atom in mol.GetAtoms():
            j = product_maps.get(atom.GetAtomMapNum())
            if j in remaining and atom.GetAtomicNum() == product.GetAtomWithIdx(j).GetAtomicNum():
                mapping[i, atom.GetIdx()] = j
                remaining.remove(j)

    for i in sorted(range(len(reactants)), key=lambda idx: -reactants[idx].GetNumHeavyAtoms()):
        if not remaining:
            break
        mol = reactants[i]
        anchors = {a: b for (idx, a), b in mapping.items() if idx == i}
        available = remaining | set(anchors.values())
        subset = Chem.RWMol(product)
        for j in sorted(set(range(product.GetNumAtoms())) - available, reverse=True):
            subset.RemoveAtom(j)
        Chem.GetSymmSSSR(subset)
        indices = sorted(available)
        mcs = rdFMCS.FindMCS([mol, subset], timeout=1, bondCompare=rdFMCS.BondCompare.CompareAny)
        if mcs.canceled or not mcs.numAtoms or (mcs.numAtoms == 1 and mol.GetNumHeavyAtoms() > 1):
            continue
        query = mcs.queryMol
        best_pairs, best_score = (), None
        for r_match in mol.GetSubstructMatches(query, uniquify=False, maxMatches=32):
            for p_match in subset.GetSubstructMatches(query, uniquify=False, maxMatches=32):
                pairs = dict(zip(r_match, (indices[j] for j in p_match)))
                if any(pairs.get(a) != b for a, b in anchors.items() if a in pairs or b in pairs.values()):
                    continue
                # Resolve symmetry by favoring preserved bonds, then preserved atom environments.
                kept = 0
                for bond in mol.GetBonds():
                    a, b = pairs.get(bond.GetBeginAtomIdx()), pairs.get(bond.GetEndAtomIdx())
                    if a is not None and b is not None:
                        pb = product.GetBondBetweenAtoms(a, b)
                        kept += bool(pb and pb.GetBondType() == bond.GetBondType())
                score = (kept, sum(_atom_state(mol.GetAtomWithIdx(a)) == _atom_state(product.GetAtomWithIdx(b))
                                   for a, b in pairs.items()))
                if best_score is None or score > best_score:
                    best_pairs, best_score = pairs.items(), score
        for a, b in best_pairs:
            mapping[i, a] = b
            remaining.discard(b)
    return mapping


def _atom_changed(reactant, idx, product, dest, atom_map):
    r_atom, p_atom = reactant.GetAtomWithIdx(idx), product.GetAtomWithIdx(dest)
    before = {atom_map(n.GetIdx()): reactant.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondType()
              for n in r_atom.GetNeighbors()}
    after = {n.GetIdx(): product.GetBondBetweenAtoms(dest, n.GetIdx()).GetBondType() for n in p_atom.GetNeighbors()}
    return before != after or _atom_state(r_atom) != _atom_state(p_atom)


@functools.lru_cache(maxsize=256)
def _candidate_atom_correspondence(reactants, products):
    product = functools.reduce(Chem.CombineMols, products)
    return product, _mcs_atom_correspondence(reactants, product)


def _same_mapped_reaction_center(reactants, products, r_queries, p_queries, use_chirality: bool = True):
    def map_numbers(queries):
        return {a.GetAtomMapNum() for q in queries for a in q.GetAtoms()} - {0}

    shared = map_numbers(r_queries) & map_numbers(p_queries)
    if not shared:
        return True  # Unmapped user-supplied SMARTS specify only the two molecular patterns.
    try:
        product, mapping = _candidate_atom_correspondence(reactants, products)
    except (ValueError, RuntimeError):
        return False
    inverse = {p: r for r, p in mapping.items()}
    offsets = [0]
    for mol in products:
        offsets.append(offsets[-1] + mol.GetNumAtoms())

    def mapped_bonds(queries):
        return {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())))
                for q in queries for b in q.GetBonds()
                if {b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()} <= shared}

    r_bonds, p_bonds = mapped_bonds(r_queries), mapped_bonds(p_queries)
    choices = []
    for side, mols, queries in ((0, reactants, r_queries), (1, products, p_queries)):
        for q in queries:
            tags = [(a.GetIdx(), a.GetAtomMapNum()) for a in q.GetAtoms() if a.GetAtomMapNum() in shared]
            options = []
            for i, mol in enumerate(mols):
                for match in mol.GetSubstructMatches(
                    q, useChirality=use_chirality, uniquify=False, maxMatches=32
                ):
                    anchor = {tag: offsets[i] + match[j] if side else mapping.get((i, match[j])) for j, tag in tags}
                    if None not in anchor.values():
                        options.append(((side, i), anchor))
            choices.append(options)
    choices.sort(key=len)
    attempts = 0

    def assign(depth, used, anchors):
        nonlocal attempts
        if depth == len(choices):
            if any(product.GetBondBetweenAtoms(anchors[a], anchors[b]) for a, b in r_bonds - p_bonds):
                return False
            for a, b in p_bonds - r_bonds:
                (i, j), (k, l) = inverse[anchors[a]], inverse[anchors[b]]
                if i == k and reactants[i].GetBondBetweenAtoms(j, l):
                    return False
            return any(_atom_changed(reactants[i], j, product, dest, lambda n: mapping.get((i, n)))
                       for dest in anchors.values() for i, j in [inverse[dest]])
        for molecule, proposed in choices[depth]:
            attempts += 1
            if attempts > 4096:
                return False  # Bound symmetric embeddings; ΔFG can still supply an independent match.
            if molecule not in used and all(anchors.get(k, v) == v for k, v in proposed.items()):
                if assign(depth + 1, used | {molecule}, anchors | proposed):
                    return True
        return False

    return assign(0, set(), {})


def _reaction_center_fragment(mol, center, radius, atom_maps):
    atoms = set(center)
    for _ in range(radius):
        atoms.update(n.GetIdx() for i in tuple(atoms) for n in mol.GetAtomWithIdx(i).GetNeighbors())
    symbols = []
    for atom in mol.GetAtoms():
        aromatic, charge, hydrogens, ring = _atom_state(atom)
        tag = f":{atom_maps[atom.GetIdx()]}" if atom.GetIdx() in atom_maps else ""
        if atom.GetIdx() in center:
            constraints = f";H{hydrogens};{charge:+d};{'R' if ring else '!R'}"
        else:
            constraints = ""
        symbols.append(f"[#{atom.GetAtomicNum()};{'a' if aromatic else 'A'}{constraints}{tag}]")
    fragment = Chem.MolFragmentToSmiles(mol, sorted(atoms), atomSymbols=symbols,
                                       allBondsExplicit=False, isomericSmiles=False, canonical=False)
    # A molecule can have disconnected reacting sites; keep these as one molecular query.
    return f"({fragment})" if "." in fragment else fragment


@functools.lru_cache(maxsize=256)
def _mcs_reaction_center_smarts(forward_smiles):
    reactants, products = _parse_reaction_mols(forward_smiles)
    if not reactants or not products:
        return ()
    product = functools.reduce(Chem.CombineMols, products)
    mapping = _mcs_atom_correspondence(reactants, product)
    inverse = {p: r for r, p in mapping.items()}
    r_centers, p_center = defaultdict(set), set()
    for (i, j), p in mapping.items():
        r_atom, p_atom = reactants[i].GetAtomWithIdx(j), product.GetAtomWithIdx(p)
        if _atom_changed(reactants[i], j, product, p, lambda n: mapping.get((i, n))):
            r_centers[i].add(j)
            p_center.add(p)
            # Retain the atoms directly lost/gained so Tier 3 is not a bond-type-only query.
            r_centers[i].update(n.GetIdx() for n in r_atom.GetNeighbors() if (i, n.GetIdx()) not in mapping)
            p_center.update(n.GetIdx() for n in p_atom.GetNeighbors() if n.GetIdx() not in inverse)
    if not r_centers or not p_center:
        return ()
    p_centers, offset = [], 0
    for mol in products:
        end = offset + mol.GetNumAtoms()
        center = {p - offset for p in p_center if offset <= p < end}
        if center:
            p_centers.append((mol, center, {p - offset: p + 1 for p in inverse if offset <= p < end}))
        offset = end
    return tuple(
        ".".join(_reaction_center_fragment(reactants[i], r_centers[i], radius,
                                           {idx: p + 1 for (r, idx), p in mapping.items() if r == i}) for i in sorted(r_centers))
        + ">>" + ".".join(_reaction_center_fragment(mol, center, radius, maps) for mol, center, maps in p_centers)
        for radius in (1, 0)
    )


def _reverse_reaction(reaction):
    parts = reaction.split(">>")
    return f"{parts[1].strip()}>>{parts[0].strip()}" if len(parts) == 2 and all(p.strip() for p in parts) else ""


def build_progressive_reaction_smarts(forward_smiles: str, template_smarts: Optional[str] = None, mapped_forward_smiles: Optional[str] = None) -> List[dict]:
    """Exact USPTO template, followed by local MCS centers at radii 1 and 0."""
    s = (forward_smiles or "").strip()
    candidates = [(1, "Tier 1: Exact USPTO Template SMARTS", 0.82, _reverse_reaction(template_smarts or ""))]
    try:
        for tier, smarts in enumerate(_mcs_reaction_center_smarts(mapped_forward_smiles or s), start=2):
            radius = 3 - tier
            candidates.append((tier, f"Tier {tier}: MCS Reaction Center (radius {radius})", (0.68, 0.58)[tier - 2], smarts))
    except (ValueError, RuntimeError):
        pass  # No trustworthy MCS; the exact template and ΔFG gate remain available.
    return [{"tier": tier, "label": label, "smarts": smarts, "queries": queries, "base_sim": base}
            for tier, label, base, smarts in candidates if smarts and (queries := _compile_rxn_smarts_queries(smarts))]


def _match_ord_gate(reactants, products, delta, tiers, query_delta):
    if not reactants or not products:
        return None
    consumed, formed, _ = delta
    query_consumed, query_formed, _ = query_delta
    consumed, formed = Counter(consumed), Counter(formed)
    query_consumed, query_formed = Counter(query_consumed), Counter(query_formed)
    extra_consumed = consumed - query_consumed
    if any(extra_consumed[name] for name in _COMPETING_ELECTROPHILES):
        return None
    for tier in tiers:
        if matches_reaction_smarts_mols(reactants, products, tier["queries"], use_chirality=True):
            return tier["label"], tier["base_sim"]
        if matches_reaction_smarts_mols(reactants, products, tier["queries"], use_chirality=False):
            return f"{tier['label']} (achiral)", max(0.0, tier["base_sim"] - 0.02)

    def jaccard(a, b):
        a, b = Counter(a), Counter(b)
        keys = set(a) | set(b)
        denominator = sum(max(a[key], b[key]) for key in keys)
        return sum(min(a[key], b[key]) for key in keys) / denominator if denominator else 0.0

    if jaccard(query_consumed, consumed) >= 0.45 and jaccard(query_formed, formed) >= 0.45:
        return "RDKit ΔFG match", 0.58
    return None


def _rdkit_extract_reacting_cores(forward_smiles: str) -> dict:
    """
    Sub-millisecond RDKit reaction-center extractor for Tier-2 online ORD fallback queries.
    Selects the largest heavy-atom limiting reactant and primary product.
    """
    s = (forward_smiles or "").strip()
    react_s, prod_s = s.split(">>", 1) if ">>" in s else (s, "")
    r_smiles = [x.strip() for x in react_s.split(".") if x.strip()]
    p_smi = _extract_primary_smiles(prod_s)
    if not r_smiles or not p_smi:
        return {"core_reactant": "", "core_product": "", "core_product_scaffold": ""}

    r_mols = [(smi, Chem.MolFromSmiles(smi)) for smi in r_smiles]
    r_mols = [(smi, m) for smi, m in r_mols if m is not None]
    r_mols.sort(key=lambda x: x[1].GetNumHeavyAtoms(), reverse=True)
    lim_r_smi = r_mols[0][0] if r_mols else r_smiles[0]
    product = Chem.MolFromSmiles(p_smi)
    if product is None:
        return {"core_reactant": lim_r_smi, "core_product": p_smi, "core_product_scaffold": p_smi}

    canonical_product = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
    centers = set()
    reactants = [mol for _, mol in r_mols]
    if reactants:
        try:
            mapping = _mcs_atom_correspondence(reactants, product)
            inverse = set(mapping.values())
            for (reactant_idx, atom_idx), product_idx in mapping.items():
                if _atom_changed(
                    reactants[reactant_idx],
                    atom_idx,
                    product,
                    product_idx,
                    lambda idx: mapping.get((reactant_idx, idx)),
                ):
                    centers.add(product_idx)
                    centers.update(
                        neighbor.GetIdx()
                        for neighbor in product.GetAtomWithIdx(product_idx).GetNeighbors()
                        if neighbor.GetIdx() not in inverse
                    )
        except (ValueError, RuntimeError):
            centers = set()

    scaffold = ""
    for radius in (3, 2):
        atoms = set(centers)
        for _ in range(radius):
            atoms.update(
                neighbor.GetIdx()
                for atom_idx in tuple(atoms)
                for neighbor in product.GetAtomWithIdx(atom_idx).GetNeighbors()
            )
        if sum(product.GetAtomWithIdx(idx).GetAtomicNum() > 1 for idx in atoms) < 4:
            continue
        try:
            fragment = Chem.MolFragmentToSmiles(
                product, sorted(atoms), isomericSmiles=True, canonical=True
            )
            fragment_mol = Chem.MolFromSmiles(fragment)
            if fragment_mol is not None:
                scaffold = Chem.MolToSmiles(fragment_mol, canonical=True, isomericSmiles=True)
                if scaffold != canonical_product:
                    break
        except (ValueError, RuntimeError):
            scaffold = ""
    if not scaffold or scaffold == canonical_product and centers:
        try:
            murcko = MurckoScaffold.GetScaffoldForMol(product)
            if murcko is not None and murcko.GetNumHeavyAtoms() >= 4:
                scaffold = Chem.MolToSmiles(murcko, canonical=True, isomericSmiles=True)
        except (ValueError, RuntimeError):
            scaffold = ""
    return {
        "core_reactant": lim_r_smi,
        "core_product": canonical_product,
        "core_product_scaffold": scaffold or canonical_product,
    }
