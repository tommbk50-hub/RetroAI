"""Resolve chemical names and identifiers to SMILES."""

import re
import urllib.parse

import requests
from rdkit import Chem


def _resolve_chemical_identifier(query: str):
    """Resolves Drug Name, CAS, PubChem CID, ChEMBL ID, DrugBank ID, InChIKey, or SMILES."""
    q = (query or "").strip()
    if not q:
        return None, "Empty identifier query."
    if ">>" in q:
        left, right = [p.strip() for p in q.split(">>", 1)]
        if left and right:
            return q, "Reaction SMILES"
        q = right if right else left
    if any(c in q for c in ("=", "#", "(", ")", "[", "]", "@", "/")):
        mol = Chem.MolFromSmiles(q)
        if mol is not None:
            return Chem.MolToSmiles(mol), "Direct SMILES (RDKit)"
    if q.startswith("InChI="):
        mol = Chem.MolFromInchi(q)
        if mol is not None:
            return Chem.MolToSmiles(mol), "InChI"
    if re.match(r"^[A-Za-z0-9@+\-\[\]()/\\#%=.]+$", q) and not q.isdigit():
        mol = Chem.MolFromSmiles(q)
        if mol is not None and mol.GetNumAtoms() > 1:
            return Chem.MolToSmiles(mol), "Direct SMILES"
    headers = {"User-Agent": "Colab-ChemForge-Ketcher/1.0"}
    if q.upper().startswith("CHEMBL"):
        try:
            r = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{q.upper()}.json", headers=headers, timeout=8)
            if r.status_code == 200:
                smi = (r.json().get("molecule_structures") or {}).get("canonical_smiles")
                if smi and Chem.MolFromSmiles(smi):
                    return Chem.MolToSmiles(Chem.MolFromSmiles(smi)), f"ChEMBL ({q.upper()})"
        except Exception:
            pass
    if re.match(r"^DB\d{5}$", q.upper()):
        try:
            r = requests.get(f"https://www.ebi.ac.uk/unichem/rest/src_compound_id/{q.upper()}/2/1", headers=headers, timeout=8)
            if r.status_code == 200 and r.json():
                return _resolve_chemical_identifier(r.json()[0].get("src_compound_id", ""))
        except Exception:
            pass
    cid_match = re.match(r"^(?:CID:?\s*)?(\d+)$", q, re.IGNORECASE)
    enc = urllib.parse.quote(q)
    endpoints = []
    if cid_match:
        endpoints.append((f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_match.group(1)}/property/CanonicalSMILES,IsomericSMILES/JSON", f"PubChem CID {cid_match.group(1)}"))
    if re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", q):
        endpoints.append((f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{q}/property/CanonicalSMILES,IsomericSMILES/JSON", f"PubChem InChIKey ({q})"))
    endpoints.append((f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{enc}/property/CanonicalSMILES,IsomericSMILES/JSON", f"PubChem ({q})"))
    for url, label in endpoints:
        try:
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code == 200:
                props = r.json()["PropertyTable"]["Properties"][0]
                smi = props.get("IsomericSMILES") or props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
                if smi and Chem.MolFromSmiles(smi):
                    return Chem.MolToSmiles(Chem.MolFromSmiles(smi)), label
        except Exception:
            continue
    try:
        r = requests.get(f"https://cactus.nci.nih.gov/chemical/structure/{enc}/smiles", headers=headers, timeout=8)
        if r.status_code == 200 and r.text.strip():
            smi = r.text.strip().splitlines()[0]
            if Chem.MolFromSmiles(smi):
                return Chem.MolToSmiles(Chem.MolFromSmiles(smi)), f"NCI CACTUS ({q})"
    except Exception:
        pass
    return None, f"Could not resolve '{q}'."
