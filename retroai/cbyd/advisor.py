"""CbyD-backed stereochemical risk and chiral-source advice."""

import re
from functools import lru_cache

from rdkit import Chem


CHIRAL_POOL_CATALOG = (
    ("L-alanine", "N[C@@H](C)C(=O)O", "α-Amino acid; inherit the α-amino-acid stereocenter."),
    ("L-proline", "O=C(O)[C@@H]1CCCN1", "α-Amino acid; rigid pyrrolidine scaffold."),
    ("L-serine", "N[C@@H](CO)C(=O)O", "α-Amino acid; amine and primary alcohol handles."),
    ("L-malic acid", "O=C(O)C[C@@H](O)C(=O)O", "Hydroxy acid; retained secondary alcohol stereocenter."),
    ("L-lactic acid", "C[C@H](O)C(=O)O", "Hydroxy acid; compact enantiopure C3 building block."),
    ("(R)-limonene", "CC1=CCC(CC1)[C@H](C)C=C", "Terpene; ring and side-chain chirality from the chiral pool."),
    ("(-)-menthol", "CC1CC[C@@H](C(C)C)[C@H](O)C1", "Terpene alcohol; rigid cyclohexane stereochemical source."),
    ("(R)-glycidol", "OC[C@@H]1CO1", "Chiral epoxide; electrophilic three-carbon fragment."),
)

CHIRAL_AUXILIARY_CATALOG = (
    ("Evans oxazolidinone", "O=C1N(*)[C@H](*)CO1", "Acylation/alkylation or aldol with an auxiliary-controlled enolate."),
    ("Ellman tert-butanesulfinamide", "CC(C)(C)[S@](=O)N", "Sulfinimine formation followed by diastereoselective addition and cleavage."),
    ("Myers pseudoephedrine amide", "O=C(*)N(C)[C@H](C)[C@H](O)c1ccccc1", "Pseudoephedrine amide enolate alkylation; remove the auxiliary after bond construction."),
)

_CHIRAL_CONTROL_TERMS = re.compile(
    r"\b(?:chiral|asymmetric|enantioselective|diastereoselective|enantiopure|"
    r"evans|ellman|myers|cinchona|taddol|binap|segphos|box|pybox|phox|salen|"
    r"chiralphos|duphos|josiphos|spar|proline|oxazolidinone|sulfinamide)\b",
    flags=re.I,
)


@lru_cache(maxsize=None)
def _catalog_patterns(smarts, allow_mirror=False):
    variants = {smarts}
    if allow_mirror and "@" in smarts:
        inverted = smarts.replace("@@", "CHIRALCCW")
        inverted = inverted.replace("@", "@@")
        variants.add(inverted.replace("CHIRALCCW", "@"))
    return tuple(pattern for variant in variants if (pattern := Chem.MolFromSmarts(variant)) is not None)


def _catalog_matches(smiles: str, catalog, *, allow_mirror=False) -> list[str]:
    reactant_side = smiles.split(">", 1)[0]
    reactants = []
    for token in reactant_side.split("."):
        mol = Chem.MolFromSmiles(token.strip())
        if mol is not None:
            reactants.append(mol)
    matches = []
    for name, catalog_smiles, rationale in catalog:
        if any(
            mol.HasSubstructMatch(pattern, useChirality=True)
            for pattern in _catalog_patterns(catalog_smiles, allow_mirror)
            for mol in reactants
        ):
            matches.append(f"{name}: {rationale}")
    return matches


def _has_chiral_control(forward_smiles: str, conditions: str) -> bool:
    if _CHIRAL_CONTROL_TERMS.search(conditions or ""):
        return True
    return bool(
        _catalog_matches(
            forward_smiles, CHIRAL_AUXILIARY_CATALOG, allow_mirror=True
        )
        or _catalog_matches(forward_smiles, CHIRAL_POOL_CATALOG)
    )


def advise_stereochemical_control(
    forward_smiles: str,
    stereochemical_changes: dict,
    *,
    conditions: str = "",
    validator=None,
    mapped_reaction_smiles: str = "",
    top_k: int = 5,
) -> dict:
    """Flag uncontrolled stereochemical events and find actionable CbyD alternatives."""
    risky_events = [
        event
        for event in stereochemical_changes.get("events", [])
        if event.get("type") in {
            "tetrahedral_creation",
            "tetrahedral_inversion",
            "alkene_creation",
            "alkene_ez_inversion",
        }
    ]
    controlled = _has_chiral_control(forward_smiles, conditions)
    risk = bool(risky_events and not controlled)
    precedents = []
    if risk and validator is not None and hasattr(validator, "find_stereoselective_cbyd_precedents"):
        precedents = validator.find_stereoselective_cbyd_precedents(
            forward_smiles,
            top_k=top_k,
            mapped_reaction_smiles=mapped_reaction_smiles,
        )
    pool_matches = _catalog_matches(forward_smiles, CHIRAL_POOL_CATALOG)
    auxiliary_matches = _catalog_matches(
        forward_smiles, CHIRAL_AUXILIARY_CATALOG, allow_mirror=True
    )
    event_types = sorted({event["type"].replace("_", " ") for event in risky_events})
    return {
        "stereochemical_control_risk": (
            "Stereochemical Control Risk: the step creates or inverts stereochemistry without an "
            f"identified chiral catalyst, ligand, pool source, or auxiliary ({', '.join(event_types)})."
            if risk
            else "None"
        ),
        "stereochemical_events": risky_events,
        "stereochemical_precedents": precedents,
        "chiral_control_detected": controlled,
        "chiral_pool_suggestions": pool_matches or [
            f"{name}: {rationale}" for name, _, rationale in CHIRAL_POOL_CATALOG
        ],
        "chiral_auxiliary_suggestions": auxiliary_matches or [
            f"{name}: {rationale}" for name, _, rationale in CHIRAL_AUXILIARY_CATALOG
        ],
    }
