import pytest

from retroai.core.chem_utils import (
    _compute_descriptors_and_ro5,
    _extract_primary_smiles,
    rdkit_enumerate_byproducts,
    rdkit_predict_lcms_ions,
    rdkit_predict_workup_properties,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("", ""),
        (" O.CCO ", "O"),
        ("CCBr.O >> CCO.[Na+]", "CCO"),
        (" CCO.O >> ", "CCO"),
    ],
)
def test_extract_primary_smiles(source, expected):
    assert _extract_primary_smiles(source) == expected


def test_descriptors_use_the_primary_product():
    descriptors = _compute_descriptors_and_ro5("CCO>>CC=O.O")
    assert descriptors["is_reaction"]
    assert descriptors["canonical_smiles"] == "CC=O"
    assert descriptors["formula"] == "C2H4O"
    assert descriptors["exact_mass"] == pytest.approx(44.0262)
    assert descriptors["hba"] == 1
    assert descriptors["hbd"] == 0
    assert descriptors["ro5_pass"]
    assert "error" in _compute_descriptors_and_ro5("not-a-smiles")


def test_byproducts_include_deduplicated_product_and_predicted_impurity():
    species = rdkit_enumerate_byproducts("B(O)(O)c1ccccc1>>Oc1ccccc1.Oc1ccccc1")
    assert [(row["origin"], row["smiles"]) for row in species] == [
        ("target product", "Oc1ccccc1"),
        ("protodeboronation impurity", "c1ccccc1"),
    ]
    assert species[1]["formula"] == "C6H6"
    assert rdkit_enumerate_byproducts("CCO") == []


def test_lcms_adducts_and_bromine_isotope_note():
    reactant, product = rdkit_predict_lcms_ions("CCBr>>CCO")
    assert reactant["role"] == "Reactant"
    assert "1:1 M/M+2" in reactant["isotope_note"]
    assert product["role"] == "Product"
    assert product["formula"] == "C2H6O"
    assert product["[M+H]+"] == pytest.approx(47.0491)
    assert product["[M+Na]+"] == pytest.approx(69.0311)
    assert product["[M-H]-"] == pytest.approx(45.0346)


def test_workup_properties_compare_product_to_reactants():
    workup = rdkit_predict_workup_properties("CCBr>>CCO")
    assert workup["product"]["smiles"] == "CCO"
    assert workup["delta_tpsa"] == pytest.approx(20.2)
    assert workup["delta_clogp"] < 0
    assert any("Large polarity shift" in note for note in workup["advice"])
    assert "error" in rdkit_predict_workup_properties("CCO")
