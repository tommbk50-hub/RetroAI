"""Opt-in benchmark against CbyD's AiZynthFinder-format route_tree.jsonl."""

import json
import re
from pathlib import Path

import pytest
from rdkit import Chem

from retroai.core.state import WorkbenchState
from retroai.engine.pipeline import run_retrosynthesis_pipeline
from retroai.ord.validator import ORDRetrosynthesisValidator


BENCHMARK_NAMES = ("morphine", "strychnine", "englerin a")


def load_route_tree_jsonl(path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"route_tree.jsonl line {line_number} is not an object")
            rows.append(record)
    return rows


def _walk(item):
    if isinstance(item, dict):
        yield item
        for value in item.values():
            yield from _walk(value)
    elif isinstance(item, list):
        for value in item:
            yield from _walk(value)


def _record_label(record):
    direct = " ".join(
        str(record.get(key) or "")
        for key in ("name", "drug_name", "target_name", "title", "id")
    )
    nested = " ".join(
        str(node.get(key) or "")
        for node in _walk(record)
        for key in ("name", "drug_name", "target_name")
    )
    return f"{direct} {nested}".casefold()


def _target_smiles(record):
    for key in ("target_smiles", "smiles"):
        value = record.get(key)
        if value and Chem.MolFromSmiles(str(value)):
            return str(value)
    molecule_nodes = [
        node for node in _walk(record)
        if node.get("type") in {"mol", "molecule"} and node.get("smiles")
    ]
    if not molecule_nodes:
        return ""
    return next(
        (
            str(node["smiles"])
            for node in molecule_nodes
            if Chem.MolFromSmiles(str(node["smiles"])) is not None
        ),
        "",
    )


def _molecule_smiles(record):
    return list(dict.fromkeys(
        str(node["smiles"])
        for node in _walk(record)
        if node.get("type") in {"mol", "molecule"} and node.get("smiles")
    ))


def _intermediate_smiles(record):
    target = _target_smiles(record)
    return next(
        (
            str(node["smiles"])
            for node in _walk(record)
            if node.get("type") in {"mol", "molecule"}
            and node.get("smiles")
            and str(node["smiles"]) != target
            and node.get("children")
        ),
        "",
    )


def _is_convergent(record):
    return any(
        node.get("type") == "reaction"
        and sum(
            isinstance(child, dict) and child.get("type") in {"mol", "molecule"}
            for child in node.get("children") or []
        ) >= 2
        for node in _walk(record)
    )


def _selected_record(records, target_name):
    normalized_name = re.sub(r"[^a-z0-9]+", " ", target_name.casefold()).strip()
    pattern = re.compile(rf"\b{re.escape(normalized_name)}\b")
    return next(
        (
            record for record in records
            if pattern.search(re.sub(r"[^a-z0-9]+", " ", _record_label(record)))
        ),
        None,
    )


@pytest.fixture
def cbyd_records(request):
    path = request.config.getoption("--cbyd-trees")
    if not path:
        pytest.skip("Set --cbyd-trees data/cbyd/data/route_tree.jsonl to run CbyD benchmarks.")
    path = Path(path)
    if not path.is_file():
        pytest.fail(f"CbyD route-tree file does not exist: {path}")
    return load_route_tree_jsonl(path)


@pytest.mark.benchmark
@pytest.mark.parametrize("target_name", BENCHMARK_NAMES)
def test_cbyd_multi_route_recovery_stock_and_convergence(cbyd_records, tmp_path, record_property, target_name):
    record = _selected_record(cbyd_records, target_name)
    if record is None:
        pytest.skip(f"{target_name} is not present in the supplied CbyD route tree.")
    target = _target_smiles(record)
    assert target, f"No native target SMILES found for {target_name}"
    state = WorkbenchState()
    state.retro_search_params["validate_ord"] = False
    result = run_retrosynthesis_pipeline(
        target,
        use_gemini=False,
        max_routes=10,
        time_limit=45,
        iteration_limit=200,
        state=state,
        ord_validator=ORDRetrosynthesisValidator(tmp_path / f"{target_name.replace(' ', '-')}.duckdb"),
    )
    assert result["routes"], f"No route recovered for benchmark target {target_name}"
    recovered = len(result["routes"])
    stock_ratios = [
        route["starting_materials"]["in_stock_count"]
        / max(1, route["starting_materials"]["total_leaves"])
        for route in result["routes"]
    ]
    record_property(f"{target_name}_routes", recovered)
    record_property(f"{target_name}_best_stock_coverage", max(stock_ratios))
    assert all(0.0 <= ratio <= 1.0 for ratio in stock_ratios)
    if _is_convergent(record):
        assert any(route["starting_materials"]["total_leaves"] >= 3 for route in result["routes"])


@pytest.mark.benchmark
def test_cbyd_locked_intermediate_search(cbyd_records, tmp_path):
    record = next(
        (
            item for item in cbyd_records
            if _intermediate_smiles(item) and _target_smiles(item)
        ),
        None,
    )
    if record is None:
        pytest.skip("No route tree with an intermediate suitable for Lock Selected.")
    target = _target_smiles(record)
    intermediate = _intermediate_smiles(record)
    mol = Chem.MolFromSmiles(intermediate)
    assert mol is not None
    state = WorkbenchState()
    state.retro_search_params["validate_ord"] = False
    result = run_retrosynthesis_pipeline(
        target,
        use_gemini=False,
        time_limit=45,
        iteration_limit=200,
        lock_substructure_smarts=Chem.MolToSmarts(mol),
        state=state,
        ord_validator=ORDRetrosynthesisValidator(tmp_path / "locked.duckdb"),
    )
    assert result["locked_substructure_smarts"]
    assert result["routes"], "No route satisfied the ground-truth Lock Selected intermediate."
