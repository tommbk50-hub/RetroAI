"""Lazy AiZynthFinder setup, template metadata, and memory management."""

import gc
import os
import sys
from functools import lru_cache
from typing import List

from rdkit import Chem

from retroai.core.config import get_process_ram_mb
from retroai.core.state import WorkbenchState
from retroai.ord.fingerprints import _parse_reaction_mols, _reverse_reaction


@lru_cache(maxsize=1)
def get_finder():
    """Load and configure AiZynthFinder on first use."""
    if not os.path.exists("data/config.yml"):
        raise FileNotFoundError("data/config.yml not found. Run the notebook's first cell to download models and stock.")

    # Force AiZynthFinder to use ONNX Runtime rather than optional TensorFlow.
    sys.modules["tensorflow"] = None
    from aizynthfinder.aizynthfinder import AiZynthFinder

    print("⏳ Initializing AiZynthFinder engine (low-arena ONNX USPTO models + ZINC stock)...")
    finder = AiZynthFinder(configfile="data/config.yml")
    finder.stock.select("zinc")
    finder.expansion_policy.select("uspto")
    finder.filter_policy.select("uspto")
    return finder


def _harvest_uspto_templates_from_finder(finder_obj, node_templates=None, *, state: WorkbenchState):
    """Keep exact template metadata as strings, without retaining route/search-tree objects."""
    def text(value):
        return value.strip() if isinstance(value, str) else ""

    try:
        policy = finder_obj.expansion_policy["uspto"]
        templates = policy.templates
        template_column = getattr(policy, "template_column", "retro_template")
    except (AttributeError, KeyError, TypeError):
        templates = None
    registry = {}
    for route in finder_obj.routes:
        for node in route["reaction_tree"].reactions():
            forward_smiles = _reverse_reaction(node.reaction_smiles())
            if not forward_smiles:
                continue
            meta = node.metadata or {}
            code = meta.get("template_code")
            retro = text(meta.get("retro_template")) or text(meta.get("template"))
            classification = text(meta.get("classification"))
            if templates is not None and code is not None and (not retro or not classification):
                try:
                    row = templates.loc[code]
                except (KeyError, TypeError):
                    try:
                        row = templates.loc[int(code)]
                    except (KeyError, TypeError, ValueError, OverflowError):
                        row = {}
                retro = retro or text(row.get(template_column))
                classification = classification or text(row.get("classification"))
            entry = {
                "retro_template": retro,
                "classification": classification,
                "template_code": str(code if code is not None else (meta.get("template_hash") or "USPTO")),
            }
            # AiZynthFinder retains atom-mapped reaction SMILES on FixedRetroReaction metadata.
            mapped = _reverse_reaction(text(meta.get("mapped_reaction_smiles")))
            if mapped and _parse_reaction_mols(mapped)[0]:
                entry["mapped_forward_smiles"] = mapped
            if node_templates is not None:
                node_templates[id(node)] = entry
            if forward_smiles not in registry or (entry["retro_template"] and not registry[forward_smiles]["retro_template"]):
                registry[forward_smiles] = entry
    state.step_templates.clear()
    state.step_templates.update(registry)
    return list(registry)


def _flatten_reaction_tree(rxn_tree, node_templates):
    """Keep the chosen templates and producer step indices without retaining the AiZynthFinder tree."""
    nodes = list(reversed(list(rxn_tree.reactions())))
    indices = {id(node): i for i, node in enumerate(nodes, start=1)}
    steps = []
    for node in nodes:
        forward = _reverse_reaction(node.reaction_smiles())
        if not forward:
            return []
        steps.append(dict(node_templates.get(id(node), {}), forward_smiles=forward, step_index=indices[id(node)],
                          precursor_steps=[indices[id(child)] for child in rxn_tree.child_reactions(node)]))
    return steps


def cleanup_aizynth_memory(
    finder_obj=None, trim_ipython_history: bool = True, *, state: WorkbenchState | None = None
) -> float:
    """
    [Improvement 1] Purges the MCTS/Retro* search tree, cached ReactionTree objects,
    and runs gc.collect() (no malloc_trim, so background ONNX/RDKit/DuckDB C++
    allocators are never trimmed mid-operation).
    """
    if finder_obj is not None:
        try:
            finder_obj.tree = None
            finder_obj.routes = []
        except Exception:
            pass
    if state is not None:
        state.chem_svg_cards.clear()
    if trim_ipython_history:
        try:
            from IPython import get_ipython

            ip = get_ipython()
            if ip is not None and hasattr(ip, "user_ns"):
                out_hist = ip.user_ns.get("Out")
                if isinstance(out_hist, dict) and len(out_hist) > 5:
                    out_hist.clear()
                for k in ("_", "__", "___"):
                    ip.user_ns[k] = None
        except Exception:
            pass
    gc.collect()
    return get_process_ram_mb()


def register_custom_lab_stock(smiles_list: List[str], *, state: WorkbenchState) -> int:
    """Validates custom SMILES via RDKit and registers them as $0 on-shelf Lab Stock."""
    added = 0
    for raw in smiles_list:
        mol = Chem.MolFromSmiles((raw or "").strip())
        if mol is not None:
            state.custom_lab_stock_smiles.add(Chem.MolToSmiles(mol))
            state.custom_lab_stock_inchikeys.add(Chem.MolToInchiKey(mol))
            added += 1
    return added
