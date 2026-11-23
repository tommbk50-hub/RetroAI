"""Mutable state owned by a workbench session, not by the notebook kernel."""

from dataclasses import dataclass, field


def _default_search_params() -> dict:
    return {
        "search_algorithm": "retrostar",
        "ranking_mode": "balanced",
        "max_transforms": 6,
        "time_limit": 120,
        "iteration_limit": 100,
        "return_first": False,
        "cutoff_cumulative": 0.995,
        "max_routes": 10,
        "target_scale_grams": 1.0,
        "exclude_reaction_names": [],
        "use_gemini": True,
        "validate_ord": True,
        "use_common_lab_reagents": True,
    }


@dataclass
class WorkbenchState:
    latest_retro_results: dict = field(default_factory=dict)
    gemini_step_cache: dict = field(default_factory=dict)
    custom_lab_stock_smiles: set[str] = field(default_factory=set)
    custom_lab_stock_inchikeys: set[str] = field(default_factory=set)
    retro_search_params: dict = field(default_factory=_default_search_params)
    step_templates: dict[str, dict] = field(default_factory=dict)
    chem_svg_cards: set = field(default_factory=set)
    chemist_chat_history: list[dict] = field(default_factory=list)
    canvas_target_smiles: str = ""
    canvas_lock_smarts: str = ""

    def active_route_steps(self) -> list[dict]:
        """Return steps and their template metadata for ORD shard filtering."""
        return [
            step
            for route in self.latest_retro_results.get("routes", [])
            for step in route.get("steps", [])
        ]
