"""Structured Gemini outputs and shared normalization helpers."""

import math
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _clamp_safety_score(score):
    try:
        return max(1, min(5, int(score)))
    except (ValueError, TypeError, OverflowError):
        return 3


def _clamp_step_yield(value, default=0.80):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.35, min(0.95, value)) if math.isfinite(value) else default


class RankedConditionSet(BaseModel):
    rank: int
    condition_label: str
    catalyst_and_ligand: str
    base_or_reagent: str
    solvent_and_molarity: str
    temp_and_time: str
    expected_yield_range: str
    tradeoff_note: str
    citation_or_ord_id: str


class ReagentStoichItem(BaseModel):
    name: str
    role: Literal["Reactant", "Catalyst", "Ligand", "Base/Reagent", "Solvent"]
    smiles_or_formula: str
    mw_g_mol: float = Field(ge=0, allow_inf_nan=False)
    equivalents: float = Field(ge=0, allow_inf_nan=False)
    molarity_m: float = Field(ge=0, allow_inf_nan=False)


class StepClassificationItem(BaseModel):
    forward_smiles: str
    reaction_name: str
    ranked_conditions: List[RankedConditionSet] = Field(min_length=3, max_length=3)
    reagent_stoichiometry: List[ReagentStoichItem]
    safety_green_score: int
    greener_solvent_swap: str
    selectivity_warning: str
    experimental_procedure: str
    source_doi_or_patent: str = ""
    typical_conditions: str = ""
    assumed_step_yield: float = 0.80

    @field_validator("safety_green_score", mode="before")
    @classmethod
    def clamp_safety(cls, value):
        return _clamp_safety_score(value)

    @field_validator("assumed_step_yield", mode="before")
    @classmethod
    def clamp_yield(cls, value):
        return _clamp_step_yield(value)

    @model_validator(mode="after")
    def set_primary_conditions(self):
        self.ranked_conditions.sort(key=lambda c: c.rank)
        if [c.rank for c in self.ranked_conditions] != [1, 2, 3]:
            raise ValueError("Exactly one condition set is required at each rank (1, 2 and 3).")
        primary = self.ranked_conditions[0]
        self.typical_conditions = f"Cat/ligand: {primary.catalyst_and_ligand} · Base/reagents: {primary.base_or_reagent} · Solvent: {primary.solvent_and_molarity} · {primary.temp_and_time}"
        return self


class StepClassificationBatch(BaseModel):
    steps: List[StepClassificationItem]


class StrategySummary(BaseModel):
    iupac_or_label: str
    disconnection_analysis: str
    strategy: str
