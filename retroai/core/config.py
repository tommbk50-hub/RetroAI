"""Runtime configuration, applied before importing the retrosynthesis engine."""

import logging
import os
from typing import Optional

# Restrict thread pools before ONNX Runtime imports NumPy or initializes sessions.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["ORT_DISABLE_MEMORY_ARENA"] = "1"

import onnxruntime as ort

# Patch once so AiZynthFinder models do not pre-allocate huge arenas.
if not getattr(ort.InferenceSession, "_chemforge_arena_patched", False):
    _orig_ort_init = ort.InferenceSession.__init__

    def _low_mem_ort_init(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
        if sess_options is None:
            sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        return _orig_ort_init(self, path_or_bytes, sess_options=sess_options, providers=providers, **kwargs)

    ort.InferenceSession.__init__ = _low_mem_ort_init
    ort.InferenceSession._chemforge_arena_patched = True

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
logging.getLogger("aizynthfinder").setLevel(logging.ERROR)


def get_secret(name: str) -> Optional[str]:
    """Read a Colab secret, falling back to an environment variable."""
    try:
        from google.colab import userdata
    except ImportError:
        return os.environ.get(name)

    try:
        return userdata.get(name) or os.environ.get(name)
    except Exception:
        # Colab raises when a secret is absent or not granted to this notebook.
        return os.environ.get(name)


# Global Model Catalog & Thinking Configuration (as of 17 Nov 2026)
GEMINI_MODEL_CATALOG = {
    "gemini-3.8-flash": {
        "label": "Gemini 3.8 Flash (Default · Fast Workhorse)",
        "family": "gemini-3",
        "default_thinking": "medium",
        "supported_thinking": ["low", "medium", "high"],
        "summary": "Default high-speed Flash model for rapid retrosynthesis step classification, experimental protocol drafting, and interactive Q&A ($0.75/$3.75 per 1M tokens).",
    },
    "gemini-4-hydrogen": {
        "label": "Gemini 4 Hydrogen (Fast Frontier Agentic · Nov 2026)",
        "family": "gemini-4",
        "default_thinking": "high",
        "supported_thinking": ["low", "medium", "high", "xhigh"],
        "summary": "Latest fast frontier model (2.5x faster than 3.8 Flash); excels at stoichiometry tables, multi-tool agent workflows, and reactant-specific procedures ($1.50/$7.50 per 1M).",
    },
    "gemini-4-argon": {
        "label": "Gemini 4 Argon (Advanced Long-Horizon Reasoning)",
        "family": "gemini-4",
        "default_thinking": "medium",
        "supported_thinking": ["low", "medium", "high"],
        "summary": "Frontier long-horizon reasoning model for multi-step synthetic route planning, chemoselectivity risk analysis, and orthogonal protecting-group strategy.",
    },
    "gemini-4-carbon": {
        "label": "Gemini 4 Carbon (Maximum Frontier Scientific Reasoning)",
        "family": "gemini-4",
        "default_thinking": "medium",
        "supported_thinking": ["low", "medium", "high"],
        "summary": "Google's strongest frontier model; allocates deep reasoning tokens to test mechanistic hypotheses, stereochemical edge cases, and process scale-up audits.",
    },
    "gemini-3.6-flash": {
        "label": "Gemini 3.6 Flash (Balanced Speed & Structured Output)",
        "family": "gemini-3",
        "default_thinking": "medium",
        "supported_thinking": ["low", "medium", "high"],
        "summary": "Balanced Gemini 3 Flash model tuned for reliable JSON schema extraction, rapid functional-group classification, and low-latency chat.",
    },
    "gemini-3.5-pro": {
        "label": "Gemini 3.5 Pro (Deep Literature & Analytical Pro)",
        "family": "gemini-3",
        "default_thinking": "medium",
        "supported_thinking": ["low", "medium", "high"],
        "summary": "High-depth Pro model ideal for comparing competing synthetic topologies, predicting impurity profiles, and interpreting LC-MS/NMR reports.",
    },
    "gemini-3.5-flash-lite": {
        "label": "Gemini 3.5 Flash-Lite (Ultra-Fast / Lowest Cost)",
        "family": "gemini-3",
        "default_thinking": "low",
        "supported_thinking": ["minimal", "low", "medium", "high"],
        "summary": "Fastest, lowest-cost model designed for sub-second high-volume reaction classification and lightweight structural lookups.",
    },
    "gemini-3.1-pro-preview": {
        "label": "Gemini 3.1 Pro (1M Context & Grounded Literature Search)",
        "family": "gemini-3",
        "default_thinking": "medium",
        "supported_thinking": ["low", "medium", "high"],
        "summary": "1M-token context Pro model with expanded medium/high thinking levels and strong Google Search grounding for precedent conditions.",
    },
    "gemini-2.5-pro": {
        "label": "Gemini 2.5 Pro (Legacy Pro · Token Budget Thinking)",
        "family": "gemini-2.5",
        "default_thinking": "medium",
        "supported_thinking": ["low", "medium", "high"],
        "summary": "Legacy 2.5 Pro model using token-based thinking_budget (1k–16k tokens) for backward compatibility.",
    },
    "gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash (Legacy Flash · Token Budget Thinking)",
        "family": "gemini-2.5",
        "default_thinking": "medium",
        "supported_thinking": ["minimal", "low", "medium", "high"],
        "summary": "Legacy 2.5 Flash model using token-based thinking_budget (0–16k tokens) for fast, low-cost fallback execution.",
    },
}

ACTIVE_GENAI_SETTINGS = {
    "model": "gemini-3.8-flash",
    "thinking_level": "medium",
}


def set_active_genai_settings(model_name: Optional[str] = None, thinking_level: Optional[str] = None) -> dict:
    """Validates and updates the global model and thinking level used across all workbench tabs."""
    if model_name and model_name in GEMINI_MODEL_CATALOG:
        ACTIVE_GENAI_SETTINGS["model"] = model_name
    spec = GEMINI_MODEL_CATALOG.get(ACTIVE_GENAI_SETTINGS["model"], GEMINI_MODEL_CATALOG["gemini-3.8-flash"])
    supported = spec["supported_thinking"]
    lvl = (thinking_level or ACTIVE_GENAI_SETTINGS.get("thinking_level") or spec["default_thinking"]).lower()
    if lvl not in supported:
        if lvl == "xhigh" and "high" in supported:
            lvl = "high"
        elif lvl == "minimal" and "low" in supported:
            lvl = "low"
        else:
            lvl = spec["default_thinking"]
    ACTIVE_GENAI_SETTINGS["thinking_level"] = lvl
    return {
        "model": ACTIVE_GENAI_SETTINGS["model"],
        "thinking_level": ACTIVE_GENAI_SETTINGS["thinking_level"],
        "label": spec["label"],
        "summary": spec["summary"],
        "supported_thinking": supported,
    }


def get_process_ram_mb() -> float:
    """Reads current process Resident Set Size (VmRSS) in MB from Linux /proc/self/status."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024.0, 1)
    except Exception:
        pass
    return 0.0
