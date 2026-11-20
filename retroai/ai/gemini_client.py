"""Lazy Gemini client and generation configuration."""

from typing import Optional

from google import genai
from google.genai import types

from retroai.core.config import ACTIVE_GENAI_SETTINGS, GEMINI_MODEL_CATALOG, get_secret

_GEMINI_CLIENT = None


def get_gemini_client():
    """Returns a cached google.genai Client using KDD_GEMINI_API_KEY."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT
    api_key = get_secret("KDD_GEMINI_API_KEY")
    if api_key:
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def _build_genai_config(
    model_name: str,
    thinking_level: str,
    temperature: float = 0.2,
    response_mime_type: Optional[str] = None,
    response_schema=None,
    system_instruction: Optional[str] = None,
    tools=None,
    include_thinking: bool = True,
) -> types.GenerateContentConfig:
    """
    Builds a types.GenerateContentConfig that maps thinking_level to ThinkingConfig(thinking_level=...)
    for Gemini 3/4 models or ThinkingConfig(thinking_budget=...) for legacy Gemini 2.5 models.
    """
    cfg_kwargs = {"temperature": temperature}
    if response_mime_type:
        cfg_kwargs["response_mime_type"] = response_mime_type
    if response_schema is not None:
        cfg_kwargs["response_schema"] = response_schema
    if system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction
    if tools is not None:
        cfg_kwargs["tools"] = tools

    if include_thinking:
        spec = GEMINI_MODEL_CATALOG.get(model_name, GEMINI_MODEL_CATALOG["gemini-3.8-flash"])
        lvl = (thinking_level or spec["default_thinking"]).lower()
        try:
            if spec["family"] == "gemini-2.5":
                budget_map = {"minimal": 0 if "flash" in model_name else 512, "low": 1024, "medium": 4096, "high": 16384, "xhigh": 24576}
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget_map.get(lvl, 4096))
            else:
                valid_lvls = spec["supported_thinking"]
                if lvl not in valid_lvls:
                    lvl = "high" if lvl == "xhigh" else ("low" if lvl == "minimal" else spec["default_thinking"])
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=lvl)
        except Exception:
            pass

    return types.GenerateContentConfig(**cfg_kwargs)


def call_gemini_with_config(
    contents: str,
    temperature: float = 0.2,
    response_mime_type: Optional[str] = None,
    response_schema=None,
    system_instruction: Optional[str] = None,
    tools=None,
):
    """
    Executes client.models.generate_content using the active model and thinking level,
    with automatic fallback if a preview/frontier model or thinking_config is unavailable.
    """
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("KDD_GEMINI_API_KEY is not configured.")

    primary_model = ACTIVE_GENAI_SETTINGS.get("model", "gemini-3.8-flash")
    thinking_lvl = ACTIVE_GENAI_SETTINGS.get("thinking_level", "medium")
    candidate_models = list(dict.fromkeys([primary_model, "gemini-3.8-flash", "gemini-2.5-flash"]))

    last_err = None
    for m_name in candidate_models:
        for use_think in (True, False):
            try:
                cfg = _build_genai_config(
                    model_name=m_name,
                    thinking_level=thinking_lvl,
                    temperature=temperature,
                    response_mime_type=response_mime_type,
                    response_schema=response_schema,
                    system_instruction=system_instruction,
                    tools=tools,
                    include_thinking=use_think,
                )
                resp = client.models.generate_content(model=m_name, contents=contents, config=cfg)
                return resp, m_name, thinking_lvl
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"All Gemini model attempts failed. Last error: {last_err}")
