import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import retroai
from retroai.core import config
from retroai.ui import workbench


def test_headless_imports_do_not_initialize_data_services(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(retroai.__file__).resolve().parents[1])
    env.pop("KDD_GEMINI_API_KEY", None)
    env.pop("HF_TOKEN", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path

import retroai
from retroai.core import config, state, schemas, chem_utils, resolver
from retroai.ord import fingerprints, parser, validator
from retroai.ai import gemini_client, literature, classifier
from retroai.engine import aizynth, pipeline

assert 'google.colab' not in sys.modules
assert 'aizynthfinder' not in sys.modules
assert validator.get_ord_validator.cache_info().currsize == 0
assert not Path('data').exists()
try:
    aizynth.get_finder()
except FileNotFoundError as exc:
    assert 'data/config.yml' in str(exc)
else:
    raise AssertionError('finder initialized without data/config.yml')
assert 'aizynthfinder' not in sys.modules
""",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )


@pytest.mark.parametrize("with_options", [False, True])
def test_onnx_session_patch_enforces_memory_options(monkeypatch, with_options):
    seen = {}

    def record_init(session, path, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(config, "_orig_ort_init", record_init)
    options = config.ort.SessionOptions() if with_options else None
    provider_options = [{}]
    config.ort.InferenceSession.__init__(
        object(), "model.onnx", options, ["CPUExecutionProvider"], provider_options
    )
    effective = seen["sess_options"]
    if with_options:
        assert effective is options
    assert not effective.enable_cpu_mem_arena
    assert not effective.enable_mem_pattern
    assert effective.intra_op_num_threads == 2
    assert effective.inter_op_num_threads == 1
    assert seen["providers"] == ["CPUExecutionProvider"]
    assert seen["provider_options"] is provider_options
    assert os.environ["ORT_DISABLE_MEMORY_ARENA"] == "1"


def test_get_secret_uses_environment_without_colab(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.setenv("RETROAI_TEST_SETTING", "from-environment")
    assert config.get_secret("RETROAI_TEST_SETTING") == "from-environment"


def test_workbench_registers_the_callbacks_used_by_the_frontend(monkeypatch):
    callbacks = {}
    colab = ModuleType("google.colab")
    colab.output = SimpleNamespace(register_callback=callbacks.__setitem__)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    finder = SimpleNamespace()
    shard_sync_calls = []

    def sync_shards(**kwargs):
        shard_sync_calls.append(kwargs)
        return {"added": 0}

    index = SimpleNamespace(get_index_size=lambda: 0, sync_huggingface_parquet_shards=sync_shards)
    monkeypatch.setattr(workbench, "get_finder", lambda: finder)
    monkeypatch.setattr(workbench, "get_ord_validator", lambda: index)
    monkeypatch.setattr(workbench, "_ensure_ketcher_bundle", lambda: ("PGh0bWw+PC9odG1sPg==", 0))
    monkeypatch.setattr(workbench, "clear_output", lambda **kwargs: None)
    rendered = []
    monkeypatch.setattr(workbench, "display", lambda output: rendered.append(output.data))

    workbench.launch_workbench()

    invoked = set(re.findall(r'invokeFunction\("(chemforge\.\w+)"', rendered[0]))
    assert invoked == callbacks.keys()
    assert len(callbacks) == 14
    canvas = callbacks["chemforge.sync_canvas"]("CCO", "[0]").data
    assert canvas["smiles"] == "CCO"
    assert canvas["locked_atoms"] == [0]
    assert canvas["locked_smarts"]
    assert callbacks["chemforge.agent_status"]().data["target_smiles"] == "CCO"
    callbacks["chemforge.sync_ord_shards"]()
    assert shard_sync_calls == [{"max_shards": 1, "route_steps": []}]


def test_agent_reply_preserves_only_plain_formatting_tags():
    assert workbench._format_agent_reply("<b>Route</b> <em>note</em>") == "<b>Route</b> &lt;em&gt;note&lt;/em&gt;"
