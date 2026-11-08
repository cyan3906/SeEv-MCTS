import json
import pickle
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

import utils.generate as generate_module


def _write_dataset(path):
    path.write_text(
        json.dumps(
            [
                {
                    "source": "MATH500",
                    "id": "first",
                    "problem": "What is 1+1?",
                    "solution": "The answer is 2.",
                    "extra_info": {"answer": "2"},
                },
                {
                    "source": "MATH500",
                    "id": "second",
                    "problem": "What is 2+2?",
                    "solution": "The answer is 4.",
                    "extra_info": {"answer": "4"},
                },
            ]
        ),
        encoding="utf-8",
    )


def test_load_math500_records_uses_problem_ids_and_honors_limit(tmp_path):
    dataset = tmp_path / "math500.json"
    _write_dataset(dataset)

    all_records = generate_module.load_math500_records(dataset)
    one_record = generate_module.load_math500_records(dataset, max_samples=1)

    assert list(all_records) == ["first", "second"]
    assert all_records["first"]["problem"] == "What is 1+1?"
    assert list(one_record) == ["first"]


def test_gen_math500_persists_result_and_removes_pickle_objects(tmp_path):
    dataset = tmp_path / "math500.json"
    _write_dataset(dataset)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            name="math500",
            input_file=str(dataset),
            max_samples=1,
        ),
        output_dir=str(output_dir),
        prompt_method=SimpleNamespace(name="mctsv8_vpw", data_id=-1),
    )

    class FakeModel:
        calls = 0

        def get_response(self, question, image_path, is_pkl, tree_save_path):
            self.calls += 1
            assert question == "What is 1+1?"
            assert image_path is None
            assert is_pkl is False
            assert tree_save_path.endswith("first.pkl")
            return {
                "response": "2",
                "raw_response": "The final answer is 2.",
                "answer_extract_status": "ok",
                "memory": {"valid_vote_count": 8},
                "root": {"kind": "root"},
                "mcts": {"kind": "mcts"},
            }

    model = FakeModel()

    assert generate_module.gen_math500(cfg, model) == 1
    assert generate_module.gen_math500(cfg, model) == 1

    saved = json.loads((output_dir / "generation.json").read_text(encoding="utf-8"))
    assert model.calls == 1
    assert list(saved) == ["first"]
    assert saved["first"]["query"] == "What is 1+1?"
    assert saved["first"]["response"] == "2"
    assert saved["first"]["raw_response"] == "The final answer is 2."
    assert saved["first"]["answer_extract_status"] == "ok"
    assert "root" not in saved["first"]["decision"]
    assert "mcts" not in saved["first"]["decision"]
    assert (output_dir / "tree_results" / "first.pkl").exists()
    assert (output_dir / "tree_results" / "firstmcts.pkl").exists()


def test_gen_math500_ignores_root_only_incomplete_checkpoint(tmp_path):
    dataset = tmp_path / "math500.json"
    _write_dataset(dataset)
    output_dir = tmp_path / "output"
    tree_dir = output_dir / "tree_results"
    tree_dir.mkdir(parents=True)
    with (tree_dir / "first.pkl").open("wb") as handle:
        pickle.dump({"incomplete": True}, handle)
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            name="math500",
            input_file=str(dataset),
            max_samples=1,
        ),
        output_dir=str(output_dir),
        prompt_method=SimpleNamespace(name="mctsv8_vpw", data_id=-1),
    )

    class FakeModel:
        def get_response(self, question, image_path, is_pkl, tree_save_path):
            assert is_pkl is False
            return {
                "response": "2",
                "raw_response": "The final answer is 2.",
                "answer_extract_status": "ok",
                "root": {"kind": "new-root"},
                "mcts": {"kind": "new-mcts"},
            }

    assert generate_module.gen_math500(cfg, FakeModel()) == 1
    assert json.loads((output_dir / "generation.json").read_text())["first"][
        "response"
    ] == "2"


def test_atomic_json_dump_keeps_previous_results_when_serialization_fails(tmp_path):
    output = tmp_path / "generation.json"
    output.write_text('{"completed": true}', encoding="utf-8")

    with pytest.raises(TypeError):
        generate_module.atomic_json_dump(
            {"bad": object()},
            output,
        )

    assert json.loads(output.read_text(encoding="utf-8")) == {"completed": True}
    assert list(tmp_path.glob(".generation.json.*.tmp")) == []


def test_atomic_pickle_dump_keeps_previous_checkpoint_on_failure(tmp_path):
    checkpoint = tmp_path / "samplemcts.pkl"
    with checkpoint.open("wb") as handle:
        pickle.dump({"old": True}, handle)

    class Unpickleable:
        def __reduce__(self):
            raise TypeError("cannot pickle")

    with pytest.raises(TypeError):
        generate_module.atomic_pickle_dump(Unpickleable(), checkpoint)

    with checkpoint.open("rb") as handle:
        assert pickle.load(handle) == {"old": True}
    assert list(tmp_path.glob(".samplemcts.pkl.*.tmp")) == []


def test_generate_dispatches_math500(monkeypatch):
    captured = []
    monkeypatch.setattr(
        generate_module,
        "gen_math500",
        lambda cfg, model: captured.append((cfg, model)),
        raising=False,
    )
    cfg = SimpleNamespace(data=SimpleNamespace(name="math500"))
    model = object()

    generate_module.generate(cfg, model)

    assert captured == [(cfg, model)]


def test_math500_qwen25_and_vpw_configs_use_requested_values():
    config_root = Path(__file__).parents[1] / "configs"
    data_cfg = OmegaConf.load(config_root / "data" / "math500.yaml")
    llm_cfg = OmegaConf.load(config_root / "llm" / "qwen25_7b.yaml")
    vpw_cfg = OmegaConf.load(config_root / "prompt_method" / "mctsv8_vpw.yaml")

    assert data_cfg.name == "math500"
    assert data_cfg.input_file == "/home/jncsnlp4/SSD2/ZJJ/rStar/data/MATH/test_all.json"
    assert data_cfg.max_samples is None
    assert llm_cfg.name == "Qwen2.5-7B-Instruct"
    assert llm_cfg.text_only is True
    assert llm_cfg.path == "/home/jncsnlp4/SSD2/model/Qwen2.5-7b-Instruct"
    assert llm_cfg.call_method == "local"
    assert vpw_cfg.mcts_rollouts == 8
