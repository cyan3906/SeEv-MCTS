from types import SimpleNamespace

from hydra import compose, initialize_config_dir

import mllms.base_models as base_models


CONFIG_DIR = "/home/jncsnlp4/SSD2/ZJJ/VReST/configs"


def make_model(method_name):
    model = object.__new__(base_models.Model)
    model.prompt_cfg = SimpleNamespace(name=method_name)
    model.cfg = SimpleNamespace(prompt_method=model.prompt_cfg)
    model.model = object()
    model.embedding_model = None
    return model


def test_vpw_dispatch_is_explicit_and_legacy_mctsv8_stays_unchanged(monkeypatch):
    monkeypatch.setattr(
        base_models,
        "mcts_forward_vpw",
        lambda *args, **kwargs: "vpw",
        raising=False,
    )
    monkeypatch.setattr(
        base_models,
        "mcts_forward_v8",
        lambda *args, **kwargs: "legacy",
    )

    assert make_model("mctsv8_vpw").get_response("q", "image") == "vpw"
    assert make_model("mctsv8_a").get_response("q", "image") == "legacy"
    assert make_model("mctsv8_b").get_response("q", "image") == "legacy"
    assert make_model("mctsv8_c").get_response("q", "image") == "legacy"


def test_vpw_hydra_config_records_the_approved_search_policy():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=["prompt_method=mctsv8_vpw", "llm=qwen25vl_7b"],
        )

    assert cfg.prompt_method.name == "mctsv8_vpw"
    assert cfg.prompt_method.shallow_progress_threshold == 0.375
    assert cfg.prompt_method.deep_progress_threshold == 0.75
    assert cfg.prompt_method.reserve_activation_threshold == 0.8
    assert cfg.prompt_method.normal_tree_width == 5
    assert cfg.prompt_method.reserve_tree_width == 3
    assert cfg.prompt_method.deep_tree_width == 3
    assert cfg.prompt_method.use_p3pro_for_low_answer is False
    assert cfg.prompt_method.is_p3pro is False
    assert cfg.embedding_model_path == "none"
