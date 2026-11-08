import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir

from data.MathVista.evaluation.build_query import create_query_data
from mllms.base_models import Model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "configs"),
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=mathvista",
                "llm=qwen25vl_7b",
                "prompt_method=mctsv8",
                "prompt_method.mcts_rollouts=2",
                "prompt_method.tree_width=2",
                "prompt_method.decompose_question_num=2",
                "prompt_method.return_num=2",
            ],
        )

    input_file = PROJECT_ROOT / cfg.data.data_dir / cfg.data.input_file
    with input_file.open(encoding="utf-8") as handle:
        data = json.load(handle)

    problem_id = next(iter(data))
    query = create_query_data(data)[problem_id]
    image_name = os.path.basename(data[problem_id]["image"])
    image_path = PROJECT_ROOT / cfg.data.image_dir / image_name

    model = Model(cfg)
    result = model.get_response(
        query,
        str(image_path),
        is_pkl=False,
        tree_save_path=None,
    )

    print(
        json.dumps(
            {
                "problem_id": problem_id,
                "model": cfg.llm.name,
                "response": result["response"],
                "kept_root_children": len(result["root"].children),
                "pruning_enabled": cfg.prompt_method.prune_low_reward_nodes,
                "pruning_threshold": cfg.prompt_method.prune_reward_threshold,
                "is_p3": cfg.prompt_method.is_p3,
                "is_p3pro": cfg.prompt_method.is_p3pro,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
