import os
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import time
import json
import torch
from mllms.base_models import Model, Llama_Model, Qwen25_Model ,Qwen3_Model
from utils.generate import generate
from utils.extract_answer import extract
from utils.get_scores import scoring

from tools.color_print import *
from tools.create_logger import get_logger

from sentence_transformers import SentenceTransformer # 加载嵌入模型用的

# A logger for this file
# logging.getLogger("httpx").setLevel(logging.WARNING) 
logging.getLogger("httpx")
log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base=None)

def main(cfg):
    # 允许动态键
    OmegaConf.set_struct(cfg, False)
    
    # embedding_model = None
    
    # if cfg.embedding_model_path != "none":
    #     printyellow(f"需要加载了嵌入模型 {cfg.embedding_model_path}")
    #     # embedding_model = SentenceTransformer(cfg.embedding_model_path)
    
    # 导入ost_step 的文件
    
    ost_sample = None
    if cfg.prompt_method.name.startswith("our") or cfg.prompt_method.name in ["mctsv8v2"]:
        if cfg.prompt_method.ost_step_go:
            with open(cfg.prompt_method.fewshot_ost_prompt, "r", encoding="utf-8") as f:
                cfg.fewshot_ost_prompt = f.read().strip()

            with open(cfg.prompt_method.fewshot_ost_config, "r", encoding="utf-8") as f:
                cfg.fewshot_ost_config = json.load(f)
    
        with open(cfg.prompt_method.fewshot_cot_prompt, "r", encoding="utf-8") as f:
            cfg.fewshot_cot_prompt = f.read().strip()

        with open(cfg.prompt_method.fewshot_cot_config, "r", encoding="utf-8") as f:
            cfg.fewshot_cot_config = json.load(f)
            
    # 告诉你这个文件是干嘛
    log.info(f"file_introduction : {cfg.prompt_method.file_introduction}")
    log.info(f"Working directory : {os.getcwd()}")
    # print(HydraConfig.get())
    output_dir = HydraConfig.get().runtime.output_dir
    # 添加新的键
    cfg.output_dir = output_dir
    
    # # 写入一些运行中记录的PK信息
    # PK_logger = get_logger("PK",os.path.join(output_dir,cfg.desc))
    
    extract_result_dir = os.path.join(output_dir, f"ext_llm={cfg.ext_llm.name}")
    extract_result_dir = os.path.join(extract_result_dir, f"method={cfg.extract_method}")
    # extract_result_dir = os.path.join(extract_result_dir, f"reward_method={cfg.reward_method}")
    printgreen(extract_result_dir)
    
    if hasattr(cfg, "extract_field"): # 是否有这样的属性
        extract_result_dir = os.path.join(extract_result_dir, f"field={cfg.extract_field}")
    extract_result_dir = os.path.join(extract_result_dir, f"n={cfg.ext_llm.n}")
    extract_result_dir = os.path.join(extract_result_dir, f"reward_method={cfg.reward_method}")
    # extract_result_dir = os.path.join(extract_result_dir, f"rollout={cfg.prompt_method.rollout_num}")
    printblue(f"{extract_result_dir}")
    
    cfg.extract_result_dir = extract_result_dir
    Path(output_dir, "config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    os.makedirs(extract_result_dir, exist_ok=True)
    # 重新启用结构化模式
    OmegaConf.set_struct(cfg, True)
    log.info(f"Output directory  : {output_dir}")
    print("看我好看的输出")
    log.info("\n" + OmegaConf.to_yaml(cfg)) # 非常好的形式化方法 牛逼了
    # return cfg
    
    if cfg.mode == "generate":
        model = Model(cfg)
        generate(cfg, model)
        
    if cfg.mode == "extract":
        # Load the model
        if cfg.ext_llm.name == "qwen25" or cfg.ext_llm.name == "qwen2":
            model = Qwen25_Model(cfg.ext_llm)
        elif "Llama" in cfg.ext_llm.name or "mistral" in cfg.ext_llm.name:
            model = Llama_Model(cfg.ext_llm)
        elif cfg.ext_llm.name == "qwen3":
            model = Qwen3_Model(cfg.ext_llm)
        else:
            raise ValueError(f"不支持的ext_llm模型: {cfg.ext_llm.name}")
        
        printgreen(f"{cfg.ext_llm.name}模型加载完毕")
        
        extract(cfg, model)

    if cfg.mode == "scoring":
        scoring(cfg)

    
if __name__ == "__main__":
    main()
