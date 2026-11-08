import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

# 全局字典，缓存已创建的 Logger 实例（key: logger name, value: logger 对象）
LOGGER_CACHE = {}

def get_logger(
    name: str, 
    dir_name: str = "logs", 
    console_output: bool = True
):
    """
    获取单例 Logger 实例（避免重复初始化）
    :param name: Logger 名称（唯一标识，决定是否复用）
    :param dir_name: 日志目录
    :param console_output: 是否输出到控制台
    :return: 单例 Logger 实例
    """
    log_file = os.path.join(dir_name, f"{name}_logs.log")
    
    # 第一步：检查缓存，若已存在则直接返回（核心：避免重复初始化）
    if name in LOGGER_CACHE:
        return LOGGER_CACHE[name],log_file
    
    # 第二步：首次调用，初始化 Logger（保留你原有的逻辑）
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # 必须显式设置级别
    logger.propagate = False  # 禁止日志向上传播到根 Logger（避免重复输出）
    
    # 清空可能的残留 handlers（首次调用时通常为空，兜底用）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建日志目录
    os.makedirs(dir_name, exist_ok=True)

    # 文件处理器
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 控制台处理器
    if console_output:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

    # 第三步：存入缓存，后续复用
    LOGGER_CACHE[name] = logger
    
    # logger.info(f"\033[31m当前处理的数据集id为：{data_id}\033[0m")
    
    return logger,log_file