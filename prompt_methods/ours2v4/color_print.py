def printred(msg):
    print("\033[91m" + msg + "\033[0m")


def printgreen(msg):
    print("\033[92m" + msg + "\033[0m")


def printblue(msg):
    print("\033[94m" + msg + "\033[0m")


def printyellow(msg):
    print("\033[93m" + msg + "\033[0m")


def printpurple(msg):
    print("\033[95m" + msg + "\033[0m")


def printcyan(msg):
    print("\033[96m" + msg + "\033[0m")


def printpink(msg):
    print("\033[97m" + msg + "\033[0m")


def printwhite(msg):
    print("\033[98m" + msg + "\033[0m")


def printbold(msg):
    print("\033[1m" + msg + "\033[0m")


def get_logger(name, dir_name="logs", console_output=True):
    # 获取日志记录器
    logger = logging.getLogger(name)
    # 关键：显式设置logger级别为INFO（必须取消注释）
    logger.setLevel(logging.INFO)  

    # 避免重复添加处理器
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建日志目录（如果不存在）
    log_dir = dir_name
    os.makedirs(log_dir, exist_ok=True)

    # 创建文件处理器，将日志保存到指定文件
    log_file = os.path.join(log_dir, f"{name}_logs.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5  # 最多保存5个备份文件
    )

    # 设置日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    # 可选：给handler单独设置级别（确保不被过滤）
    file_handler.setLevel(logging.INFO)
    
    # 添加文件处理器
    logger.addHandler(file_handler)
    
    # 添加控制台处理器（如果需要）
    if console_output:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)  # 可选：控制台handler级别
        logger.addHandler(stream_handler)
    
    return logger
