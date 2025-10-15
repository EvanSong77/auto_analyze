# -*- coding: utf-8 -*-
# @Time    : 2025/10/13 11:12
# @Author  : EvanSong
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

import coloredlogs

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}


def setup_logging(
        level: Union[int, str] = "info",
        log_file: Optional[str] = "app.log",
        log_dir: Optional[Union[str, Path]] = "./logs",
        max_bytes: int = 50 * 1024 * 1024,
        backup_count: int = 5
) -> None:
    """
    初始化日志配置

    Args:
        level: 日志级别，可为字符串或整数
        log_file: 日志文件名
        log_dir: 日志目录
        max_bytes: 单个日志文件最大大小（字节）
        backup_count: 保留的历史日志文件数
    """
    # 转换日志级别
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.lower(), logging.INFO)

    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    # 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.propagate = False  # 防止重复打印

    # 避免重复添加 handler
    if root_logger.handlers:
        return

    # 文件处理器（带轮转）
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 彩色控制台处理器
    console_handler = logging.StreamHandler()
    console_formatter = coloredlogs.ColoredFormatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
        level_styles={
            'debug': {'color': 'cyan'},
            'info': {'color': 'green'},
            'warning': {'color': 'yellow'},
            'error': {'color': 'red', 'bold': True},
            'critical': {'color': 'magenta', 'bold': True}
        },
        field_styles={
            'asctime': {'color': 'blue'},
            'levelname': {'color': 'black', 'bold': True},
            'filename': {'color': 'blue'},
            'lineno': {'color': 'blue'}
        }
    )
    console_handler.setFormatter(console_formatter)

    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取日志器
    保持调用方式不变： logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def get_module_logger() -> logging.Logger:
    """
    自动根据调用模块名返回 logger
    """
    import inspect
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    return get_logger(module.__name__ if module else None)


setup_logging(level="info", log_dir="./logs")
