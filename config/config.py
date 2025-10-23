# -*- coding: utf-8 -*-
# @Time    : 2025/10/13 11:11
# @Author  : EvanSong
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.logger import get_logger

logger = get_logger(__name__)


class ModelConfig(BaseModel):
    """模型配置 - 增强版本"""
    type: str = Field(description="模型类型: openai, anthropic, azure等")
    api_key: str = Field(description="API密钥")
    endpoint: str = Field(description="API端点URL")
    model: str = Field(description="模型名称")
    max_tokens: int = Field(default=4096, description="最大token数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    timeout: int = Field(default=120, gt=0, description="请求超时时间(秒)")
    max_retries: int = Field(default=3, ge=0, description="最大重试次数")

    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """验证端点URL格式"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('端点URL必须以http://或https://开头')
        return v.rstrip('/')

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """验证API密钥不为空"""
        if not v or v.strip() == '':
            raise ValueError('API密钥不能为空')
        return v


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = Field(default="127.0.0.1", description="服务器主机")
    port: int = Field(default=8000, ge=1, le=65535, description="服务器端口")
    frontend_port: int = Field(default=5173, ge=1, le=65535, description="前端端口")
    workers: int = Field(default=1, ge=1, description="工作进程数")
    reload: bool = Field(default=False, description="开发模式自动重载")


class ExecutionConfig(BaseModel):
    """执行配置"""
    max_execution_time: int = Field(default=300, gt=0, description="最大执行时间(秒)")
    max_memory_mb: int = Field(default=2048, gt=0, description="最大内存使用(MB)")
    code_timeout: int = Field(default=60, gt=0, description="代码执行超时(秒)")
    enable_gpu: bool = Field(default=False, description="是否启用GPU")
    max_concurrent_executions: int = Field(default=5, ge=1, description="最大并发执行数")


class FileConfig(BaseModel):
    """文件处理配置"""
    max_file_size_mb: int = Field(default=100, gt=0, description="最大文件大小(MB)")
    preview_rows: int = Field(default=20, ge=1, description="数据预览行数")
    max_preview_chars: int = Field(default=10000, ge=100, description="文本预览字符数")
    supported_formats: list[str] = Field(
        default=['csv', 'xlsx', 'xls', 'txt', 'json', 'docx', 'pptx', 'pdf'],
        description="支持的文件格式"
    )


class Settings(BaseSettings):
    """应用全局配置 - 重构版本"""
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__',
        case_sensitive=False
    )

    # 基本信息
    version: str = Field(default="2.0.0", description="应用版本")
    environment: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=False, description="调试模式")

    # 工作目录
    workspace: Optional[Path] = Field(default=None, description="工作目录")
    config_path: Path = Field(default=Path("config.yaml"), description="配置文件路径")

    # 语言配置
    language: str = Field(default="zh", pattern="^(zh|en)$", description="界面语言")

    # 模型配置 - 提供默认值
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            type="openai",
            api_key="your-api-key-here",
            endpoint="https://api.openai.com/v1",
            model="gpt-4"
        ),
        description="主模型配置(工具调用)"
    )
    user_model: Optional[ModelConfig] = Field(default=None, description="用户代理模型配置")
    vision_model: Optional[ModelConfig] = Field(default=None, description="视觉模型配置")

    # 服务器配置
    server: ServerConfig = Field(default_factory=ServerConfig, description="服务器配置")

    # 执行配置
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="执行配置")

    # 文件配置
    file: FileConfig = Field(default_factory=FileConfig, description="文件配置")

    # 日志配置
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: Optional[Path] = Field(default=Path("logs/app.log"), description="日志文件路径")

    @field_validator('workspace')
    @classmethod
    def validate_workspace(cls, v: Optional[Path]) -> Optional[Path]:
        """验证工作目录"""
        if v is not None:
            if not v.exists():
                logger.warning(f"工作目录不存在: {v}")
            elif not v.is_dir():
                raise ValueError(f"工作目录必须是目录: {v}")
        return v

    @field_validator('config_path', mode='before')
    @classmethod
    def validate_config_path(cls, v: Any) -> Path:
        """验证配置文件路径"""
        if v is None:
            return Path("config.yaml")
        if isinstance(v, str):
            return Path(v)
        return v

    def model_post_init(self, __context: Any) -> None:
        """初始化后处理"""
        # 创建必要的目录
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # 如果未配置user_model，使用主模型配置
        if self.user_model is None:
            self.user_model = self.model.model_copy()
            logger.info("用户模型未配置，使用主模型配置")

    @classmethod
    def load_from_file(cls, config_path: Path) -> Settings:
        """从YAML文件加载配置

        Args:
            config_path: 配置文件路径

        Returns:
            Settings实例

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}

            # 确保配置文件中包含正确的 config_path
            config_data['config_path'] = str(config_path.resolve())

            # 合并环境变量和文件配置
            return cls(**config_data)
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise ValueError(f"加载配置失败: {e}")

    def save_to_file(self, config_path: Optional[Path] = None) -> None:
        """保存配置到文件

        Args:
            config_path: 配置文件路径，默认使用self.config_path
        """
        path = config_path or self.config_path
        path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为字典并保存
        config_dict = self.model_dump(mode='json', exclude={'config_path'})

        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                config_dict,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False
            )

        logger.info(f"配置已保存到: {path}")

    def get_model_config(self, model_type: str = "main") -> ModelConfig:
        """获取指定类型的模型配置

        Args:
            model_type: 模型类型 (main, user, vision)

        Returns:
            ModelConfig实例

        Raises:
            ValueError: 无效的模型类型
        """
        config_map = {
            "main": self.model,
            "user": self.user_model or self.model,
            "vision": self.vision_model
        }

        if model_type not in config_map:
            raise ValueError(f"无效的模型类型: {model_type}")

        config = config_map[model_type]
        if config is None:
            raise ValueError(f"模型配置未设置: {model_type}")

        return config


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置实例（单例模式）

    Returns:
        Settings实例
    """
    config_path = Path(Settings().config_path)

    try:
        if os.path.exists(config_path):
            settings = Settings.load_from_file(config_path)
            logger.info(f"从文件加载配置: {config_path}")
        else:
            # 使用默认配置，但设置正确的config_path
            settings = Settings()
            settings.config_path = config_path
            # 创建默认配置文件
            settings.save_to_file(config_path)
            logger.info(f"创建默认配置文件: {config_path}")

        return settings
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        # 返回默认配置，并设置正确的config_path
        settings = Settings()
        settings.config_path = config_path
        return settings


def reload_settings() -> Settings:
    """重新加载配置

    Returns:
        新的Settings实例
    """
    get_settings.cache_clear()
    return get_settings()


# 配置变更通知
class ConfigObserver:
    """配置观察者模式实现"""

    def __init__(self):
        self._observers: list[callable] = []

    def subscribe(self, callback: callable) -> None:
        """订阅配置变更"""
        if callback not in self._observers:
            self._observers.append(callback)

    def unsubscribe(self, callback: callable) -> None:
        """取消订阅"""
        if callback in self._observers:
            self._observers.remove(callback)

    def notify(self, settings: Settings) -> None:
        """通知所有观察者"""
        for callback in self._observers:
            try:
                callback(settings)
            except Exception as e:
                logger.error(f"配置通知失败: {e}")


# 全局配置观察者
config_observer = ConfigObserver()


if __name__ == "__main__":
    # 测试配置管理
    settings = get_settings()
    print(f"版本: {settings.version}")
    print(f"环境: {settings.environment}")
    print(f"主模型: {settings.model.model}")
    print(f"工作目录: {settings.workspace}")
    print(f"配置文件路径: {settings.config_path}")

    # 保存配置
    settings.save_to_file()
    print("配置已保存")
