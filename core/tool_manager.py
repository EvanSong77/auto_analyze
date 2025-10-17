# -*- coding: utf-8 -*-
# @Time    : 2025/10/13 11:18
# @Author  : EvanSong
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "success": self.success,
            "execution_time": self.execution_time
        }

        if self.success:
            result["data"] = self.data
        else:
            result["error"] = self.error

        if self.metadata:
            result["metadata"] = self.metadata

        return result


class Tool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.execution_count = 0
        self.total_time = 0.0
        self.error_count = 0

    @abstractmethod
    async def execute(
            self,
            arguments: Dict[str, Any],
            context: Dict[str, Any]
    ) -> ToolResult:
        """执行工具

        Args:
            arguments: 工具参数
            context: 执行上下文

        Returns:
            ToolResult: 执行结果
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """获取工具的JSON Schema"""
        pass

    async def __call__(
            self,
            arguments: Dict[str, Any],
            context: Dict[str, Any]
    ) -> ToolResult:
        """使工具可调用"""
        start_time = time.time()
        self.execution_count += 1

        try:
            result = await self.execute(arguments, context)
            result.execution_time = time.time() - start_time
            self.total_time += result.execution_time

            if not result.success:
                self.error_count += 1

            return result

        except Exception as e:
            logger.error(f"工具执行失败 [{self.name}]: {e}", exc_info=True)
            self.error_count += 1

            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def get_statistics(self) -> Dict[str, Any]:
        """获取工具统计信息"""
        return {
            "name": self.name,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "error_rate": (
                self.error_count / self.execution_count
                if self.execution_count > 0 else 0
            ),
            "total_time": self.total_time,
            "avg_time": (
                self.total_time / self.execution_count
                if self.execution_count > 0 else 0
            )
        }


class ReadDirectoryTool(Tool):
    """读取目录工具"""

    def __init__(self, filesystem_manager):
        super().__init__(
            name="read_directory",
            description="读取指定目录下的文件列表，返回文件名、类型、大小等信息"
        )
        self.fs_manager = filesystem_manager

    async def execute(
            self,
            arguments: Dict[str, Any],
            context: Dict[str, Any]
    ) -> ToolResult:
        """执行目录读取"""
        try:
            path = arguments.get("path")
            result = self.fs_manager.get_files(path)

            if result.get("status") == "error":
                return ToolResult(
                    success=False,
                    data=None,
                    error=result.get("error")
                )

            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "path": path or "root",
                    "item_count": len(result.get("items", []))
                }
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def get_schema(self) -> Dict[str, Any]:
        """获取工具Schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "相对于工作目录的子目录路径（可选）"
                        }
                    },
                    "required": []
                }
            }
        }


class ReadFilesTool(Tool):
    """读取文件工具"""

    def __init__(self, filesystem_manager):
        super().__init__(
            name="read_files",
            description="读取一个或多个文件的内容，支持文本、CSV、Excel、Word等格式"
        )
        self.fs_manager = filesystem_manager

    async def execute(
            self,
            arguments: Dict[str, Any],
            context: Dict[str, Any]
    ) -> ToolResult:
        """执行文件读取"""
        try:
            filenames = arguments.get("filenames", [])

            if not filenames:
                return ToolResult(
                    success=False,
                    data=None,
                    error="未提供文件名"
                )

            results = []
            for filename in filenames:
                file_result = self.fs_manager.process_file(filename)
                results.append(file_result)

            # 检查是否所有文件都成功读取
            success = all(
                r.get("status") == "success" for r in results
            )

            return ToolResult(
                success=success,
                data=results,
                metadata={
                    "file_count": len(filenames),
                    "filenames": filenames
                }
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def get_schema(self) -> Dict[str, Any]:
        """获取工具Schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filenames": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "要读取的文件路径列表"
                        }
                    },
                    "required": ["filenames"]
                }
            }
        }


class ExecCodeTool(Tool):
    """代码执行工具"""

    def __init__(self, execution_engine, conversation_id: str):
        super().__init__(
            name="exec_code",
            description="执行Python代码，支持数据处理和可视化"
        )
        self.execution_engine = execution_engine
        self.conversation_id = conversation_id

    async def execute(
            self,
            arguments: Dict[str, Any],
            context: Dict[str, Any]
    ) -> ToolResult:
        """执行代码"""
        try:
            code = arguments.get("code")

            if not code:
                return ToolResult(
                    success=False,
                    data=None,
                    error="未提供代码"
                )

            # 执行代码
            from core.agent.exec_code import exec_code
            result = await exec_code(
                code=code,
                conversation_id=self.conversation_id
            )

            success = result.get("status") == "success"

            return ToolResult(
                success=success,
                data=result,
                error=result.get("message") if not success else None,
                metadata={
                    "output_count": len(result.get("output", []))
                }
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def get_schema(self) -> Dict[str, Any]:
        """获取工具Schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "要执行的Python代码"
                        }
                    },
                    "required": ["code"]
                }
            }
        }


class InstallPackageTool(Tool):
    """安装包工具"""

    def __init__(self):
        super().__init__(
            name="install_package",
            description="安装Python包到当前环境"
        )

    async def execute(
            self,
            arguments: Dict[str, Any],
            context: Dict[str, Any]
    ) -> ToolResult:
        """执行包安装"""
        try:
            package_name = arguments.get("package_name")

            if not package_name:
                return ToolResult(
                    success=False,
                    data=None,
                    error="未提供包名"
                )

            # 调用安装函数
            from core.agent.functions import install_package
            result = await install_package(package_name)

            success = result.get("status") == "success"

            return ToolResult(
                success=success,
                data=result,
                error=result.get("message") if not success else None,
                metadata={"package": package_name}
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def get_schema(self) -> Dict[str, Any]:
        """获取工具Schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "package_name": {
                            "type": "string",
                            "description": "要安装的包名（可包含版本号）"
                        }
                    },
                    "required": ["package_name"]
                }
            }
        }


class ToolManager:
    """工具管理器 - 统一管理所有工具"""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.tools: Dict[str, Tool] = {}
        self.callbacks: List[Callable] = []

        # 注册默认工具
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """注册默认工具"""
        from core.filesystem import FileSystemManager
        from core.jupyter_execution import jupyter_execution_engine
        import os

        fs_manager = FileSystemManager()
        if os.environ.get("DATA_PATH"):
            fs_manager.set_workspace(os.environ.get("DATA_PATH"))

        # 注册工具
        self.register_tool(ReadDirectoryTool(fs_manager))
        self.register_tool(ReadFilesTool(fs_manager))
        self.register_tool(ExecCodeTool(
            jupyter_execution_engine,
            self.conversation_id
        ))
        self.register_tool(InstallPackageTool())

        logger.info(f"已注册 {len(self.tools)} 个工具")

    def register_tool(self, tool: Tool) -> None:
        """注册工具

        Args:
            tool: 工具实例
        """
        if tool.name in self.tools:
            logger.warning(f"工具已存在，将被覆盖: {tool.name}")

        self.tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister_tool(self, tool_name: str) -> bool:
        """注销工具

        Args:
            tool_name: 工具名称

        Returns:
            是否成功注销
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"注销工具: {tool_name}")
            return True
        return False

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """获取工具实例

        Args:
            tool_name: 工具名称

        Returns:
            Tool实例或None
        """
        return self.tools.get(tool_name)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具的Schema

        Returns:
            工具Schema列表
        """
        return [tool.get_schema() for tool in self.tools.values()]

    async def execute_tool(
            self,
            tool_name: str,
            arguments: Dict[str, Any],
            context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """执行工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            context: 执行上下文

        Returns:
            ToolResult: 执行结果
        """
        tool = self.get_tool(tool_name)

        if not tool:
            logger.error(f"工具不存在: {tool_name}")
            return ToolResult(
                success=False,
                data=None,
                error=f"工具不存在: {tool_name}"
            )

        # 准备上下文
        exec_context = context or {}
        exec_context["conversation_id"] = self.conversation_id

        # 执行工具
        logger.info(f"执行工具: {tool_name}")
        result = await tool(arguments, exec_context)

        # 通知回调
        await self._notify_callbacks(tool_name, arguments, result)

        return result

    def register_callback(self, callback: Callable) -> None:
        """注册执行回调

        Args:
            callback: 回调函数
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def unregister_callback(self, callback: Callable) -> None:
        """注销回调

        Args:
            callback: 回调函数
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    async def _notify_callbacks(
            self,
            tool_name: str,
            arguments: Dict[str, Any],
            result: ToolResult
    ) -> None:
        """通知所有回调"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tool_name, arguments, result)
                else:
                    callback(tool_name, arguments, result)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取所有工具的统计信息"""
        return {
            "tools": [
                tool.get_statistics()
                for tool in self.tools.values()
            ],
            "total_executions": sum(
                tool.execution_count for tool in self.tools.values()
            ),
            "total_errors": sum(
                tool.error_count for tool in self.tools.values()
            )
        }

    def reset_statistics(self) -> None:
        """重置统计信息"""
        for tool in self.tools.values():
            tool.execution_count = 0
            tool.total_time = 0.0
            tool.error_count = 0


if __name__ == "__main__":
    # 测试工具管理器
    async def test_tool_manager():
        manager = ToolManager(conversation_id="test")

        # 测试读取目录
        result = await manager.execute_tool(
            tool_name="read_directory",
            arguments={"path": None}
        )
        print(f"读取目录: {result.success}")

        # 获取统计
        stats = manager.get_statistics()
        print(f"统计信息: {stats}")


    asyncio.run(test_tool_manager())
