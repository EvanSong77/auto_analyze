# -*- coding: utf-8 -*-
# @Time    : 2025/10/15
# @Author  : EvanSong
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.agent.collaboration import EnhancedMultiAgentSystem
from core.agent.multi_agent_system import MultiAgentSystem
from core.agent.report_generator import ReportGeneratorAgent
from core.model_client import ModelClient
from core.tool_manager import ToolManager
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedAnalysisSystem:
    """增强的分析系统 - 整合多智能体和报告生成"""

    def __init__(
            self,
            model_client: ModelClient,
            tool_manager: ToolManager,
            conversation_id: str
    ):
        self.conversation_id = conversation_id
        self.model_client = model_client
        self.tool_manager = tool_manager

        # 创建基础多智能体系统
        self.base_system = MultiAgentSystem(model_client, tool_manager, conversation_id)

        # 创建增强系统（带协作）
        self.enhanced_system = EnhancedMultiAgentSystem(self.base_system)

        # 创建报告生成器
        self.report_generator = ReportGeneratorAgent(model_client, tool_manager, conversation_id)

        # 系统状态
        self.analysis_results: List[Dict[str, Any]] = []
        self.final_report: Optional[str] = None
        self.execution_stats: Dict[str, Any] = {}

    async def analyze(self, user_query: str) -> str:
        """执行完整的分析流程"""
        start_time = time.time()

        try:
            logger.info(f"开始增强分析: {user_query}")

            # 1. 使用增强系统执行分析
            analysis_results = await self.enhanced_system.enhanced_process_request(user_query)
            logger.info(f"分析结果: {analysis_results}")

            # 2. 收集分析结果
            self.analysis_results = self._collect_analysis_results()

            # 3. 生成HTML报告
            # 安全处理user_query切片，确保是字符串
            title_suffix = str(user_query)[:50] + "..." if len(str(user_query)) > 50 else str(user_query)
            self.final_report = await self.report_generator.generate_report(
                user_query=user_query,
                analysis_results=self.analysis_results,
                report_title=f"数据分析报告 - {title_suffix}"
            )

            # 4. 验证报告质量
            validation_results = await self.report_generator.validate_report(self.final_report)

            # 5. 记录执行统计
            self.execution_stats = {
                "total_time": time.time() - start_time,
                "analysis_successful": True,
                "report_generated": True,
                "report_validation": validation_results,
                "system_stats": self.enhanced_system.get_collaboration_stats(),
                "base_system_stats": self.base_system.get_system_status()
            }

            logger.info(f"分析完成，耗时: {self.execution_stats['total_time']:.2f}秒")

            return self.final_report

        except Exception as e:
            logger.error(f"增强分析失败: {e}")

            # 记录错误统计
            self.execution_stats = {
                "total_time": time.time() - start_time,
                "analysis_successful": False,
                "error": str(e),
                "report_generated": False
            }

            # 生成错误报告
            self.final_report = self._generate_error_report(user_query, str(e))

            return self.final_report

    def _collect_analysis_results(self) -> List[Dict[str, Any]]:
        """收集分析结果"""
        results = []

        for task_id, task in self.base_system.tasks.items():
            if task.status == "completed":
                results.append({
                    "task_id": task_id,
                    "task_description": task.description,
                    "agent_role": task.agent_role.value,
                    "result": task.result,
                    "completion_time": task.completed_at
                })

        return results

    def _generate_error_report(self, user_query: str, error_message: str) -> str:
        """生成错误报告"""

        error_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>分析错误报告</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .error-header {{ background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row error-header p-4 mb-4">
            <div class="col">
                <h1>分析错误报告</h1>
                <p class="lead">用户需求: {user_query}</p>
            </div>
        </div>
        
        <div class="row">
            <div class="col">
                <div class="alert alert-danger">
                    <h4>分析过程中出现错误</h4>
                    <p><strong>错误信息:</strong> {error_message}</p>
                    <p><strong>建议:</strong> 请检查数据文件是否存在，或联系系统管理员。</p>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h5>系统状态</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>执行时间:</strong> {self.execution_stats.get('total_time', 0):.2f}秒</p>
                        <p><strong>完成的任务:</strong> {len([t for t in self.base_system.tasks.values() if t.status == 'completed'])}</p>
                        <p><strong>失败的任务:</strong> {len([t for t in self.base_system.tasks.values() if t.status == 'failed'])}</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

        return error_html

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "conversation_id": self.conversation_id,
            "analysis_completed": self.final_report is not None,
            "execution_stats": self.execution_stats,
            "analysis_results_count": len(self.analysis_results),
            "collaboration_stats": self.enhanced_system.get_collaboration_stats()
        }

    async def close(self):
        """关闭系统资源"""
        await self.enhanced_system.close()


class AnalysisOrchestrator:
    """分析编排器 - 高级接口"""

    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.active_systems: Dict[str, EnhancedAnalysisSystem] = {}

    async def create_analysis_session(self, session_id: str) -> EnhancedAnalysisSystem:
        """创建分析会话"""
        from core.model_client import create_client
        from config.config import ModelConfig

        # 创建模型客户端
        model_config = ModelConfig(**self.model_config)
        model_client = create_client(model_config)

        # 创建工具管理器
        tool_manager = ToolManager(session_id)

        # 创建增强分析系统
        system = EnhancedAnalysisSystem(model_client, tool_manager, session_id)
        self.active_systems[session_id] = system

        return system

    async def execute_analysis(self, session_id: str, user_query: str) -> str:
        """执行分析"""
        if session_id not in self.active_systems:
            system = await self.create_analysis_session(session_id)
        else:
            system = self.active_systems[session_id]

        return await system.analyze(user_query)

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话状态"""
        if session_id in self.active_systems:
            return self.active_systems[session_id].get_system_status()
        return None

    async def close_session(self, session_id: str):
        """关闭会话"""
        if session_id in self.active_systems:
            await self.active_systems[session_id].close()
            del self.active_systems[session_id]


def create_enhanced_system(
        model_client: ModelClient,
        tool_manager: ToolManager,
        conversation_id: str
) -> EnhancedAnalysisSystem:
    """创建增强分析系统"""
    return EnhancedAnalysisSystem(model_client, tool_manager, conversation_id)


def create_orchestrator(model_config: Dict[str, Any]) -> AnalysisOrchestrator:
    """创建分析编排器"""
    return AnalysisOrchestrator(model_config)


# 便捷函数
def create_analysis_system_from_config(config_path: str) -> AnalysisOrchestrator:
    """从配置文件创建分析系统"""
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 获取模型配置
    model_config = config.get('model', {})

    return create_orchestrator(model_config)
