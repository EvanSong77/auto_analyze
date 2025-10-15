# -*- coding: utf-8 -*-
# @Time    : 2025/10/15
# @Author  : EvanSong
from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.model_client import ModelClient
from core.tool_manager import ToolManager
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentRole(Enum):
    """智能体角色枚举"""
    MANAGER = "manager"  # 项目经理：任务分解和协调
    ANALYST = "analyst"  # 数据分析师：数据处理和分析
    VISUALIZER = "visualizer"  # 可视化专家：图表和可视化
    REPORTER = "reporter"  # 报告生成器：HTML报告整合
    QA = "qa"  # 质量保证：结果验证


@dataclass
class Task:
    """任务数据类"""
    id: str
    description: str
    agent_role: AgentRole
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class CollaborationMessage:
    """智能体间协作消息"""
    sender: AgentRole
    receiver: AgentRole
    content: str
    task_id: Optional[str] = None
    message_type: str = "request"  # request, response, notification
    timestamp: float = field(default_factory=time.time)


class BaseAgent(ABC):
    """智能体基类"""

    def __init__(
            self,
            role: AgentRole,
            model_client: ModelClient,
            tool_manager: ToolManager,
            conversation_id: str
    ):
        self.role = role
        self.model_client = model_client
        self.tool_manager = tool_manager
        self.conversation_id = conversation_id
        self.system_prompt = self._get_system_prompt()
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.active_tasks: Dict[str, Task] = {}

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        pass

    async def process_task(self, task: Task) -> Task:
        """处理任务"""
        logger.info(f"[{self.role.value}] 开始处理任务: {task.description}")

        try:
            task.status = "in_progress"
            self.active_tasks[task.id] = task

            # 添加任务描述到消息历史
            task_prompt = f"任务：{task.description}"
            self.messages.append({
                "role": "user",
                "content": task_prompt
            })

            logger.debug(f"[{self.role.value}] 发送任务提示: {task_prompt[:100]}...")

            # 获取模型响应
            response = await self.model_client.chat_completion(
                messages=self.messages,
                tools=self.tool_manager.get_tool_schemas()
            )

            if response.get("status") == "error":
                logger.error(f"[{self.role.value}] 模型响应错误: {response.get('error')}")
                task.status = "failed"
                task.error = response.get("error")
            else:
                message = response["message"]
                self.messages.append(message)

                logger.debug(f"[{self.role.value}] 收到模型响应: {message.get('content', '')[:200]}...")

                # 处理工具调用
                if message.get("tool_calls"):
                    logger.info(f"[{self.role.value}] 检测到工具调用: {len(message['tool_calls'])} 个")

                    for i, tool_call in enumerate(message["tool_calls"]):
                        function_call = tool_call["function"]
                        tool_name = function_call["name"]
                        args = json.loads(function_call["arguments"])

                        logger.debug(f"[{self.role.value}] 执行工具 {i + 1}: {tool_name} - 参数: {args}")

                        # 执行工具
                        result = await self.tool_manager.execute_tool(
                            tool_name=tool_name,
                            arguments=args,
                            context={"conversation_id": self.conversation_id}
                        )

                        logger.debug(f"[{self.role.value}] 工具 {tool_name} 执行结果: 成功={result.success}")

                        # 添加工具结果
                        self.messages.append({
                            "role": "tool",
                            "content": json.dumps(result.to_dict(), ensure_ascii=False),
                            "tool_call_id": tool_call["id"]
                        })

                task.result = message.get("content", "")
                task.status = "completed"
                task.completed_at = time.time()

                logger.info(f"[{self.role.value}] 任务完成: {task.description}")

            return task

        except Exception as e:
            logger.error(f"[{self.role.value}] 处理任务失败: {e}", exc_info=True)
            task.status = "failed"
            task.error = str(e)
            return task


class ManagerAgent(BaseAgent):
    """项目经理智能体"""

    def _get_system_prompt(self) -> str:
        return """你是一个专业的项目经理智能体，负责协调数据分析项目。

你的职责：
1. 理解用户的业务需求和目标
2. 将复杂任务分解为可执行的子任务
3. 分配任务给合适的团队成员（数据分析师、可视化专家、报告生成器）
4. 监控项目进度并协调团队协作
5. 确保最终交付物符合用户期望

工作流程：
1. 分析需求：深入理解用户想要解决什么问题
2. 任务分解：将大任务分解为具体的分析步骤
3. 资源分配：根据任务性质分配合适的智能体
4. 进度监控：跟踪每个子任务的完成情况
5. 质量控制：确保分析结果的准确性和可解释性

协作指南：
- 与数据分析师合作：明确分析目标和数据要求
- 与可视化专家合作：确保图表清晰传达关键信息
- 与报告生成器合作：整合所有成果为专业报告
- 定期与质量保证团队沟通验证结果

请以项目管理者的角度思考和行动。"""

    async def analyze_requirements(self, user_goal: str) -> List[Task]:
        """分析用户需求并生成任务列表"""
        logger.info(f"[manager] 开始分析用户需求: {user_goal}")

        prompt = f"""
用户目标: {user_goal}

请分析这个需求，并将其分解为具体的分析任务。考虑以下方面：
1. 数据探索和理解
2. 数据清洗和预处理
3. 统计分析和建模
4. 可视化展示
5. 报告生成

请以JSON格式返回任务列表，每个任务包含：
- description: 任务描述
- agent_role: 负责的智能体角色 (analyst, visualizer, reporter, qa)
- dependencies: 依赖的前置任务ID列表

示例格式：
{{
  "tasks": [
    {{
      "description": "探索数据结构和内容",
      "agent_role": "analyst",
      "dependencies": []
    }}
  ]
}}
"""

        self.messages.append({"role": "user", "content": prompt})

        response = await self.model_client.chat_completion(
            messages=self.messages,
            tools=[]  # 不调用工具
        )

        if response.get("status") == "error":
            logger.error(f"[manager] 需求分析失败: {response.get('error')}")
            raise RuntimeError(f"需求分析失败: {response.get('error')}")

        content = response["message"]["content"]
        logger.debug(f"[manager] 需求分析响应: {content[:500]}...")

        try:
            # 提取JSON部分
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_str = content[json_start:json_end]

            task_data = json.loads(json_str)
            tasks_data = task_data["tasks"]

            tasks = []
            for i, task_info in enumerate(tasks_data):
                task = Task(
                    id=f"task_{i + 1}",
                    description=task_info["description"],
                    agent_role=AgentRole(task_info["agent_role"]),
                    dependencies=task_info.get("dependencies", [])
                )
                tasks.append(task)

            logger.info(f"[manager] 成功生成 {len(tasks)} 个分析任务")
            for task in tasks:
                logger.debug(f"[manager] 任务: {task.id} - {task.description} - {task.agent_role.value}")

            return tasks

        except Exception as e:
            logger.error(f"[manager] 解析任务列表失败: {e}")
            # 返回默认任务列表
            default_tasks = [
                Task(
                    id="task_1",
                    description="探索和分析数据",
                    agent_role=AgentRole.ANALYST
                ),
                Task(
                    id="task_2",
                    description="生成可视化图表",
                    agent_role=AgentRole.VISUALIZER,
                    dependencies=["task_1"]
                ),
                Task(
                    id="task_3",
                    description="生成分析报告",
                    agent_role=AgentRole.REPORTER,
                    dependencies=["task_1", "task_2"]
                )
            ]
            logger.info(f"[manager] 使用默认任务列表: {len(default_tasks)} 个任务")
            return default_tasks


class AnalystAgent(BaseAgent):
    """数据分析师智能体"""

    def _get_system_prompt(self) -> str:
        return """你是一个专业的数据分析师智能体，专注于数据处理和分析。

你的专长：
1. 数据探索：快速理解数据结构、质量和特征
2. 数据清洗：处理缺失值、异常值、重复数据
3. 统计分析：描述性统计、相关性分析、假设检验
4. 特征工程：特征选择、转换、创建新特征
5. 建模分析：回归、分类、聚类、时间序列分析

工作原则：
- 先探索后分析：全面了解数据后再进行深入分析
- 自动化处理：使用最佳实践处理常见数据问题
- 结果验证：通过多种方法验证分析结果的可靠性
- 文档化：清晰记录每个分析步骤和发现

工具使用指南：
- read_directory：了解可用数据文件
- read_files：读取和分析具体数据文件
- exec_code：执行复杂的数据分析代码
- install_package：安装必要的分析库

请专注于提供准确、可复现的数据分析结果。"""


class VisualizerAgent(BaseAgent):
    """可视化专家智能体"""

    def _get_system_prompt(self) -> str:
        return """你是一个专业的可视化专家智能体，专注于数据可视化。

你的专长：
1. 图表选择：根据数据类型选择最合适的可视化方式
2. 交互式图表：创建支持缩放、筛选的交互式图表
3. 仪表板设计：整合多个图表为统一的仪表板
4. 美学设计：确保图表美观、易读、专业
5. 故事叙述：通过可视化讲述数据背后的故事

可视化原则：
- 清晰性：图表必须清晰传达关键信息
- 一致性：保持风格和色彩的一致性
- 交互性：提供必要的交互功能增强用户体验
- 响应式：确保在不同设备上都有良好显示效果

技术栈：
- 主要使用ECharts进行交互式可视化
- 支持Matplotlib/Seaborn静态图表
- HTML/CSS/JavaScript前端技术
- 响应式设计确保移动端兼容

请创造信息丰富且美观的可视化作品。"""


class ReporterAgent(BaseAgent):
    """报告生成器智能体"""

    def _get_system_prompt(self) -> str:
        return """你是一个专业的报告生成器智能体，专注于生成高质量的HTML分析报告。

你的专长：
1. 报告结构：设计清晰、逻辑严谨的报告结构
2. 内容整合：将分析结果、图表、结论整合为统一报告
3. 交互式元素：嵌入交互式图表和动态内容
4. 响应式设计：确保报告在各种设备上完美显示
5. 专业排版：使用专业的排版和设计原则

报告结构：
- 执行摘要：突出关键发现和结论
- 方法说明：清晰描述分析方法和过程
- 结果展示：以图表和表格形式展示结果
- 深入分析：详细的分析和解释
- 结论建议：基于数据的业务建议

设计原则：
- 用户体验：确保报告易于阅读和理解
- 视觉层次：使用清晰的视觉层次突出重要信息
- 一致性：保持整体风格和设计的一致性
- 可访问性：确保报告对所有人友好

请生成专业、美观、实用的HTML分析报告。"""


class QAAgent(BaseAgent):
    """质量保证智能体"""

    def _get_system_prompt(self) -> str:
        return """你是一个专业的质量保证智能体，负责验证分析结果的质量。

你的职责：
1. 结果验证：检查分析结果的准确性和合理性
2. 方法评估：验证分析方法的科学性和适用性
3. 一致性检查：确保各环节结果的一致性
4. 完整性检查：确认所有需求都得到满足
5. 问题识别：发现潜在的问题和风险

验证标准：
- 准确性：结果必须基于正确的数据和方法
- 可复现性：分析过程必须可复现
- 完整性：所有用户需求必须得到满足
- 一致性：不同部分的结果必须逻辑一致
- 实用性：结果必须对用户有实际价值

验证方法：
- 交叉验证：使用不同方法验证相同结果
- 敏感性分析：检查结果对假设的敏感性
- 边界测试：验证边界条件下的结果
- 合理性检查：基于领域知识评估结果合理性

请确保最终交付物的高质量和可靠性。"""


class MultiAgentSystem:
    """多智能体协作系统"""

    def __init__(
            self,
            model_client: ModelClient,
            tool_manager: ToolManager,
            conversation_id: str
    ):
        self.conversation_id = conversation_id
        self.model_client = model_client
        self.tool_manager = tool_manager

        # 创建智能体团队
        self.agents = {
            AgentRole.MANAGER: ManagerAgent(
                AgentRole.MANAGER, model_client, tool_manager, conversation_id
            ),
            AgentRole.ANALYST: AnalystAgent(
                AgentRole.ANALYST, model_client, tool_manager, conversation_id
            ),
            AgentRole.VISUALIZER: VisualizerAgent(
                AgentRole.VISUALIZER, model_client, tool_manager, conversation_id
            ),
            AgentRole.REPORTER: ReporterAgent(
                AgentRole.REPORTER, model_client, tool_manager, conversation_id
            ),
            AgentRole.QA: QAAgent(
                AgentRole.QA, model_client, tool_manager, conversation_id
            )
        }

        self.tasks: Dict[str, Task] = {}
        self.collaboration_log: List[CollaborationMessage] = []
        self.final_report: Optional[str] = None

    async def process_request(self, user_query: str) -> str:
        """处理用户请求"""
        logger.info(f"开始处理用户请求: {user_query}")

        try:
            # 1. 项目经理分析需求
            manager = self.agents[AgentRole.MANAGER]
            tasks = await manager.analyze_requirements(user_query)

            # 存储任务
            for task in tasks:
                self.tasks[task.id] = task

            # 2. 执行任务（考虑依赖关系）
            completed_tasks = await self._execute_tasks_with_dependencies(tasks)

            # 3. 生成最终报告
            self.final_report = await self._generate_final_report(completed_tasks)

            return self.final_report

        except Exception as e:
            logger.error(f"多智能体系统执行失败: {e}")
            return f"分析失败: {str(e)}"

    async def _execute_tasks_with_dependencies(self, tasks: List[Task]) -> List[Task]:
        """考虑依赖关系执行任务"""
        completed_tasks = []
        round_num = 1

        logger.info(f"开始执行 {len(tasks)} 个任务，考虑依赖关系")

        while tasks:
            logger.info(f"第 {round_num} 轮执行 - 剩余任务: {len(tasks)}")

            # 找到没有未完成依赖的任务
            ready_tasks = []
            for task in tasks:
                if all(dep in [t.id for t in completed_tasks] for dep in task.dependencies):
                    ready_tasks.append(task)

            if not ready_tasks:
                # 检查是否有循环依赖
                remaining_ids = [t.id for t in tasks]
                logger.error(f"检测到可能的循环依赖，剩余任务: {remaining_ids}")
                break

            logger.info(f"本轮可执行任务: {len(ready_tasks)} 个")
            for task in ready_tasks:
                logger.debug(f"可执行任务: {task.id} - {task.description}")

            # 并行执行就绪的任务
            tasks_to_execute = []
            for task in ready_tasks:
                tasks.remove(task)
                tasks_to_execute.append(task)

            # 执行任务
            execution_tasks = []
            for task in tasks_to_execute:
                agent = self.agents[task.agent_role]
                logger.info(f"分配任务 {task.id} 给 {task.agent_role.value}")
                execution_tasks.append(agent.process_task(task))

            results = await asyncio.gather(*execution_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"任务执行异常: {result}")
                else:
                    completed_tasks.append(result)
                    self.tasks[result.id] = result
                    logger.info(f"任务 {result.id} 完成 - 状态: {result.status}")

            round_num += 1

        logger.info(
            f"所有任务执行完成 - 成功: {len([t for t in completed_tasks if t.status == 'completed'])} 失败: {len([t for t in completed_tasks if t.status == 'failed'])}")
        return completed_tasks

    async def _generate_final_report(self, completed_tasks: List[Task]) -> str:
        """生成最终HTML报告"""
        reporter = self.agents[AgentRole.REPORTER]

        # 构建报告生成提示
        task_summaries = []
        for task in completed_tasks:
            if task.status == "completed":
                task_summaries.append(f"- {task.description}: {task.result[:200]}...")

        # 创建报告生成任务
        report_task = Task(
            id="final_report",
            description="生成最终HTML分析报告",
            agent_role=AgentRole.REPORTER
        )

        # 处理报告生成任务
        result = await reporter.process_task(report_task)

        if result.status == "completed":
            return result.result
        else:
            return f"报告生成失败: {result.error}"

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "conversation_id": self.conversation_id,
            "total_tasks": len(self.tasks),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == "completed"]),
            "failed_tasks": len([t for t in self.tasks.values() if t.status == "failed"]),
            "collaboration_messages": len(self.collaboration_log),
            "has_final_report": self.final_report is not None
        }

    async def close(self):
        """关闭系统资源"""
        pass


def create_agent_system(
        model_client: ModelClient,
        tool_manager: ToolManager,
        conversation_id: str
) -> MultiAgentSystem:
    """创建多智能体系统"""
    return MultiAgentSystem(model_client, tool_manager, conversation_id)
