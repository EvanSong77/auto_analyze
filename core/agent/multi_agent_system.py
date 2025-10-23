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
    REPORTER = "reporter"  # 报告生成器：HTML报告整合
    DATA_AGENT = "data_agent"  # 数据智能体：统一数据访问和代码执行


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
    analysis_results: Optional[List[Dict[str, Any]]] = None  # 用于存储完整的分析结果
    structured_summary: Optional[Dict[str, Any]] = None  # 结构化总结


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

            # 构建任务提示，包含analysis_results（如果存在）
            task_prompt = f"任务：{task.description}"
            
            # 如果有完整的分析结果，添加到提示中
            if hasattr(task, 'analysis_results') and task.analysis_results:
                analysis_info = f"\n\n## 完整分析数据\n系统提供了 {len(task.analysis_results)} 个分析任务的完整结果。"
                analysis_info += f"每个结果都包含完整的分析内容，请基于这些详细数据生成报告。"
                task_prompt += analysis_info

            self.messages.append({
                "role": "user",
                "content": task_prompt
            })

            logger.info(f"[{self.role.value}] 发送任务提示: {task_prompt[:100]}...")

            # 获取模型响应
            response = await self.model_client.chat_completion(
                messages=self.messages,
                tools=self.tool_manager.get_tool_schemas()
            )
            logger.info(f"[{self.role.value}] 完成了{task.description}: {response}")
            if response.get("status") == "error":
                logger.error(f"[{self.role.value}] 模型响应错误: {response.get('error')}")
                task.status = "failed"
                task.error = response.get("error")
            else:
                message = response["message"]
                self.messages.append(message)

                logger.info(f"[{self.role.value}] 收到模型响应: {message.get('content', '')[:200]}...")

                # 处理工具调用
                tool_execution_results = []  # 存储工具执行的实际结果
                if message.get("tool_calls"):
                    logger.info(f"[{self.role.value}] 检测到工具调用: {len(message['tool_calls'])} 个")

                    for i, tool_call in enumerate(message["tool_calls"]):
                        function_call = tool_call["function"]
                        tool_name = function_call["name"]
                        args = json.loads(function_call["arguments"])

                        logger.info(f"[{self.role.value}] 执行工具 {i + 1}: {tool_name} - 参数: {args}")

                        # 执行工具
                        result = await self.tool_manager.execute_tool(
                            tool_name=tool_name,
                            arguments=args,
                            context={"conversation_id": self.conversation_id}
                        )
                        logger.info(f"[{self.role.value}] 工具 {tool_name} 执行结果: 成功={result.success}")

                        # 保存工具执行的实际数据结果
                        if result.success and result.data:
                            tool_execution_results.append({
                                "tool_name": tool_name,
                                "arguments": args,
                                "execution_result": result.data,
                                "execution_time": result.execution_time
                            })

                        # 添加工具结果
                        self.messages.append({
                            "role": "tool",
                            "content": json.dumps(result.to_dict(), ensure_ascii=False),
                            "tool_call_id": tool_call["id"]
                        })

                # 构建包含实际数据的结果
                final_result = {
                    "model_response": message.get("content", ""),
                    "tool_execution_results": tool_execution_results,
                    "has_data_results": len(tool_execution_results) > 0
                }

                task.result = final_result
                task.status = "completed"
                task.completed_at = time.time()

                # 对任务结果进行结构化总结
                if task.agent_role in [AgentRole.ANALYST]:
                    try:
                        structured_summary = await self._create_structured_summary(task)
                        task.structured_summary = structured_summary
                        logger.info(f"[{self.role.value}] 任务结构化总结完成")
                    except Exception as e:
                        logger.warning(f"[{self.role.value}] 结构化总结失败: {e}")
                        task.structured_summary = {"error": str(e)}

                logger.info(f"[{self.role.value}] 任务完成: {task.description}")

            return task

        except Exception as e:
            logger.error(f"[{self.role.value}] 处理任务失败: {e}", exc_info=True)
            task.status = "failed"
            task.error = str(e)
            return task

    async def _create_structured_summary(self, task: Task) -> Dict[str, Any]:
        """使用模型对任务结果进行结构化总结"""
        summary_prompt = f"""请对以下分析任务的结果进行结构化总结：

任务描述: {task.description}
任务结果: {task.result}

请按照以下JSON格式返回总结：
{{
  "key_findings": ["发现1", "发现2", "发现3"],
  "data_insights": ["数据洞察1", "数据洞察2"],
  "business_implications": ["业务影响1", "业务影响2"],
  "recommendations": ["建议1", "建议2", "建议3"],
  "technical_details": ["技术细节1", "技术细节2"],
  "summary": "总体总结"
}}

请确保总结准确、完整，并基于实际分析结果。"""

        # 创建临时消息列表进行总结
        summary_messages = [
            {"role": "system", "content": "你是一个专业的数据分析总结专家，擅长从分析结果中提取关键信息并进行结构化总结。"},
            {"role": "user", "content": summary_prompt}
        ]

        response = await self.model_client.chat_completion(
            messages=summary_messages,
            tools=[]  # 总结不需要工具调用
        )

        if response.get("status") == "error":
            raise RuntimeError(f"总结生成失败: {response.get('error')}")

        content = response["message"]["content"]
        
        # 提取JSON格式的总结
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 如果无法解析JSON，返回原始内容
        return {"raw_summary": content}


class ManagerAgent(BaseAgent):
    """项目经理智能体"""

    def _get_system_prompt(self) -> str:
        return """你是一个专业的项目经理智能体，负责协调数据分析项目。

你的职责：
1. 理解用户的业务需求和目标
2. 将复杂任务分解为可执行的子任务
3. 分配任务给合适的团队成员（数据分析师、报告生成器、质量保证员）
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
- 与报告生成器合作：整合所有成果为专业报告
- 定期与质量保证团队沟通验证结果

请以项目管理者的角度思考和行动。"""

    async def analyze_requirements(self, user_goal: str) -> List[Task]:
        """分析用户需求并生成任务列表（新分工模式）"""
        logger.info(f"[manager] 开始分析用户需求: {user_goal}")

        prompt = f"""
用户目标: {user_goal}

请分析这个需求，并根据新的分工模式分解任务：

## 分工模式：
- **数据智能体 (data_agent)**：负责技术执行、数据处理、代码执行
- **数据分析师 (analyst)**：负责业务分析、洞察发现、建议生成  
- **报告生成器 (reporter)**：负责报告整合和展示

请以JSON格式返回任务列表，每个任务包含：
- description: 任务描述
- agent_role: 负责的智能体角色 (data_agent, analyst, reporter)
- dependencies: 依赖的前置任务ID列表

示例格式：
{{
  "tasks": [
    {{
      "description": "数据智能体：准备基础数据和初步分析",
      "agent_role": "data_agent",
      "dependencies": []
    }},
    {{
      "description": "数据分析师：基于数据结果进行业务分析",
      "agent_role": "analyst", 
      "dependencies": ["task_1"]
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
        logger.info(f"[manager] 需求分析响应: {content[:500]}...")

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
                logger.info(f"[manager] 任务: {task.id} - {task.description} - {task.agent_role.value}")

            return tasks

        except Exception as e:
            logger.error(f"[manager] 解析任务列表失败: {e}")
            # 返回默认任务列表（新分工模式）
            default_tasks = [
                Task(
                    id="task_1",
                    description="数据智能体：准备基础数据和初步技术分析",
                    agent_role=AgentRole.DATA_AGENT
                ),
                Task(
                    id="task_2",
                    description="数据分析师：基于技术结果进行业务分析和洞察发现",
                    agent_role=AgentRole.ANALYST,
                    dependencies=["task_1"]
                ),
                Task(
                    id="task_3",
                    description="报告生成器：整合分析结果生成专业报告",
                    agent_role=AgentRole.REPORTER,
                    dependencies=["task_2"]
                )
            ]
            logger.info(f"[manager] 使用默认任务列表: {len(default_tasks)} 个任务")
            return default_tasks


class AnalystAgent(BaseAgent):
    """数据分析师智能体 - 专注于业务分析和洞察发现"""

    def _get_system_prompt(self) -> str:
        return """你是一个专业的数据分析师智能体，专注于业务分析和洞察发现。

你的专长：
1. **业务理解**：深入理解业务需求和问题背景
2. **分析规划**：设计分析方案和实验设计
3. **洞察发现**：从数据中发现业务洞察和模式
4. **结果解读**：将技术结果转化为业务语言
5. **建议生成**：基于分析结果提出可执行建议

工作原则：
- 业务导向：始终围绕业务价值进行分析
- 问题驱动：聚焦解决具体的业务问题
- 可解释性：确保分析结果对业务人员可理解
- 行动导向：提供具体的行动建议

工作流程：
1. 向数据智能体请求所需的数据和分析结果
2. 基于数据智能体提供的结果进行业务分析
3. 将技术发现转化为业务洞察
4. 生成基于数据的业务建议

请专注于业务分析和洞察发现，数据获取和处理由数据智能体负责。"""


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


class DataAgent(BaseAgent):
    """数据智能体 - 统一数据基础设施和工具执行"""

    def _get_system_prompt(self) -> str:
        return """你是一个专业的数据智能体，负责统一管理数据基础设施和工具执行。

核心职责：
1. **数据基础设施**：管理数据文件访问、存储和组织
2. **代码执行引擎**：执行所有技术性代码（数据处理、计算、可视化）
3. **工具协调**：管理和执行所有技术工具调用
4. **数据质量保证**：确保数据处理的准确性和一致性

具体分工：
- **数据获取**：读取文件、数据库连接、API调用
- **数据处理**：数据清洗、转换、聚合等基础操作
- **技术执行**：统计分析、机器学习、可视化生成
- **结果标准化**：将技术结果整理为标准格式

协作模式：
- 数据分析师专注于业务分析和洞察发现
- 数据智能体专注于技术执行和数据管理
- 数据分析师向数据智能体请求技术执行服务
- 数据智能体返回标准化的技术结果

请专注于技术执行和数据管理，确保高效可靠的数据处理。"""


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
            AgentRole.REPORTER: ReporterAgent(
                AgentRole.REPORTER, model_client, tool_manager, conversation_id
            ),
            AgentRole.DATA_AGENT: DataAgent(
                AgentRole.DATA_AGENT, model_client, tool_manager, conversation_id
            )
        }

        self.tasks: Dict[str, Task] = {}
        self.collaboration_log: List[CollaborationMessage] = []
        self.final_report: Optional[str] = None

    async def process_request(self, user_query: str) -> str:
        """处理用户请求（支持动态任务规划）"""
        logger.info(f"开始处理用户请求: {user_query}")

        try:
            # 1. 项目经理分析需求并生成初始任务
            manager = self.agents[AgentRole.MANAGER]
            initial_tasks = await manager.analyze_requirements(user_query)

            # 存储初始任务
            for task in initial_tasks:
                self.tasks[task.id] = task

            # 2. 动态执行任务（根据执行结果迭代规划）
            completed_tasks = await self._execute_tasks_dynamically(initial_tasks)

            # 3. 生成最终报告
            self.final_report = await self._generate_final_report(completed_tasks)

            return self.final_report

        except Exception as e:
            logger.error(f"多智能体系统执行失败: {e}")
            return f"分析失败: {str(e)}"

    async def _execute_tasks_dynamically(self, initial_tasks: List[Task]) -> List[Task]:
        """动态执行任务（根据执行结果迭代规划）"""
        completed_tasks = []
        pending_tasks = initial_tasks.copy()
        iteration = 1
        max_iterations = 5  # 防止无限循环

        logger.info(f"开始动态任务执行，初始任务: {len(initial_tasks)} 个")

        while pending_tasks and iteration <= max_iterations:
            logger.info(f"第 {iteration} 轮迭代 - 待处理任务: {len(pending_tasks)}")

            # 1. 执行当前轮次的任务
            round_completed = await self._execute_round_tasks(pending_tasks)
            
            # 2. 将完成的任务移到完成列表
            for task in round_completed:
                if task.status == "completed":
                    completed_tasks.append(task)
                    if task in pending_tasks:
                        pending_tasks.remove(task)

            # 3. 并行进行结构化总结
            summary_tasks = []
            for task in round_completed:
                if (task.status == "completed" and 
                    task.agent_role in [AgentRole.ANALYST] and
                    hasattr(task, 'structured_summary') and task.structured_summary is None):
                    summary_tasks.append(self._create_structured_summary_parallel(task))
            
            if summary_tasks:
                await asyncio.gather(*summary_tasks, return_exceptions=True)

            # 4. 基于执行结果生成新任务
            new_tasks = await self._generate_new_tasks_based_on_results(completed_tasks)
            
            if new_tasks:
                logger.info(f"基于执行结果生成 {len(new_tasks)} 个新任务")
                pending_tasks.extend(new_tasks)
                for task in new_tasks:
                    self.tasks[task.id] = task

            # 5. 检查是否满足终止条件
            if not new_tasks and not pending_tasks:
                logger.info("没有新任务生成，执行完成")
                break

            iteration += 1

        logger.info(f"动态执行完成 - 迭代次数: {iteration-1}, 完成任务: {len(completed_tasks)}")
        return completed_tasks

    async def _execute_round_tasks(self, tasks: List[Task]) -> List[Task]:
        """执行一轮任务"""
        completed_tasks = []
        
        # 并行执行任务
        execution_tasks = []
        for task in tasks:
            agent = self.agents[task.agent_role]
            logger.info(f"分配任务 {task.id} 给 {task.agent_role.value}")
            execution_tasks.append(agent.process_task(task))

        results = await asyncio.gather(*execution_tasks, return_exceptions=True)

        # 处理执行结果
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"任务执行异常: {result}")
            else:
                completed_tasks.append(result)
                self.tasks[result.id] = result
                logger.info(f"任务 {result.id} 完成 - 状态: {result.status}")

        return completed_tasks

    async def _generate_new_tasks_based_on_results(self, completed_tasks: List[Task]) -> List[Task]:
        """基于执行结果生成新任务"""
        new_tasks = []
        manager = self.agents[AgentRole.MANAGER]
        
        # 分析已完成任务的结果
        for task in completed_tasks:
            if task.status == "completed" and task.agent_role == AgentRole.ANALYST:
                # 分析结果是否需要进一步处理
                if self._needs_further_analysis(task):
                    new_task = Task(
                        id=f"deep_analysis_{task.id}",
                        description=f"深入分析：{task.description}",
                        agent_role=AgentRole.ANALYST,
                        dependencies=[task.id]
                    )
                    new_tasks.append(new_task)
                    logger.info(f"生成深入分析任务: {new_task.description}")
                
                # 检查是否需要数据验证
                if self._needs_data_validation(task):
                    validation_task = Task(
                        id=f"validation_{task.id}",
                        description=f"数据验证：{task.description}",
                        agent_role=AgentRole.DATA_AGENT,
                        dependencies=[task.id]
                    )
                    new_tasks.append(validation_task)
                    logger.info(f"生成数据验证任务: {validation_task.description}")

        return new_tasks

    def _needs_further_analysis(self, task: Task) -> bool:
        """判断是否需要进一步分析"""
        if not task.result:
            return False
        
        result_str = str(task.result).lower()
        
        # 如果结果包含需要深入分析的指示
        indicators = [
            '需要进一步分析', '深入分析', '更多数据', '复杂模式', 
            '异常发现', '需要验证', '不确定', '可能', '潜在'
        ]
        
        return any(indicator in result_str for indicator in indicators)

    def _needs_data_validation(self, task: Task) -> bool:
        """判断是否需要数据验证"""
        if not task.result:
            return False
        
        result_str = str(task.result).lower()
        
        # 如果结果包含数据质量问题
        issues = [
            '数据质量', '异常值', '缺失值', '不一致', '可疑',
            '需要验证', '准确性', '可靠性'
        ]
        
        return any(issue in result_str for issue in issues)

    async def _generate_final_report(self, completed_tasks: List[Task]) -> str:
        """生成最终HTML报告"""
        reporter = self.agents[AgentRole.REPORTER]

        # 收集完整的分析结果和结构化总结
        analysis_results = []
        structured_summaries = []
        
        for task in completed_tasks:
            if task.status == "completed":
                analysis_result = {
                    "task_id": task.id,
                    "description": task.description,
                    "result": task.result,  # 保留完整结果
                    "agent_role": task.agent_role.value,
                    "completion_time": task.completed_at
                }
                
                # 如果有结构化总结，添加到结果中
                if hasattr(task, 'structured_summary') and task.structured_summary:
                    analysis_result["structured_summary"] = task.structured_summary
                    structured_summaries.append({
                        "task_id": task.id,
                        "description": task.description,
                        "summary": task.structured_summary
                    })
                
                analysis_results.append(analysis_result)

        # 使用结构化总结构建任务描述
        enhanced_description = self._build_report_description(analysis_results, structured_summaries)

        # 创建报告生成任务，并传递完整分析结果
        report_task = Task(
            id="final_report",
            description=enhanced_description,
            agent_role=AgentRole.REPORTER
        )

        # 将完整分析结果存储在任务中供报告生成器使用
        report_task.analysis_results = analysis_results

        # 处理报告生成任务
        result = await reporter.process_task(report_task)

        if result.status == "completed":
            return result.result
        else:
            return f"报告生成失败: {result.error}"

    # 结构化总结已移至BaseAgent类中，使用模型进行智能总结

    async def _create_structured_summary(self, task: Task) -> Dict[str, Any]:
        """使用模型对任务结果进行结构化总结"""
        summary_prompt = f"""请对以下分析任务的结果进行结构化总结：

任务描述: {task.description}
任务结果: {task.result}

请按照以下JSON格式返回总结：
{{
  "key_findings": ["发现1", "发现2", "发现3"],
  "data_insights": ["数据洞察1", "数据洞察2"],
  "business_implications": ["业务影响1", "业务影响2"],
  "recommendations": ["建议1", "建议2", "建议3"],
  "technical_details": ["技术细节1", "技术细节2"],
  "summary": "总体总结"
}}

请确保总结准确、完整，并基于实际分析结果。"""

        # 创建临时消息列表进行总结
        summary_messages = [
            {"role": "system", "content": "你是一个专业的数据分析总结专家，擅长从分析结果中提取关键信息并进行结构化总结。"},
            {"role": "user", "content": summary_prompt}
        ]

        response = await self.model_client.chat_completion(
            messages=summary_messages,
            tools=[]  # 总结不需要工具调用
        )

        if response.get("status") == "error":
            raise RuntimeError(f"总结生成失败: {response.get('error')}")

        content = response["message"]["content"]
        
        # 提取JSON格式的总结
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 如果无法解析JSON，返回原始内容
        return {"raw_summary": content}

    async def _create_structured_summary_parallel(self, task: Task):
        """并行执行结构化总结"""
        try:
            agent = self.agents[task.agent_role]
            if hasattr(agent, '_create_structured_summary'):
                structured_summary = await agent._create_structured_summary(task)
                task.structured_summary = structured_summary
                logger.info(f"并行总结完成: {task.id}")
        except Exception as e:
            logger.warning(f"并行总结失败 {task.id}: {e}")
            task.structured_summary = {"error": str(e)}

    def _build_report_description(self, analysis_results: List[Dict], structured_summaries: List[Dict]) -> str:
        """基于结构化总结构建报告描述"""
        description = f"""生成最终HTML分析报告

## 分析任务概览
已完成 {len(analysis_results)} 个分析任务

## 结构化总结概览"""

        if structured_summaries:
            description += f"\n已完成 {len(structured_summaries)} 个任务的结构化总结：\n"
            
            for summary in structured_summaries:
                desc = summary['description']
                structured = summary['summary']
                
                if isinstance(structured, dict):
                    description += f"\n### {desc}\n"
                    
                    if 'key_findings' in structured and structured['key_findings']:
                        description += f"**关键发现:**\n"
                        for finding in structured['key_findings'][:3]:  # 限制显示数量
                            description += f"- {finding}\n"
                    
                    if 'summary' in structured and structured['summary']:
                        description += f"**总体总结:** {structured['summary']}\n"
                else:
                    description += f"\n### {desc}\n{str(structured)[:200]}...\n"
        else:
            description += "\n暂无结构化总结数据，将使用原始分析结果。\n"

        description += """

## 完整分析数据
所有分析任务的完整结果已收集，请基于结构化总结和原始数据生成专业的HTML报告。

请确保报告包含：
1. 基于结构化总结的关键发现和洞察
2. 数据分析的可视化展示
3. 业务建议和行动计划
4. 技术实现细节"""

        return description

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
