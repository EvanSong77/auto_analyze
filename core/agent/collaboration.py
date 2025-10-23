# -*- coding: utf-8 -*-
# @Time    : 2025/10/15
# @Author  : EvanSong
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from core.agent.multi_agent_system import AgentRole, CollaborationMessage
from utils.logger import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """消息类型枚举"""
    TASK_REQUEST = "task_request"  # 任务请求
    TASK_RESULT = "task_result"  # 任务结果
    DATA_SHARING = "data_sharing"  # 数据共享
    COLLABORATION_REQUEST = "collaboration_request"  # 协作请求
    FEEDBACK = "feedback"  # 反馈
    NOTIFICATION = "notification"  # 通知


@dataclass
class SharedData:
    """共享数据结构"""
    id: str
    data_type: str  # dataframe, visualization, analysis_result, etc.
    content: Any
    created_by: AgentRole
    created_at: float = field(default_factory=time.time)
    accessed_by: Set[AgentRole] = field(default_factory=set)


class CollaborationManager:
    """协作管理器"""

    def __init__(self):
        self.message_queue: asyncio.Queue[CollaborationMessage] = asyncio.Queue()
        self.shared_data: Dict[str, SharedData] = {}
        self.message_handlers: Dict[MessageType, List[callable]] = {}
        self.conversation_log: List[CollaborationMessage] = []

        # 注册默认消息处理器
        self._register_default_handlers()

    def _register_default_handlers(self):
        """注册默认消息处理器"""
        self.register_handler(MessageType.TASK_REQUEST, self._handle_task_request)
        self.register_handler(MessageType.TASK_RESULT, self._handle_task_result)
        self.register_handler(MessageType.DATA_SHARING, self._handle_data_sharing)
        self.register_handler(MessageType.FEEDBACK, self._handle_feedback)
        self.register_handler(MessageType.NOTIFICATION, self._handle_notification)
        self.register_handler(MessageType.COLLABORATION_REQUEST, self._handle_collaboration_request)

    def register_handler(self, message_type: MessageType, handler: callable):
        """注册消息处理器"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)

    async def send_message(self, message: CollaborationMessage):
        """发送消息"""
        logger.info(f"发送消息: {message.sender.value} -> {message.receiver.value}")
        self.conversation_log.append(message)
        await self.message_queue.put(message)

    async def process_messages(self):
        """处理消息队列"""
        while True:
            try:
                message = await self.message_queue.get()
                await self._process_single_message(message)
                self.message_queue.task_done()
            except Exception as e:
                logger.error(f"消息处理错误: {e}")

    async def _process_single_message(self, message: CollaborationMessage):
        """处理单个消息"""
        try:
            # 根据消息类型调用相应的处理器
            if message.message_type in self.message_handlers:
                handlers = self.message_handlers[message.message_type]
                for handler in handlers:
                    await handler(message)

        except Exception as e:
            logger.error(f"处理消息失败: {e}")

    @staticmethod
    async def _handle_task_request(message: CollaborationMessage):
        """处理任务请求"""
        logger.info(f"处理任务请求: {message.content}")

        # 这里可以添加任务分配逻辑
        # 例如：根据任务类型分配给合适的智能体

    @staticmethod
    async def _handle_task_result(message: CollaborationMessage):
        """处理任务结果"""
        logger.info(f"收到任务结果: {message.content[:100]}...")

        # 这里可以添加结果验证和后续处理逻辑

    async def _handle_data_sharing(self, message: CollaborationMessage):
        """处理数据共享"""
        try:
            data_content = json.loads(message.content)
            data_id = data_content.get("id")

            if data_id:
                shared_data = SharedData(
                    id=data_id,
                    data_type=data_content.get("type", "unknown"),
                    content=data_content.get("content"),
                    created_by=message.sender
                )
                self.shared_data[data_id] = shared_data
                logger.info(f"数据共享成功: {data_id}")

        except Exception as e:
            logger.error(f"处理数据共享失败: {e}")

    @staticmethod
    async def _handle_feedback(message: CollaborationMessage):
        """处理反馈消息"""
        logger.info(f"收到反馈: {message.sender.value} -> {message.receiver.value}")

        # 这里可以添加反馈处理逻辑

    @staticmethod
    async def _handle_notification(message: CollaborationMessage):
        """处理通知消息"""
        logger.info(f"收到通知: {message.sender.value} -> {message.receiver.value}: {message.content[:100]}")

        # 这里可以添加通知处理逻辑

    @staticmethod
    async def _handle_collaboration_request(message: CollaborationMessage):
        """处理协作请求消息"""
        logger.info(f"收到协作请求: {message.sender.value} -> {message.receiver.value}: {message.content[:100]}")

        # 这里可以添加协作请求处理逻辑

    def get_shared_data(self, data_id: str) -> Optional[SharedData]:
        """获取共享数据"""
        return self.shared_data.get(data_id)

    def get_all_shared_data(self) -> List[SharedData]:
        """获取所有共享数据"""
        return list(self.shared_data.values())

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return [
            {
                "timestamp": msg.timestamp,
                "sender": msg.sender.value,
                "receiver": msg.receiver.value,
                "type": msg.message_type,
                "content": str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content)
            }
            for msg in self.conversation_log
        ]


class CollaborationProtocol:
    """协作协议 - 定义智能体间的标准交互模式"""

    def __init__(self, collaboration_manager: CollaborationManager):
        self.collaboration_manager = collaboration_manager

    async def request_analysis(self, requester: AgentRole, analyst: AgentRole,
                               data_description: str, analysis_goal: str) -> str:
        """请求数据分析"""
        message = CollaborationMessage(
            sender=requester,
            receiver=analyst,
            message_type=MessageType.TASK_REQUEST.value,
            content=json.dumps({
                "action": "analyze_data",
                "data_description": data_description,
                "analysis_goal": analysis_goal,
                "priority": "normal"
            }, ensure_ascii=False)
        )

        await self.collaboration_manager.send_message(message)
        return f"分析请求已发送给 {analyst.value}"

    async def share_visualization(self, sharer: AgentRole, receiver: AgentRole,
                                  chart_data: Dict[str, Any], description: str) -> str:
        """共享可视化结果"""
        data_id = f"viz_{int(time.time())}"

        message = CollaborationMessage(
            sender=sharer,
            receiver=receiver,
            message_type=MessageType.DATA_SHARING.value,
            content=json.dumps({
                "id": data_id,
                "type": "visualization",
                "content": chart_data,
                "description": description
            }, ensure_ascii=False)
        )

        await self.collaboration_manager.send_message(message)
        return f"可视化结果已共享 ({data_id})"

    async def data_processing_request(self, requester: AgentRole, data_agent: AgentRole,
                                      analysis_description: str, processing_needs: List[str]) -> str:
        """请求数据处理服务"""
        message = CollaborationMessage(
            sender=requester,
            receiver=data_agent,
            message_type=MessageType.TASK_REQUEST.value,
            content=json.dumps({
                "action": "data_processing",
                "analysis_description": analysis_description,
                "processing_needs": processing_needs,
                "priority": "normal"
            }, ensure_ascii=False)
        )

        await self.collaboration_manager.send_message(message)
        return f"数据处理请求已发送给 {data_agent.value}"

    async def request_feedback(self, requester: AgentRole, reviewer: AgentRole,
                               work_description: str, specific_questions: List[str]) -> str:
        """请求反馈"""
        message = CollaborationMessage(
            sender=requester,
            receiver=reviewer,
            message_type=MessageType.FEEDBACK.value,
            content=json.dumps({
                "work_description": work_description,
                "questions": specific_questions,
                "urgency": "normal"
            }, ensure_ascii=False)
        )

        await self.collaboration_manager.send_message(message)
        return f"反馈请求已发送给 {reviewer.value}"

    async def notify_completion(self, notifier: AgentRole, recipients: List[AgentRole],
                                task_description: str, result_summary: str) -> str:
        """通知任务完成"""
        completion_messages = []

        for recipient in recipients:
            message = CollaborationMessage(
                sender=notifier,
                receiver=recipient,
                message_type=MessageType.NOTIFICATION.value,
                content=json.dumps({
                    "notification_type": "task_completion",
                    "task": task_description,
                    "result": result_summary,
                    "timestamp": time.time()
                }, ensure_ascii=False)
            )
            completion_messages.append(message)

        # 批量发送通知
        for message in completion_messages:
            await self.collaboration_manager.send_message(message)

        return f"完成通知已发送给 {len(recipients)} 个智能体"


class EnhancedMultiAgentSystem:
    """增强的多智能体系统（包含协作功能）"""

    def __init__(self, base_system):
        self.base_system = base_system
        self.collaboration_manager = CollaborationManager()
        self.collaboration_protocol = CollaborationProtocol(self.collaboration_manager)

        # 启动消息处理循环
        self.message_processing_task = asyncio.create_task(
            self.collaboration_manager.process_messages()
        )

    async def enhanced_process_request(self, user_query: str) -> str:
        """增强的处理请求方法"""
        logger.info(f"使用增强系统处理请求: {user_query}")

        try:
            # 1. 项目经理分析需求（带协作）
            manager = self.base_system.agents[AgentRole.MANAGER]

            # 通知团队开始新项目
            await self.collaboration_protocol.notify_completion(
                AgentRole.MANAGER,
                [AgentRole.ANALYST, AgentRole.REPORTER, AgentRole.DATA_AGENT],
                "项目启动",
                f"新项目开始：{user_query}"
            )

            tasks = await manager.analyze_requirements(user_query)

            # 存储任务
            for task in tasks:
                self.base_system.tasks[task.id] = task

            # 2. 执行任务（带协作监控）
            completed_tasks = await self._execute_tasks_with_collaboration(tasks)

            # 3. 质量保证检查
            await self._perform_quality_assurance(completed_tasks)

            # 4. 生成最终报告
            final_report = await self._generate_enhanced_report(completed_tasks)

            # 通知项目完成
            await self.collaboration_protocol.notify_completion(
                AgentRole.MANAGER,
                [AgentRole.ANALYST, AgentRole.REPORTER, AgentRole.DATA_AGENT],
                "项目完成",
                "所有分析任务已完成，报告已生成"
            )

            return final_report

        except Exception as e:
            logger.error(f"增强系统执行失败: {e}")
            return f"分析失败: {str(e)}"

    async def _execute_tasks_with_collaboration(self, tasks: List) -> List:
        """带协作监控的任务执行"""
        completed_tasks = []

        for task in tasks:
            # 通知任务开始
            await self.collaboration_protocol.notify_completion(
                AgentRole.MANAGER,
                [task.agent_role],
                "任务分配",
                f"新任务：{task.description}"
            )

            # 执行任务
            agent = self.base_system.agents[task.agent_role]
            result = await agent.process_task(task)

            if result.status == "completed":
                # 数据分析师完成分析后，可能请求进一步的数据处理
                if task.agent_role in [AgentRole.ANALYST]:
                    await self.collaboration_protocol.data_processing_request(
                        task.agent_role,
                        AgentRole.DATA_AGENT,
                        f"基于分析结果的数据处理：{task.description}",
                        ["生成可视化图表", "执行深度计算", "准备报告数据"]
                    )

            completed_tasks.append(result)

        return completed_tasks

    async def _perform_quality_assurance(self, completed_tasks: List):
        """执行数据质量检查（由数据智能体处理）"""
        data_agent = self.base_system.agents[AgentRole.DATA_AGENT]

        for task in completed_tasks:
            if task.status == "completed":
                # 创建数据验证任务
                validation_task = type('Task', (), {
                    'id': f"validation_{task.id}",
                    'description': f"验证数据质量：{task.description}",
                    'agent_role': AgentRole.DATA_AGENT,
                    'result': None,
                    'status': 'pending'
                })()

                # 执行数据验证
                await data_agent.process_task(validation_task)

    async def _generate_enhanced_report(self, completed_tasks: List) -> str:
        """生成增强的报告"""
        reporter = self.base_system.agents[AgentRole.REPORTER]

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
                    "completion_time": task.completed_at if hasattr(task, 'completed_at') else None
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
        enhanced_description = self._build_collaboration_description(analysis_results, structured_summaries)

        # 创建报告生成任务
        report_task = type('Task', (), {
            'id': "enhanced_report",
            'description': enhanced_description,
            'agent_role': AgentRole.REPORTER,
            'result': None,
            'status': 'pending',
            'analysis_results': analysis_results  # 传递完整分析结果
        })()

        # 处理报告生成
        result = await reporter.process_task(report_task)

        if result.status == "completed":
            return result.result
        else:
            return f"报告生成失败: {result.error}"

    def _summarize_analysis_result(self, result: str) -> str:
        """智能摘要分析结果，保留关键信息"""
        if len(result) <= 500:
            return result
        
        # 提取关键信息：数字、百分比、重要结论等
        import re
        
        # 安全处理结果，确保是字符串
        result_str = str(result)
        
        # 提取数字和关键指标
        numbers = re.findall(r'\d+(?:\.\d+)?%?', result_str[:1000])
        
        # 提取结论性语句
        conclusions = []
        conclusion_patterns = [
            r'结论[：:](.*?)(?:\.|$)',
            r'发现[：:](.*?)(?:\.|$)', 
            r'建议[：:](.*?)(?:\.|$)',
            r'因此[，,](.*?)(?:\.|$)'
        ]
        
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, result_str[:1000])
            conclusions.extend(matches)
        
        # 构建智能摘要
        summary_parts = []
        
        if numbers:
            summary_parts.append(f"关键数值: {'、'.join(numbers[:5])}")
        
        if conclusions:
            summary_parts.append(f"主要发现: {'; '.join(conclusions[:3])}")
        
        if not summary_parts:
            # 如果无法提取关键信息，使用前300字符
            return result_str[:300] + "..." if len(result_str) > 300 else result_str
        
        return "; ".join(summary_parts)

    def _build_collaboration_description(self, analysis_results: List[Dict], structured_summaries: List[Dict]) -> str:
        """基于结构化总结构建协作报告描述"""
        description = f"""生成协作增强的HTML分析报告

## 协作分析任务概览
已完成 {len(analysis_results)} 个协作分析任务

## 结构化协作总结概览"""

        if structured_summaries:
            description += f"\n已完成 {len(structured_summaries)} 个协作任务的结构化总结：\n"
            
            for summary in structured_summaries:
                desc = summary['description']
                structured = summary['summary']
                
                if isinstance(structured, dict):
                    description += f"\n### {desc}\n"
                    
                    if 'key_findings' in structured and structured['key_findings']:
                        description += f"**协作发现:**\n"
                        for finding in structured['key_findings'][:3]:
                            description += f"- {finding}\n"
                    
                    if 'business_implications' in structured and structured['business_implications']:
                        description += f"**业务影响:**\n"
                        for implication in structured['business_implications'][:2]:
                            description += f"- {implication}\n"
                    
                    if 'summary' in structured and structured['summary']:
                        description += f"**协作总结:** {structured['summary']}\n"
                else:
                    description += f"\n### {desc}\n{str(structured)[:200]}...\n"
        else:
            description += "\n暂无协作结构化总结数据，将使用原始分析结果。\n"

        description += """

## 完整协作分析数据
所有协作分析任务的完整结果已收集，请基于结构化总结生成专业的HTML报告。

请确保报告包含：
1. 基于结构化总结的协作关键发现
2. 多智能体协作的成果展示
3. 数据洞察和业务建议
4. 协作过程的亮点总结"""

        return description

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """获取协作统计信息"""
        return {
            "total_messages": len(self.collaboration_manager.conversation_log),
            "shared_data_count": len(self.collaboration_manager.shared_data),
            "message_types": {
                msg_type.value: len([m for m in self.collaboration_manager.conversation_log
                                     if m.message_type == msg_type.value])
                for msg_type in MessageType
            }
        }

    async def close(self):
        """关闭系统"""
        self.message_processing_task.cancel()
        try:
            await self.message_processing_task
        except asyncio.CancelledError:
            pass
