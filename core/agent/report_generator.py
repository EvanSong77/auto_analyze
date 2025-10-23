# -*- coding: utf-8 -*-
# @Time    : 2025/10/15
# @Author  : EvanSong
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from core.agent.multi_agent_system import BaseAgent, AgentRole
from core.model_client import ModelClient
from core.tool_manager import ToolManager
from utils.logger import get_logger

logger = get_logger(__name__)


class ReportGeneratorAgent(BaseAgent):
    """HTML报告生成器智能体"""

    def __init__(
            self,
            model_client: ModelClient,
            tool_manager: ToolManager,
            conversation_id: str
    ):
        super().__init__(AgentRole.REPORTER, model_client, tool_manager, conversation_id)
        # 覆盖系统提示词
        self.system_prompt = self._get_system_prompt()
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    @staticmethod
    def _get_system_prompt() -> str:
        return """你是一个专业的HTML报告生成器智能体，专注于创建高质量的数据分析报告。

## 核心能力
1. **报告结构设计**：创建清晰、逻辑严谨的报告结构
2. **内容整合**：将分析结果、图表、结论整合为统一报告
3. **交互式元素**：嵌入ECharts交互式图表和动态内容
4. **响应式设计**：确保报告在各种设备上完美显示
5. **专业排版**：使用现代Web设计原则和最佳实践

## 重要说明
当任务描述中包含分析任务结果时，这些结果是前面多个智能体分析的实际结果。系统会提供：
1. **任务结果摘要**：每个任务的简要概述
2. **完整分析数据**：所有分析任务的完整结果（可通过任务对象的analysis_results属性访问）

你需要：
1. **深入分析完整数据**：访问analysis_results获取详细的分析结果
2. **提取关键洞察**：从完整数据中提取重要的发现和结论
3. **整合内容**：将不同任务的结果有机整合到报告中
4. **保持数据完整性**：确保报告基于完整的分析数据，而非简化的摘要

## 数据访问方法
- 通过任务对象的analysis_results属性访问完整分析结果
- 每个分析结果包含：任务描述、完整结果、执行时间等信息
- 优先使用完整数据生成报告，摘要仅用于概览

## 报告结构模板
- **标题页**：醒目的标题、项目概述、关键指标
- **执行摘要**：突出关键发现和业务影响，基于完整分析数据
- **分析背景**：项目目标、数据来源、分析方法
- **详细分析**：分章节展示各个任务的完整分析过程和结果
- **可视化展示**：交互式图表和图形化结果
- **结论建议**：基于所有完整分析数据的洞察和可执行建议
- **附录**：技术细节、数据字典、参考资料

## 设计原则
- **数据完整性**：确保报告反映完整的分析结果
- **用户体验优先**：确保报告易于阅读和理解
- **视觉层次清晰**：使用清晰的视觉层次突出重要信息
- **一致性**：保持整体风格和设计的一致性
- **可访问性**：确保报告对所有用户友好

## 技术规范
- 使用ECharts进行交互式数据可视化
- 采用Bootstrap框架确保响应式设计
- 使用现代CSS技术（Flexbox/Grid）
- 确保HTML5语义化标记
- 优化图片和资源加载

## 交互功能要求
- 图表支持缩放、筛选、悬停提示
- 响应式导航和目录
- 数据表格排序和搜索
- 深色/浅色主题切换
- 打印优化版本

## 代码生成规范
- 生成完整的、可运行的HTML代码
- 包含必要的CSS和JavaScript
- 使用CDN引入外部库（ECharts、Bootstrap）
- 确保代码格式化和注释清晰
- 测试生成的HTML在不同浏览器中的兼容性

请基于完整的分析结果数据生成专业、美观、实用的HTML分析报告。"""

    async def generate_report(
            self,
            user_query: str,
            analysis_results: List[Dict[str, Any]],
            visualization_data: Optional[List[Dict[str, Any]]] = None,
            report_title: Optional[str] = None
    ) -> str:
        """生成HTML报告
        
        Args:
            user_query: 用户原始查询
            analysis_results: 分析结果列表
            visualization_data: 可视化数据
            report_title: 报告标题
            
        Returns:
            HTML报告内容
        """

        # 构建报告生成提示
        prompt = self._build_report_prompt(
            user_query, analysis_results, visualization_data, report_title
        )

        self.messages.append({"role": "user", "content": prompt})

        # 获取模型响应
        response = await self.model_client.chat_completion(
            messages=self.messages,
            tools=[]  # 报告生成不需要工具调用
        )

        if response.get("status") == "error":
            error_msg = f"报告生成失败: {response.get('error')}"
            logger.error(error_msg)
            return self._generate_fallback_report(user_query, analysis_results, error_msg)

        content = response["message"]["content"]

        # 提取HTML代码
        html_content = self._extract_html_content(content)

        if html_content:
            # 验证HTML格式
            validated_html = self._validate_and_fix_html(html_content)
            return validated_html
        else:
            # 如果没有找到HTML，生成默认报告
            return self._generate_fallback_report(user_query, analysis_results, content)

    @staticmethod
    def _build_report_prompt(
            user_query: str,
            analysis_results: List[Dict[str, Any]],
            visualization_data: Optional[List[Dict[str, Any]]],
            report_title: Optional[str]
    ) -> str:
        """构建报告生成提示"""

        # 格式化分析结果
        analysis_summary = ""
        for i, result in enumerate(analysis_results, 1):
            result_content = result.get('result', '无结果')
            # 安全处理结果内容，确保是字符串
            if isinstance(result_content, dict):
                # 如果是字典，转换为字符串表示
                result_str = str(result_content)[:500] + "..." if len(str(result_content)) > 500 else str(result_content)
            else:
                result_str = str(result_content)[:500] + "..." if len(str(result_content)) > 500 else str(result_content)
            
            analysis_summary += f"\n{i}. {result.get('task', '未知任务')}:\n"
            analysis_summary += f"   结果: {result_str}\n"

        # 格式化可视化数据
        viz_summary = ""
        if visualization_data:
            for i, viz in enumerate(visualization_data, 1):
                viz_summary += f"\n{i}. {viz.get('type', '未知类型')}: {viz.get('description', '无描述')}"

        prompt = f"""
# 报告生成请求

## 项目背景
- **用户需求**: {user_query}
- **报告标题**: {report_title or '数据分析报告'}

## 分析结果摘要
{analysis_summary}

## 可视化内容
{viz_summary or '无可视化数据'}

## 报告要求
请基于以上分析结果，生成一个完整的、专业的HTML分析报告。报告应包含：

### 必须包含的部分
1. **专业标题页**：包含项目名称、日期、关键指标概览
2. **执行摘要**：突出显示最重要的发现和结论
3. **详细分析章节**：
   - 分析方法说明
   - 数据质量评估
   - 关键发现展示
   - 统计结果呈现
4. **交互式可视化**：使用ECharts创建可交互的图表
5. **结论与建议**：基于数据的业务洞察
6. **响应式设计**：确保在手机、平板、桌面设备上都能良好显示

### 技术要求
- 使用Bootstrap 5框架确保响应式设计
- 集成ECharts进行数据可视化
- 包含完整的CSS样式和JavaScript代码
- 确保HTML语义化和可访问性
- 优化加载性能

### 设计规范
- 现代、专业的视觉风格
- 清晰的视觉层次结构
- 一致的颜色方案和字体read_directory
- 良好的可读性和用户体验

请直接生成完整的HTML代码，确保代码可以直接在浏览器中运行。
"""

        return prompt

    @staticmethod
    def _extract_html_content(content: str) -> Optional[str]:
        """从模型响应中提取HTML内容"""

        # 尝试提取HTML代码块
        html_pattern = r'```html\n(.*?)\n```'
        html_matches = re.findall(html_pattern, content, re.DOTALL)

        if html_matches:
            return html_matches[0]

        # 如果没有代码块标记，尝试提取整个HTML文档
        if '<!DOCTYPE html>' in content or '<html' in content:
            # 提取从<!DOCTYPE>或<html>开始的内容
            start_pattern = r'(<!DOCTYPE html>.*?|.*?<html.*?>)'
            match = re.search(start_pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                start_index = match.start()
                return content[start_index:]

        return None

    @staticmethod
    def _validate_and_fix_html(html_content: str) -> str:
        """验证和修复HTML代码"""

        # 确保包含必要的HTML结构
        if not html_content.strip().startswith('<!DOCTYPE html>'):
            html_content = '<!DOCTYPE html>\n' + html_content

        # 确保有完整的HTML标签
        if '<html' not in html_content:
            html_content = f'<html lang="zh-CN">\n{html_content}\n</html>'

        # 确保有head和body
        if '<head>' not in html_content:
            head_content = """<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据分析报告</title>
</head>"""
            html_content = html_content.replace('<html', f'<html lang="zh-CN">\n{head_content}\n<body>')
            html_content += '\n</body>'

        # 确保包含必要的CSS和JS库
        if 'bootstrap' not in html_content.lower():
            # 添加Bootstrap CDN
            bootstrap_css = '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">'
            bootstrap_js = '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>'

            html_content = html_content.replace('</head>', f'{bootstrap_css}\n</head>')
            html_content = html_content.replace('</body>', f'{bootstrap_js}\n</body>')

        if 'echarts' not in html_content.lower():
            # 添加ECharts CDN
            echarts_js = '<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>'
            html_content = html_content.replace('</body>', f'{echarts_js}\n</body>')

        return html_content

    @staticmethod
    def _generate_fallback_report(
            user_query: str,
            analysis_results: List[Dict[str, Any]],
            error_message: str
    ) -> str:
        """生成备用报告（当模型生成失败时）"""

        # 构建简单的HTML报告
        analysis_summary = ""
        for i, result in enumerate(analysis_results, 1):
            analysis_summary += f"<h4>分析 {i}: {result.get('task', '未知任务')}</h4>"
            analysis_summary += f"<p>{result.get('result', '无结果')}</p>"

        fallback_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据分析报告</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .report-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
        .analysis-section {{ margin: 2rem 0; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row report-header p-4 mb-4">
            <div class="col">
                <h1>数据分析报告</h1>
                <p class="lead">用户需求: {user_query}</p>
                <p class="text-warning">注意: 这是备用报告，模型生成失败: {error_message}</p>
            </div>
        </div>
        
        <div class="row">
            <div class="col">
                <h2>分析结果</h2>
                {analysis_summary}
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col">
                <div class="alert alert-info">
                    <h4>报告说明</h4>
                    <p>此报告由备用系统生成。如需更详细的交互式报告，请检查模型服务状态。</p>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

        return fallback_html

    async def process_task(self, task) -> Task:
        """处理报告生成任务（重写基类方法以处理analysis_results）"""
        logger.info(f"[reporter] 开始处理报告生成任务: {task.description}")

        try:
            task.status = "in_progress"
            self.active_tasks[task.id] = task

            # 构建增强的任务提示，包含analysis_results的详细信息
            enhanced_prompt = self._build_enhanced_prompt(task)
            
            self.messages.append({
                "role": "user",
                "content": enhanced_prompt
            })

            # 确保Reporter能够访问完整的分析数据
            if hasattr(task, 'analysis_results') and task.analysis_results:
                logger.info(f"[reporter] 访问到 {len(task.analysis_results)} 个完整分析结果")
                
                # 为每个分析结果添加详细数据访问
                for i, result in enumerate(task.analysis_results):
                    if 'result' in result and isinstance(result['result'], dict):
                        # 提取关键信息，避免提示过长
                        key_info = self._extract_key_analysis_info(result['result'])
                        logger.info(f"[reporter] 分析结果 {i+1} 关键信息: {key_info[:200]}...")

            logger.info(f"[reporter] 发送增强任务提示，包含 {len(getattr(task, 'analysis_results', []))} 个分析结果")

            # 获取模型响应
            response = await self.model_client.chat_completion(
                messages=self.messages,
                tools=self.tool_manager.get_tool_schemas()
            )

            if response.get("status") == "error":
                logger.error(f"[reporter] 模型响应错误: {response.get('error')}")
                task.status = "failed"
                task.error = response.get("error")
            else:
                message = response["message"]
                self.messages.append(message)

                logger.info(f"[reporter] 收到模型响应: {message.get('content', '')[:200]}...")

                # 处理工具调用
                if message.get("tool_calls"):
                    logger.info(f"[reporter] 检测到工具调用: {len(message['tool_calls'])} 个")

                    for i, tool_call in enumerate(message["tool_calls"]):
                        function_call = tool_call["function"]
                        tool_name = function_call["name"]
                        args = function_call["arguments"]

                        logger.info(f"[reporter] 执行工具 {i + 1}: {tool_name} - 参数: {args}")

                        # 执行工具
                        result = await self.tool_manager.execute_tool(
                            tool_name=tool_name,
                            arguments=json.loads(args),
                            context={"conversation_id": self.conversation_id}
                        )
                        logger.info(f"[reporter] 工具 {tool_name} 执行结果: 成功={result.success}")

                        # 添加工具结果
                        self.messages.append({
                            "role": "tool",
                            "content": json.dumps(result.to_dict(), ensure_ascii=False),
                            "tool_call_id": tool_call["id"]
                        })

                task.result = message.get("content", "")
                task.status = "completed"
                task.completed_at = time.time()

                logger.info(f"[reporter] 报告生成任务完成")

            return task

        except Exception as e:
            logger.error(f"[reporter] 处理报告生成任务失败: {e}", exc_info=True)
            task.status = "failed"
            task.error = str(e)
            return task

    def _build_enhanced_prompt(self, task) -> str:
        """构建增强的任务提示，包含完整的分析结果"""
        base_prompt = task.description
        
        # 如果有完整的分析结果，添加详细信息
        if hasattr(task, 'analysis_results') and task.analysis_results:
            enhanced_info = "\n\n## 完整分析数据详情\n"
            enhanced_info += f"系统已收集 {len(task.analysis_results)} 个分析任务的完整结果数据。"
            enhanced_info += "请基于这些完整数据进行报告生成，避免产生幻觉。\n"
            
            for i, result in enumerate(task.analysis_results, 1):
                enhanced_info += f"\n### 分析任务 {i}: {result.get('description', '未知任务')}\n"
                enhanced_info += f"**执行角色:** {result.get('agent_role', '未知')}\n"
                
                # 智能提取关键信息，避免提示过长
                result_content = result.get('result', '无结果')
                if isinstance(result_content, dict):
                    # 提取结构化数据的关键信息
                    key_info = self._extract_key_analysis_info(result_content)
                    enhanced_info += f"**关键发现:**\n{key_info}\n"
                else:
                    # 对文本结果进行智能摘要
                    summary = self._summarize_text_result(str(result_content))
                    enhanced_info += f"**结果摘要:**\n{summary}\n"
                
                # 限制每个任务的显示长度，避免提示过长
                if len(enhanced_info) > 6000:  # 总长度限制
                    enhanced_info += "\n...（更多分析结果已省略，请参考完整数据）"
                    break
            
            base_prompt += enhanced_info
        
        return base_prompt

    def _extract_key_analysis_info(self, result_dict: dict) -> str:
        """从分析结果字典中提取关键信息"""
        key_info = []
        
        # 常见的关键字段
        key_fields = ['key_findings', 'insights', 'conclusions', 'recommendations', 
                     'summary', 'results', 'analysis', 'findings']
        
        for field in key_fields:
            if field in result_dict:
                value = result_dict[field]
                if isinstance(value, list):
                    key_info.extend([f"- {item}" for item in value[:3]])  # 限制数量
                elif isinstance(value, str):
                    key_info.append(f"- {value[:200]}")
                else:
                    key_info.append(f"- {str(value)[:200]}")
        
        # 如果没有找到标准字段，提取所有非技术性字段
        if not key_info:
            for key, value in result_dict.items():
                if not key.startswith('_') and not isinstance(value, (dict, list)):
                    key_info.append(f"- {key}: {str(value)[:100]}")
                if len(key_info) >= 5:  # 限制数量
                    break
        
        return '\n'.join(key_info) if key_info else "无关键信息可提取"

    def _summarize_text_result(self, text: str, max_length: int = 500) -> str:
        """智能摘要文本结果"""
        if len(text) <= max_length:
            return text
        
        # 提取关键句子（包含数字、结论性词语的句子）
        import re
        sentences = re.split(r'[.!?。！？]+', text)
        
        key_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 判断是否为关键句子
            if (re.search(r'\d+', sentence) or  # 包含数字
                any(word in sentence.lower() for word in ['结论', '发现', '建议', '重要', '关键', '显著', '因此', '所以'])):
                key_sentences.append(sentence)
        
        # 如果有关键句子，使用关键句子
        if key_sentences:
            summary = '。'.join(key_sentences[:3]) + '。'
            if len(summary) <= max_length:
                return summary
        
        # 否则使用开头和结尾
        return text[:max_length//2] + "..." + text[-max_length//2:]

    @staticmethod
    async def validate_report(html_content: str) -> Dict[str, Any]:
        """验证生成的HTML报告"""

        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }

        # 基本结构验证
        if '<!DOCTYPE html>' not in html_content:
            validation_results["warnings"].append("缺少DOCTYPE声明")

        if '<html' not in html_content:
            validation_results["errors"].append("缺少HTML根标签")
            validation_results["valid"] = False

        if '<head>' not in html_content:
            validation_results["warnings"].append("缺少head标签")

        if '<body>' not in html_content:
            validation_results["warnings"].append("缺少body标签")

        # 响应式设计检查
        if 'viewport' not in html_content:
            validation_results["suggestions"].append("建议添加viewport meta标签以确保响应式设计")

        # 库依赖检查
        if 'bootstrap' not in html_content.lower():
            validation_results["suggestions"].append("建议集成Bootstrap框架以确保响应式设计")

        if 'echarts' not in html_content.lower():
            validation_results["suggestions"].append("建议集成ECharts库以支持交互式图表")

        return validation_results


def create_report_generator(
        model_client: ModelClient,
        tool_manager: ToolManager,
        conversation_id: str
) -> ReportGeneratorAgent:
    """创建报告生成器实例"""
    return ReportGeneratorAgent(model_client, tool_manager, conversation_id)
