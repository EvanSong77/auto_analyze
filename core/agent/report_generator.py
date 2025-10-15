# -*- coding: utf-8 -*-
# @Time    : 2025/10/15
# @Author  : EvanSong
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.model_client import ModelClient
from core.tool_manager import ToolManager
from utils.logger import get_logger

logger = get_logger(__name__)


class ReportGeneratorAgent:
    """HTML报告生成器智能体"""

    def __init__(
            self,
            model_client: ModelClient,
            tool_manager: ToolManager,
            conversation_id: str
    ):
        self.model_client = model_client
        self.tool_manager = tool_manager
        self.conversation_id = conversation_id
        self.system_prompt = self._get_system_prompt()
        self.messages: List[Dict[str, Any]] = [
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

## 报告结构模板
- **标题页**：醒目的标题、项目概述、关键指标
- **执行摘要**：突出关键发现和业务影响
- **分析背景**：项目目标、数据来源、分析方法
- **详细分析**：分章节展示分析过程和结果
- **可视化展示**：交互式图表和图形化结果
- **结论建议**：基于数据的洞察和可执行建议
- **附录**：技术细节、数据字典、参考资料

## 设计原则
- **用户体验优先**：确保报告易于阅读和理解
- **视觉层次清晰**：使用清晰的视觉层次突出重要信息
- **一致性**：保持整体风格和设计的一致性
- **可访问性**：确保报告对所有用户友好
- **性能优化**：优化加载速度和响应性能

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

请生成专业、美观、实用的HTML分析报告。"""

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
            analysis_summary += f"\n{i}. {result.get('task', '未知任务')}:\n"
            analysis_summary += f"   结果: {result.get('result', '无结果')[:500]}...\n"

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
- 一致的颜色方案和字体
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
