# Auto Analyze - 增强多智能体数据分析系统

## 项目概述

Auto Analyze 是一个基于多智能体协作的增强数据分析系统，专门设计用于处理和分析结构化数据（如CSV、Excel文件），并生成详细的HTML分析报告。系统采用模块化架构，整合了文件处理、代码执行、智能分析和报告生成等功能。

### 核心特性

- **多智能体协作分析**: 基于角色分工的智能体系统协同工作，包括项目经理、数据分析师、报告生成器和质量保证员
- **结构化数据总结**: 使用AI模型对分析结果进行结构化总结，确保关键信息不丢失
- **自动化数据处理**: 支持多种文件格式的自动解析和处理
- **代码执行引擎**: 内置Jupyter执行环境，支持Python代码动态执行
- **HTML报告生成**: 自动生成美观、交互式的分析报告，集成ECharts可视化
- **工具化架构**: 模块化的工具系统，支持功能扩展

## 技术架构

### 核心模块

- **`main.py`**: 系统主入口，提供命令行接口
- **`core/agent/`**: 智能体系统核心模块
  - `enhanced_system.py`: 增强分析系统主控，整合多智能体和报告生成
  - `multi_agent_system.py`: 多智能体协作框架，支持任务分解和依赖管理
  - `report_generator.py`: HTML报告生成器，支持ECharts交互式图表
  - `collaboration.py`: 智能体协作机制，支持消息传递和数据共享
  - `exec_code.py`: 代码执行工具，集成Jupyter内核
  - `functions.py`: 基础功能模块
- **`core/tool_manager.py`**: 工具管理系统，支持动态工具注册和执行
- **`core/filesystem.py`**: 文件系统管理器，支持目录浏览和文件读取
- **`core/jupyter_execution.py`**: Jupyter执行引擎，提供安全的代码执行环境
- **`core/model_client.py`**: 模型客户端接口，支持多种AI模型调用
- **`core/image_utils.py`**: 图像处理工具

### 配置系统

- **`config.yaml`**: 主配置文件（模型、服务器、执行参数）
- **`schemas/config.py`**: 配置数据模型和验证

### 数据目录

- **`data/`**: 存储分析数据和生成的报告
  - `销售数据.csv`: 示例销售数据文件
  - `异常预警数据.xlsx`: 示例异常数据文件
  - `analysis_report.html`: 生成的HTML报告

### 日志系统

- **`logs/`**: 系统运行日志
  - `app.log`: 应用程序日志文件

## 快速开始

### 环境要求

- Python 3.8+
- Windows/Linux/macOS
- 网络连接（用于模型API调用）
- Jupyter内核（用于代码执行）

### 安装依赖

```bash
# 安装项目依赖
pip install -r requirements.txt
```

### 运行系统

```bash
# 基本使用
python main.py "分析销售数据"

# 启用详细日志
python main.py -v "帮我分析销售数据，找出业绩最好的产品"

# 指定配置文件
python main.py --config config.yaml "数据分析查询"
```

### 配置说明

系统使用 `config.yaml` 文件进行配置，主要配置项包括：

- **模型配置**: DeepSeek V3 和 GLM-4V 模型端点
- **服务器配置**: 本地服务端口设置
- **执行配置**: 超时时间、内存限制等

## 核心功能

### 1. 多智能体协作系统

**智能体角色分工：**
- **项目经理**: 需求分析、任务分解、进度协调
- **数据分析师**: 数据处理、统计分析、模型构建
- **报告生成器**: HTML报告设计、内容整合、可视化展示
- **质量保证员**: 结果验证、方法评估、问题识别

**协作流程：**
1. 项目经理分析用户需求并分解任务
2. 数据分析师执行具体的数据分析任务
3. 质量保证员验证分析结果的准确性
4. 报告生成器整合所有成果生成最终报告

### 2. 结构化数据总结

系统通过AI模型对每个任务的结果进行结构化总结，确保：
- **关键发现**：提取最重要的分析结果
- **数据洞察**：识别数据中的模式和趋势
- **业务影响**：分析结果对业务的实际影响
- **技术细节**：记录分析方法和技术实现
- **建议措施**：提供可执行的改进建议

### 3. 文件处理工具

- **读取目录**: 扫描和分析工作目录结构
- **读取文件**: 支持多种文件格式（CSV、Excel、文本等）
- **数据处理**: 自动解析结构化数据

### 4. 代码执行引擎

- **Python代码执行**: 在隔离环境中执行数据分析代码
- **可视化支持**: 支持图表生成和数据可视化
- **错误处理**: 完善的异常处理和结果捕获
- **并发执行**: 支持多个任务的并行执行

### 5. HTML报告生成

- **响应式设计**: 支持桌面、平板、手机等不同设备
- **交互式图表**: 集成ECharts实现动态数据可视化
- **专业排版**: 使用Bootstrap框架确保美观性
- **完整结构**: 包含摘要、分析、结论、建议等完整章节

## 开发指南

### 项目结构

```
auto_analyze/
├── main.py                 # 主程序入口
├── config.yaml            # 配置文件
├── README.md              # 项目说明文档
├── core/                  # 核心模块
│   ├── agent/            # 智能体系统
│   │   ├── enhanced_system.py
│   │   ├── multi_agent_system.py
│   │   ├── report_generator.py
│   │   ├── collaboration.py
│   │   ├── exec_code.py
│   │   └── functions.py
│   ├── filesystem.py     # 文件系统管理
│   ├── tool_manager.py   # 工具管理
│   ├── jupyter_execution.py
│   ├── model_client.py
│   └── image_utils.py
├── data/                 # 数据目录
├── schemas/             # 配置模式
├── utils/               # 工具函数
└── logs/               # 日志目录
```

### 添加新工具

1. 在 `tool_manager.py` 中创建新的 `Tool` 子类
2. 实现 `execute()` 和 `get_schema()` 方法
3. 在 `ToolManager._register_default_tools()` 中注册工具

```python
class NewTool(Tool):
    def __init__(self):
        super().__init__("new_tool", "新工具描述")
    
    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        # 实现工具逻辑
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        # 定义工具参数模式
        pass
```

### 扩展智能体功能

1. 在 `core/agent/` 下创建新的智能体模块
2. 继承 `BaseAgent` 类并实现特定逻辑
3. 集成到 `MultiAgentSystem` 中

```python
class NewAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return "智能体系统提示词"
    
    async def process_task(self, task: Task) -> Task:
        # 实现任务处理逻辑
        pass
```

## 使用示例

### 基本数据分析

```bash
python main.py "分析销售数据，找出3月份业绩最好的产品"
```

### 多维度分析

```bash
python main.py "帮我分析数据中的销售趋势、区域分布和产品表现"
```

### 自定义分析任务

```bash
python main.py "分析异常预警数据，识别主要风险点和改进建议"
```

## 故障排除

### 常见问题

1. **API连接失败**: 检查网络连接和模型端点配置
2. **文件读取错误**: 确认数据文件路径和权限
3. **内存不足**: 调整执行配置中的内存限制
4. **Jupyter内核问题**: 确保Jupyter内核正常运行

### 日志查看

启用详细日志模式查看执行过程：

```bash
python main.py -v "分析查询"
```

查看日志文件：
```bash
tail -f logs/app.log
```

### 调试模式

启用调试模式获取更详细的执行信息：

```python
# 在代码中设置日志级别
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化

### 并发处理

系统支持多任务的并发执行，通过以下方式优化性能：
- 任务依赖关系管理
- 并行结构化总结
- 异步工具执行

### 内存管理

- 及时清理临时文件
- 优化数据结构大小
- 使用流式处理大文件

## 安全考虑

### 代码执行安全

- 在隔离环境中执行用户代码
- 限制执行时间和内存使用
- 监控异常行为

### 数据安全

- 不存储敏感数据
- 使用安全的数据传输协议
- 定期清理临时文件

## 贡献指南

欢迎提交问题报告和功能改进建议。请确保：

- 代码符合项目编码规范
- 添加适当的测试用例
- 更新相关文档
- 遵循安全最佳实践

### 开发流程

1. Fork 项目仓库
2. 创建功能分支
3. 提交代码变更
4. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。

---

## 更新日志

### v1.0.0 (2025-10-17)

- **新增**: 结构化数据总结功能
- **改进**: 多智能体协作机制
- **优化**: 代码执行性能和安全
- **修复**: 类型安全问题和切片操作错误
- **增强**: HTML报告生成功能

### 主要改进

1. **结构化总结**: 使用AI模型对分析结果进行智能总结，确保关键信息不丢失
2. **并发处理**: 支持多任务的并行执行和总结
3. **错误修复**: 解决了切片操作的类型安全问题
4. **性能优化**: 改进了代码执行和数据处理效率

---

*最后更新: 2025年10月17日*