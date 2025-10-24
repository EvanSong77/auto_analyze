# Auto Analyze - 多智能体数据分析系统

## 项目概述

Auto Analyze 是一个基于多智能体协作的数据分析系统，专门设计用于处理和分析结构化数据（如CSV、Excel文件），并生成详细的HTML分析报告。系统采用模块化架构，整合了文件处理、代码执行、智能分析和报告生成等功能。

## 核心特性

- **多智能体协作**: 基于职责分工的智能体系统，包括项目经理、数据分析师、数据智能体和报告生成器
- **动态任务规划**: 根据执行结果智能迭代生成新任务，实现自适应分析流程
- **代码执行引擎**: 内置Jupyter执行环境，支持Python代码动态执行
- **HTML报告生成**: 自动生成美观的分析报告
- **工具化架构**: 模块化的工具系统，支持功能扩展

## 技术架构

### 核心模块

- **`main.py`**: 系统主入口，提供命令行接口
- **`core/agent/`**: 智能体系统核心模块
  - `enhanced_system.py`: 增强分析系统主控
  - `multi_agent_system.py`: 多智能体协作框架
  - `report_generator.py`: HTML报告生成器
  - `collaboration.py`: 智能体协作机制
  - `exec_code.py`: 代码执行工具
  - `functions.py`: 基础功能模块
- **`core/tool_manager.py`**: 工具管理系统
- **`core/filesystem.py`**: 文件系统管理器
- **`core/jupyter_execution.py`**: Jupyter执行引擎
- **`core/model_client.py`**: 模型客户端接口
- **`core/image_utils.py`**: 图像处理工具

### 配置系统

- **`config.yaml`**: 主配置文件（模型、服务器、执行参数）
- **`config/config.py`**: 配置数据模型

### 数据目录

- **`data/`**: 存储分析数据和生成的报告
  - `销售报告/`: 示例销售数据和分析报告
  - `毛利诊断/`: 示例毛利分析数据

## 快速开始

### 环境要求

- Python 3.8+
- Windows/Linux/macOS
- 网络连接（用于模型API调用）
- Jupyter内核（用于代码执行）

### 安装依赖

由于项目使用本地部署的模型服务，主要依赖包括：
- requests: HTTP请求库
- openai: OpenAI API兼容接口
- pandas: 数据处理
- numpy: 数值计算
- matplotlib: 数据可视化

### 运行系统

```bash
# 基本使用
python main.py "分析销售数据"

# 启用详细日志
python main.py -v "帮我分析销售数据，找出业绩最好的产品"

# 使用默认查询
python main.py
```

### 配置说明

系统使用 `config.yaml` 文件进行配置，主要配置项包括：

- **模型配置**: DeepSeek V3 模型端点
- **服务器配置**: 本地服务端口设置

## 核心功能

### 1. 多智能体协作系统

**智能体角色分工：**
- **项目经理**: 需求分析、任务规划、进度协调
- **数据智能体**: 技术执行、数据处理、代码执行
- **数据分析师**: 业务分析、洞察发现、建议生成
- **报告生成器**: HTML报告设计、内容整合

### 2. 动态任务规划

- **迭代式分析**: 根据执行结果智能生成新任务
- **自适应流程**: 基于分析发现自动调整分析方向
- **智能终止**: 检测分析收敛条件，避免无限循环

### 3. 文件处理工具

- **读取目录**: 扫描和分析工作目录结构
- **读取文件**: 支持多种文件格式（CSV、Excel、文本等）
- **数据处理**: 自动解析结构化数据

### 4. 代码执行能力

- **Jupyter集成**: 在隔离环境中执行Python代码
- **安全执行**: 限制执行时间和内存使用
- **结果捕获**: 自动捕获代码执行结果和可视化

## 项目结构

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
├── config/              # 配置模块
├── utils/               # 工具函数
└── logs/               # 日志目录
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

## 开发说明

### 添加新工具

1. 在 `tool_manager.py` 中创建新的 `Tool` 子类
2. 实现 `execute()` 和 `get_schema()` 方法
3. 在 `ToolManager._register_default_tools()` 中注册工具

### 扩展智能体功能

1. 在 `core/agent/` 下创建新的智能体模块
2. 继承 `BaseAgent` 类并实现特定逻辑
3. 集成到 `MultiAgentSystem` 中

## 许可证

本项目采用 MIT 许可证。