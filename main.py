# -*- coding: utf-8 -*-
# @Time    : 2025/10/15
# @Author  : EvanSong
import argparse
import asyncio
import logging
import os

from core.agent.enhanced_system import create_analysis_system_from_config
from schemas.config import get_settings


async def enhanced_analysis(user_query: str, verbose: bool = False):
    """增强的多智能体分析系统"""
    settings = get_settings()
    os.environ["DATA_PATH"] = "D:/codewen_workspace/DM-AI/auto_analyze/data"

    # 设置详细日志模式
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    try:
        # 创建分析编排器
        orchestrator = create_analysis_system_from_config("config.yaml")

        print("=" * 60)
        print("开始增强多智能体分析...")
        print(f"用户查询: {user_query}")
        print(f"详细日志: {'启用' if verbose else '禁用'}")
        print("=" * 60)

        result = await orchestrator.execute_analysis("analysis-session", user_query)

        print("=" * 60)
        print("分析完成!")
        print("=" * 60)

        # 保存HTML报告
        report_filename = "analysis_report.html"
        with open(f"data/{report_filename}", "w", encoding="utf-8") as f:
            f.write(result)

        print(f"HTML报告已保存到: data/{report_filename}")

        # 显示系统状态
        status = orchestrator.get_session_status("analysis-session")
        if status:
            print("\n系统状态摘要:")
            print(f"- 分析完成: {status.get('analysis_completed', False)}")
            print(f"- 结果数量: {status.get('analysis_results_count', 0)}")
            if status.get('execution_stats'):
                stats = status['execution_stats']
                print(f"- 总耗时: {stats.get('total_time', 0):.2f}秒")
                print(f"- 分析成功: {stats.get('analysis_successful', False)}")
                print(f"- 报告生成: {stats.get('report_generated', False)}")

        # 关闭会话
        await orchestrator.close_session("analysis-session")

        return result

    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        return f"错误: {str(e)}"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强多智能体分析系统")
    parser.add_argument("query", default="帮我分析销售数据，找出3月份业绩最好的产品，并生成详细的HTML报告",
                        help="分析查询内容", )
    parser.add_argument("-v", "--verbose", action="store_true", help="启用详细日志")

    args = parser.parse_args()

    # Windows事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    result = asyncio.run(enhanced_analysis(args.query, args.verbose))

    if not args.verbose:
        print("\n提示: 使用 -v 参数查看详细执行过程")


if __name__ == '__main__':
    main()
