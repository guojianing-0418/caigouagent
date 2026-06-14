"""命令行调试入口。

用途：IT 或开发人员可以绕过 WebUI，直接用 YAML 配置跑一次计划阶段识别。

示例：
    python scripts/run_plan_stage.py --config examples/p725.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# 脚本位于 scripts/ 下，运行时把项目根目录加入 import 路径。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agent import run_plan_stage
from backend.models import ProjectConfig, ProjectState
from backend.storage import append_log, save_project


def main() -> None:
    """读取 YAML 配置并运行 Agent。"""

    parser = argparse.ArgumentParser(description="运行计划阶段采购风险物料识别")
    parser.add_argument("--config", required=True, help="项目 YAML 配置路径")
    args = parser.parse_args()

    config_path = Path(args.config)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    project_config = ProjectConfig(**data)
    state = ProjectState(config=project_config)
    append_log(state, "命令行项目已创建。")
    save_project(state)

    result = run_plan_stage(state)
    print(f"项目ID: {result.id}")
    print(f"状态: {result.status}")
    print(f"风险物料数量: {len(result.risks)}")
    print(f"导出文件: {result.export_path}")
    if result.error:
        print(f"错误: {result.error}")


if __name__ == "__main__":
    main()

