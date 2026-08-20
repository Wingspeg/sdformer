#!/usr/bin/env python3
"""
通过 rsync 将本地代码同步到服务器。
同步内容：models/*.py、scripts/*.sh、exp/*.py、layers/*.py
"""

import os
import subprocess
import sys

# 配置 - 使用当前工作目录
LOCAL_DIR = os.getcwd() + "/"
REMOTE = "leosue@10.0.0.1:~/sdformer/"

print(f"本地目录: {LOCAL_DIR}")


def run_rsync(dry_run=False):
    """执行 rsync 同步命令"""
    cmd = [
        "rsync", "-avz", "--progress",
        # include 需要的目录和文件
        "--include", "models/",
        "--include", "models/*.py",
        "--include", "scripts/",
        "--include", "scripts/**/",
        "--include", "scripts/**/*.sh",
        "--include", "exp/",
        "--include", "exp/*.py",
        "--include", "layers/",
        "--include", "layers/*.py",
        # exclude 不需要的内容
        "--exclude", "dataset/",
        "--exclude", "checkpoints/",
        "--exclude", "*.md",
        "--exclude", "*.pdf",
        "--exclude", "*.xlsx",
        "--exclude", "*.log",
        "--exclude", "__pycache__/",
        "--exclude", "*",
    ]

    if dry_run:
        cmd.append("--dry-run")

    # 添加源和目标
    cmd.extend([LOCAL_DIR, REMOTE])

    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("-" * 60)
        print("✅ 同步成功" if not dry_run else "✅ 干跑完成，以上文件将被同步")
    else:
        print("-" * 60)
        print(f"❌ 同步失败，返回码: {result.returncode}")

    return result.returncode


def main():
    print("=" * 60)
    print("开始同步代码到服务器")
    print("=" * 60)
    exit_code = run_rsync(dry_run=False)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
