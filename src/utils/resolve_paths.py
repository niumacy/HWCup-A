"""将 paths.yaml 中的 ${REPO_ROOT} 占位符替换为本地仓库路径。

用法:
    python -m src.utils.resolve_paths  # 默认修改 config/paths.yaml
    python -m src.utils.resolve_paths --show  # 只显示当前解析结果
"""
import os
import sys
import re
from pathlib import Path

REPO_ROOT_PLACEHOLDER = '${REPO_ROOT}'
REPO_ROOT = str(Path(__file__).resolve().parents[2])


def resolve(content: str) -> str:
    """将 ${REPO_ROOT} 替换为本地仓库绝对路径。"""
    return content.replace(REPO_ROOT_PLACEHOLDER, REPO_ROOT)


def resolve_file(path: str, in_place: bool = True) -> str:
    """解析单个 yaml 文件中的占位符。"""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    resolved = resolve(content)
    if in_place:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(resolved)
    return resolved


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='config/paths.yaml')
    parser.add_argument('--show', action='store_true', help='只显示，不修改')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f'❌ 找不到 {args.file}', file=sys.stderr)
        sys.exit(1)

    if args.show:
        with open(args.file, 'r', encoding='utf-8') as f:
            print(resolve(f.read()))
    else:
        resolve_file(args.file, in_place=True)
        print(f'✅ 已解析 {args.file} 中的 ${REPO_ROOT} → {REPO_ROOT}')


if __name__ == '__main__':
    main()
