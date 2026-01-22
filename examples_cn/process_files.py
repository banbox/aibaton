from aibaton import run, set_default, setup_logger, logger

setup_logger("process_files.log")
set_default(provider="codex", dangerous_permissions=True)

prompt = """
帮我审查当前页面代码并优化。发现冗余代码时，提取为子函数，确保遵循DRY原则，减少重复或相似的代码片段；
* 核心原则是尽量减少冗余或相似代码逻辑，方便维护。
* 保持业务逻辑不变。保持样式整体不变可细微调整。
* 当某些部分可能和其他文件中的某些重合时，考虑提取公共部分复用；
* 如果某函数body只有一行且参数不超过3个，则应该删除，在引用地方直接改为简短代码。
* 针对html冗余的，提取为snippet片段。
* 对于大部分相似但细微不同的，提取为带参数的可复用函数、组件或片段
"""

prompt_skill = '$code-refactor'

par_dir = '/app/banbot'

import os

def count_lines(filepath: str) -> int:
    """统计文件行数"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)

def get_go_files(root_dir: str, min_lines: int = 0) -> list[str]:
    """递归获取目录下所有 .go 文件的绝对路径，可选过滤最小行数"""
    go_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.go'):
                fpath = os.path.join(dirpath, f)
                if min_lines <= 0 or count_lines(fpath) > min_lines:
                    go_files.append(fpath)
    return go_files

if __name__ == '__main__':
    go_files = get_go_files(par_dir, min_lines=300)
    logger.info(f"共找到 {len(go_files)} 个 Go 文件")
    confirm = input("是否继续处理？(y/n): ").strip().lower()
    if confirm != 'y':
        logger.info("已取消")
        exit(0)
    
    for fpath in go_files:
        logger.info(f"Processing: {fpath}")
        run(f"文件路径: {fpath}\n\n{prompt}", cwd=par_dir, add_dirs=[par_dir], stream=True)
