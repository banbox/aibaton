from aibaton import run, set_default, setup_logger, logger

setup_logger("process_files.log")
set_default(provider="codex", dangerous_permissions=True)

prompt = """
Help me review and optimize the current page code. When redundant code is found, extract it into subfunctions to ensure adherence to the DRY principle and reduce duplicate or similar code fragments;
* The core principle is to minimize redundant or similar code logic for easier maintenance.
* Keep business logic unchanged. Keep overall style unchanged with minor adjustments allowed.
* When certain parts may overlap with those in other files, consider extracting common parts for reuse;
* If a function body has only one line and no more than 3 parameters, it should be deleted and replaced with concise code at the reference location.
* For html redundancy, extract into snippet fragments.
* For mostly similar but slightly different cases, extract into parameterized reusable functions, components or fragments
"""

prompt_skill = '$code-refactor'

par_dir = '/app/banbot'

import os

def count_lines(filepath: str) -> int:
    """Count file lines"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)

def get_go_files(root_dir: str, min_lines: int = 0) -> list[str]:
    """Recursively get absolute paths of all .go files in directory, optionally filter by minimum lines"""
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
    logger.info(f"Found {len(go_files)} Go files in total")
    confirm = input("Continue processing? (y/n): ").strip().lower()
    if confirm != 'y':
        logger.info("Cancelled")
        exit(0)
    
    for fpath in go_files:
        logger.info(f"Processing: {fpath}")
        run(f"File path: {fpath}\n\n{prompt}", cwd=par_dir, add_dirs=[par_dir], stream=True)
