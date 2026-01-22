from aibaton import run, set_default, start_process, setup_logger, logger
import os

"""
This workflow is an automation script for integrating the bybit exchange into banexg (a cryptocurrency exchange library).
You can generate the current script based on the aibaton SKILL and work_flow_prompt.md prompt.
"""

base_dir = os.path.dirname(os.path.abspath(__file__))

setup_logger(filepath="aibaton.log")
set_default(provider="codex", dangerous_permissions=True, cwd=os.path.dirname(base_dir))

gen_doc_index = """
Please help me create the docs/bybit_index.md file. This file should describe the paths of all files under docs/bybit_v5 and a brief single-line description of each file.
It will be used by AI to quickly locate the path of the required functionality through this file. Please use minimal text to briefly summarize the main purpose of each markdown file.
For each file under bybit_v5, just read the first 20~40 lines, as each file is an API endpoint and understanding its general purpose is sufficient.
Finally, create bybit_index.md with minimal redundant text, ensuring every file is introduced. This should be done in batches: read a batch, update the file once, then continue reading the next batch.
"""

pick_plan_step = """
@docs/help.md @docs/contribute.md  @docs/bybit_dev.md  @docs/bybit_index.md 
Currently need to integrate bybit exchange and implement relevant interfaces required in banexg. Please select the next part to implement based on the implementation plan in bybit_dev.md.
Output in the format <option>This is the title of the part to implement</option> at the end of the response. If all parts are completed, do not output the <option> part.
"""

run_plan_step = """
@docs/help.md @docs/contribute.md  @docs/bybit_dev.md  @docs/bybit_index.md 
Currently need to integrate bybit exchange and implement relevant interfaces required in banexg. Please help me start integrating bybit exchange step by step according to the detailed implementation plan in bybit_dev.md.
Based on the existing interface specifications in banexg and references from Binance and OKX, find the required interfaces from bybit_index. When implementing interfaces, read detailed documentation from docs/bybit_v5 based on the interface path.
Always follow the DRY principle during integration, check for redundant or similar code, and extract common parts if any for easier maintenance.
Always ensure compliance with banexg's specification requirements and related specifications of the root structure. If there is common logic for several exchanges, extract it to code files in external common packages.
The part to integrate now is: {section}
"""

run_plan_check = """
@contribute.md @help.md @docs/bybit_dev.md @docs/bybit_index.md 
Currently integrating bybit, the {section} part has been completed and needs to be checked for errors or incomplete implementation.
Please read the banexg interface and parameter requirements, appropriately refer to the handling in binance to understand which parameters and logic need to be handled.
Then locate the interface file path based on bybit_index and read the detailed interface documentation under docs/bybit_v5.
Note that some common important parameters need to be supported, but some uncommon, exchange-specific parameters do not need support. Can refer to related methods in binance/okx interfaces.
Finally, summarize the places that need to be modified or improved. If the implementation of this part is all correct and without omissions, output <promise>DONE</promise> at the end of the response.
Please only focus on the {section} part.
"""

run_code_refactor = """
Use `git status -s` to view currently modified files and focus on code review and optimization of these files.
When redundant code is found, extract it into subfunctions to ensure adherence to the DRY principle and reduce duplicate or similar code fragments;
* The core principle is to minimize redundant or similar code logic for easier maintenance.
* Keep business logic unchanged. Keep overall style unchanged with minor adjustments allowed.
* When certain parts may overlap with those in other files, consider extracting common parts for reuse;
* If a function body has only one line and no more than 3 parameters, it should be deleted and replaced with concise code at the reference location.
* For mostly similar but slightly different cases, extract into parameterized reusable functions, components or fragments
"""

run_plan_test = """
@docs/contribute.md @docs/help.md @docs/bybit_dev.md @docs/bybit_index.md 
Currently integrating bybit, the {section} part has been completed. Now need to improve unit test cases for this part and ensure tests pass.
Unit tests need two types: one is simple function tests (no API requests); the other is actual interface tests submitted to the exchange (uniformly use `TestApi_` prefix). Can refer to related unit tests in binance;
Then locate the interface file path based on bybit_index and read the detailed interface documentation under docs/bybit_v5.
First ensure the first type of tests are complete and all pass. If there are errors, analyze and resolve them yourself, and repeat testing until they pass.
Then start the second type of tests. These tests should use apiKey and apiSecret configured in local.json to create a valid exchange object, then call actual interface methods to interact with the exchange production environment.
Some of the second type of tests need prior positions, you can first execute a unit test to place orders to create positions, then test related interfaces.
Please only focus on the {section} part. If you are confident that all tests pass without omissions or errors, output <promise>DONE</promise> at the end of the response.
"""

run_plan_mark = """
@docs/bybit_dev.md Please help me mark the implementation of the {section} part in this document as completed
"""


logger.info('Starting to generate doc index...')
run(gen_doc_index)

logger.info('Starting to generate implementation plan...')
run("""
@docs/help.md @docs/contribute.md  @docs/bybit_v5  @docs/bybit_index.md 
Currently need to integrate bybit exchange and implement relevant interfaces required in banexg.
Please first read help.md and contribute.md to understand the architecture and implementation specifications of banexg. Clarify the interfaces that need to be implemented.
Then read bybit_index.md to understand all interfaces provided by bybit;
Then randomly select 7 interface documents under bybit_v5 to read and understand the format characteristics of bybit interface parameters and returned data. Clarify how to handle interface data parsing.
Currently there are mainly two types: binance has completely different interface return data, so each interface can directly define structures for parsing; okx has partially consistent interface return data, such as all nested in the data field, which can reduce code redundancy through generics passing in different parts.
Then according to other requirements in banexg, read the required information outline from bybit documentation by yourself.
Finally, integrate all information and formulate an implementation plan; large blocks of code are prohibited in this plan. Describe step-by-step implementation steps in a concise and condensed style, but task granularity should be small enough and as detailed as possible.
Note that all exchange interfaces and various parameters involved in banexg need to be implemented. Find corresponding ones from interface documentation as much as possible and organize them into bybit_dev
The estimated time for each part should be close. They should be arranged in order according to dependency relationships. Mark completed parts as done.
Output the plan content to docs/bybit_dev.md
""")

def run_plan_steps():
    while True:
        logger.info('Selecting next plan step to process...')
        pick_res = run(pick_plan_step)
        section = pick_res.select()
        if not section:
            break
        for i in range(5):
            logger.info(f'Running plan step: {section}')
            run(run_plan_step.format(section=section))
            logger.info(f'Checking plan step: {section}')
            check_res = run(run_plan_check.format(section=section))
            if check_res.select("promise") == "DONE":
                break
        
        logger.info(f'Running code refactor optimization: {section}')
        run(run_code_refactor)
        logger.info(f'Running unit tests: {section}')
        run(run_plan_test.format(section=section), loop_max=5)
        logger.info(f'Marking plan as completed: {section}')
        run(run_plan_mark.format(section=section))

# First implementation following the plan step by step
run_plan_steps()

logger.info('Starting overall check for interface errors...')
run("""
@docs/help.md @docs/contribute.md  @docs/bybit_dev.md @docs/bybit_v5  @docs/bybit_index.md 
Currently integrating bybit exchange, most necessary interfaces have been initially implemented. But there may still be many potential issues, bugs or omissions.
Please first read help.md and contribute.md to understand banexg's architecture and implementation specifications. Clarify the interfaces that need to be implemented.
Then read bybit_dev.md to understand the initial implementation plan. Then read bybit_index.md to understand all interfaces provided by bybit;
Note that all exchange interfaces and various parameters involved in banexg need to be implemented. Find corresponding ones from interface documentation as much as possible and organize them into bybit_dev
Then, using banexg interfaces as units, check each parameter individually, appropriately refer to parameter implementation in binance; then read related interface documentation in bybit, all necessary parameters need to be supported, check for omissions or errors.
Update all discovered errors or omissions to bybit_dev.md, brief description is sufficient, no detailed code description needed. Change the status of parts that need modification to pending.
""")

# Re-implement step by step for issues found in second pass
run_plan_steps()

set_default(cwd=os.path.realpath(os.path.join(base_dir, "../..")))

logger.info('Modifying test yaml...')
run("""
banbot/go.mod
banstrats/go.mod
Please help me enable the replace directive for the dependent banexg and banbot in the mod files of the above 2 projects to ensure direct compilation using local code.
data/config.local.yml
Then in this yaml configuration, ensure the following modifications are made:
```yaml
env: prod
market_type: linear
time_start: "20250101"
time_end: "20260101"
put_limit_secs: 300
stake_amount: 50
leverage: 10
pairs: ['ETC']
run_policy:
  - name: tmp:limit_order
    run_timeframes: [1m]
exchange:
  name: bybit
```
Then help me compile under banstrats with `go build -o bot`, and then start a separate visual terminal (using gnome-terminal) to run `./bot spider` asynchronously and continuously.
""")

strat_dir = os.path.realpath(os.path.join(base_dir, "../../banstrats"))

def get_fix_bug(source: str, path: str):
    tpl = """
banbot/doc/help.md
banexg/docs/help.md
banexg/docs/bybit_index.md
{holder}
Please read relevant code in banbot and banexg's bybit integration. Help me analyze and resolve;
If bybit interface documentation is needed, first read bybit_index to locate the interface file path, then read detailed interface documentation under docs/bybit_v5.
Think deeply, understand the root cause and resolve it. Only need to fix the code, allow unit tests and compilation tests, but prohibit re-running live trading.
If the log meets expectations without anomalies and no fix is needed, output <promise>DONE</promise> at the end of the response.
"""
    hold_text = f"""
{path}
banstrats/trade.log This is the live trading test log of banexg's newly integrated bybit in banbot.
The log is expected to have an order placement record, and later have order execution logs. If it doesn't meet expectations, there may be a bug."""
    if source == "backtest":
        hold_text = f"""
{path}
banstrats/backtest.log This is the quantitative backtest log of banexg's newly integrated bybit in banbot.
The log is expected to have many orders, and in the final statistics BarNum should be at least >100. If it doesn't meet expectations, there may be a bug."""
    elif source == "compile":
        hold_text = f"""
{path}
This is a banbot compilation error log, please help me investigate and fix based on relevant code. After fixing, you can compile and test under banstrats with `go build -o bot`."""
    return tpl.format(holder=hold_text)

# Run live trading test and automatically fix bugs
def run_and_fix(source: str, path: str):
    fix_tip = get_fix_bug(source, path)
    for i in range(20):
        # compile
        res = start_process("go build -o bot", cwd=strat_dir, timeout_s=360).watch(stream=True)
        if res.returncode != 0:
            logger.error(f'compile failed: {res.output}')
            run(get_fix_bug("compile", res.output))
            continue
        # run
        res = start_process("./bot "+source, cwd=strat_dir, timeout_s=360).watch(stream=True)
        with open(os.path.join(strat_dir, f"{source}.log"), "w", encoding="utf-8") as f:
            f.write(res.output)
        logger.info(f'[{i+1}/20] run and fix bug...')
        fix_res = run(fix_tip)
        if fix_res.select("promise") == "DONE":
            break

# Run limit order test
logger.info('Running limit order test...')
run("Please help me modify the run_policy in data/config.local.yml to tmp:limit")
run_and_fix("trade", "banstrats/tmp/limit_order.go")
# Run trigger entry test
logger.info('Running trigger entry test...')
run("Please help me modify the run_policy in data/config.local.yml to tmp:trigger")
run_and_fix("trade", "banstrats/tmp/trigger_ent.go")


# Run moving average backtest
logger.info('Running moving average backtest...')
run("Please help me modify the run_policy in data/config.local.yml to ma:demo")
run_and_fix("backtest", "banstrats/ma/demo.go")
