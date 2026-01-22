
Help me write a aibaton automation script to integrate the bybit exchange into the banexg project.

## Project Background
- banexg is a Go language unified interface library for cryptocurrency exchanges, already supporting exchanges like binance, okx, etc.
- Now need to add support for bybit exchange
- Related documentation is in the docs/ directory: help.md (architecture explanation), contribute.md (contribution guidelines), bybit_v5/ (bybit API documentation)

## Workflow Requirements

### 1. Preparation Phase

#### gen_doc_index prompt (generate index file)
Create the docs/bybit_index.md file to describe the paths of all files under docs/bybit_v5 and a brief single-line description of each file.
Used by AI to quickly locate the path of required functionality through this file. Use minimal text to briefly summarize the main purpose of each markdown file.
For each file under bybit_v5, just read the first 20~40 lines, as each file is an API endpoint and understanding its general purpose is sufficient.
Reduce redundant text, every file must be introduced. Should be done in batches: read a batch, update the file once, then continue reading the next batch.

#### Generate implementation plan prompt (inline directly in run call)
Reference @docs/help.md @docs/contribute.md @docs/bybit_v5 @docs/bybit_index.md
First read help.md and contribute.md to understand banexg's architecture and implementation specifications, clarify the interfaces that need to be implemented.
Then read bybit_index.md to understand all interfaces provided by bybit;
Then randomly select 7 interface documents under bybit_v5 to read, understand the format characteristics of bybit interface parameters and returned data, clarify how to handle interface data parsing.
Currently there are mainly two types: binance has completely different interface return data, so each interface can directly define structures for parsing; okx has partially consistent interface return data, such as all nested in the data field, which can reduce code redundancy through generics passing in different parts.
Then according to other requirements in banexg, read the required information outline from bybit documentation by yourself.
Finally, integrate all information and formulate an implementation plan; large blocks of code are prohibited in this plan. Describe step-by-step implementation steps in a concise and condensed style, but task granularity should be small enough and as detailed as possible.
Note that all exchange interfaces and various parameters involved in banexg need to be implemented. Find corresponding ones from interface documentation as much as possible and organize them into bybit_dev.
The estimated time for each part should be close. They should be arranged in order according to dependency relationships. Mark completed parts as done.
Output the plan content to docs/bybit_dev.md

### 2. Iterative Development Phase (Plan-Driven)
Encapsulate as `run_plan_steps()` function for reuse, loop until all steps are completed:

#### pick_plan_step prompt
Reference @docs/help.md @docs/contribute.md @docs/bybit_dev.md @docs/bybit_index.md
Currently need to integrate bybit exchange and implement relevant interfaces required in banexg. Please select the next part to implement based on the implementation plan in bybit_dev.md.
Output in the format <option>This is the title of the part to implement</option> at the end of the response. If all parts are completed, do not output the <option> part.

#### run_plan_step prompt (with {section} placeholder)
Reference @docs/help.md @docs/contribute.md @docs/bybit_dev.md @docs/bybit_index.md
Currently need to integrate bybit exchange and implement relevant interfaces required in banexg. Please help me start integrating bybit exchange step by step according to the detailed implementation plan in bybit_dev.md.
Based on the existing interface specifications in banexg and references from Binance and OKX, find the required interfaces from bybit_index. When implementing interfaces, read detailed documentation from docs/bybit_v5 based on the interface path.
Always follow the DRY principle during integration, check for redundant or similar code, and extract common parts if any for easier maintenance.
Always ensure compliance with banexg's specification requirements and related specifications of the root structure. If there is common logic for several exchanges, extract it to code files in external common packages.
The part to integrate now is: {section}

#### run_plan_check prompt (with {section} placeholder, max 5 retries)
Reference @contribute.md @help.md @docs/bybit_dev.md @docs/bybit_index.md
Currently integrating bybit, the {section} part has been completed and needs to be checked for errors or incomplete implementation.
Please read the banexg interface and parameter requirements, appropriately refer to the handling in binance to understand which parameters and logic need to be handled.
Then locate the interface file path based on bybit_index and read the detailed interface documentation under docs/bybit_v5.
Note that some common important parameters need to be supported, but some uncommon, exchange-specific parameters do not need support. Can refer to related methods in binance/okx interfaces.
Finally, summarize the places that need to be modified or improved. If the implementation of this part is all correct and without omissions, output <promise>DONE</promise> at the end of the response.
Please only focus on the {section} part.

#### run_code_refactor prompt
Use `git status -s` to view currently modified files and focus on code review and optimization of these files.
When redundant code is found, extract it into subfunctions to ensure adherence to the DRY principle and reduce duplicate or similar code fragments;
The core principle is to minimize redundant or similar code logic for easier maintenance. Keep business logic unchanged. Keep overall style unchanged with minor adjustments allowed.
When certain parts may overlap with those in other files, consider extracting common parts for reuse;
If a function body has only one line and no more than 3 parameters, it should be deleted and replaced with concise code at the reference location.
For mostly similar but slightly different cases, extract into parameterized reusable functions, components or fragments.

#### run_plan_test prompt (with {section} placeholder, loop_max=5)
Reference @docs/contribute.md @docs/help.md @docs/bybit_dev.md @docs/bybit_index.md
Currently integrating bybit, the {section} part has been completed. Now need to improve unit test cases for this part and ensure tests pass.
Unit tests need two types: one is simple function tests (no API requests); the other is actual interface tests submitted to the exchange (uniformly use `TestApi_` prefix). Can refer to related unit tests in binance;
Then locate the interface file path based on bybit_index and read the detailed interface documentation under docs/bybit_v5.
First ensure the first type of tests are complete and all pass. If there are errors, analyze and resolve them yourself, and repeat testing until they pass.
Then start the second type of tests. These tests should use apiKey and apiSecret configured in local.json to create a valid exchange object, then call actual interface methods to interact with the exchange production environment.
Some of the second type of tests need prior positions, you can first execute a unit test to place orders to create positions, then test related interfaces.
Please only focus on the {section} part. If you are confident that all tests pass without omissions or errors, output <promise>DONE</promise> at the end of the response.

#### run_plan_mark prompt (with {section} placeholder)
Reference @docs/bybit_dev.md Please help me mark the implementation of the {section} part in this document as completed

### 3. Overall Check Phase
Reference @docs/help.md @docs/contribute.md @docs/bybit_dev.md @docs/bybit_v5 @docs/bybit_index.md
Currently integrating bybit exchange, most necessary interfaces have been initially implemented. But there may still be many potential issues, bugs or omissions.
Please first read help.md and contribute.md to understand banexg's architecture and implementation specifications. Clarify the interfaces that need to be implemented.
Then read bybit_dev.md to understand the initial implementation plan. Then read bybit_index.md to understand all interfaces provided by bybit;
Note that all exchange interfaces and various parameters involved in banexg need to be implemented. Find corresponding ones from interface documentation as much as possible and organize them into bybit_dev.
Then, using banexg interfaces as units, check each parameter individually, appropriately refer to parameter implementation in binance; then read related interface documentation in bybit, all necessary parameters need to be supported, check for omissions or errors.
Update all discovered errors or omissions to bybit_dev.md, brief description is sufficient, no detailed code description needed. Change the status of parts that need modification to pending.
Then call run_plan_steps() again to fix issues

### 4. Integration Test Phase
Switch working directory to two levels up: `set_default(cwd=os.path.realpath(os.path.join(base_dir, "../..")))`

#### Integration test configuration prompt (inline directly in run call)
Reference banbot/go.mod banstrats/go.mod
Please help me enable the replace directive for the dependent banexg and banbot in the mod files of the above 2 projects to ensure direct compilation using local code.
Reference data/config.local.yml
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

### 5. Live Trading Verification Phase (Compile-Run-Fix Loop)
Define strat_dir variable before this phase begins. Encapsulate as `get_fix_bug(source, path)` and `run_and_fix(source, path)` two functions:

#### get_fix_bug function (using tpl template + {holder} placeholder)
tpl template content:
Reference banbot/doc/help.md banexg/docs/help.md banexg/docs/bybit_index.md
{holder}
Please read relevant code in banbot and banexg's bybit integration. Help me analyze and resolve;
If bybit interface documentation is needed, first read bybit_index to locate the interface file path, then read detailed interface documentation under docs/bybit_v5.
Think deeply, understand the root cause and resolve it. Only need to fix the code, allow unit tests and compilation tests, but prohibit re-running live trading.
If the log meets expectations without anomalies and no fix is needed, output <promise>DONE</promise> at the end of the response.

Generate different hold_text based on source type (default is trade):
- trade: {path}\nbanstrats/trade.log This is the live trading test log of banexg's newly integrated bybit in banbot.\nThe log is expected to have an order placement record, and later have order execution logs. If it doesn't meet expectations, there may be a bug.
- backtest: {path}\nbanstrats/backtest.log This is the quantitative backtest log of banexg's newly integrated bybit in banbot.\nThe log is expected to have many orders, and in the final statistics BarNum should be at least >100. If it doesn't meet expectations, there may be a bug.
- compile: {path}\nThis is a banbot compilation error log, please help me investigate and fix based on relevant code. After fixing, you can compile and test under banstrats with `go build -o bot`.

#### run_and_fix function
Loop up to 20 times:
1. Compile: start_process("go build -o bot", cwd=strat_dir, timeout_s=360).watch(stream=True)
2. On compile failure: logger.error record, pass res.output directly to get_fix_bug("compile", res.output), continue loop
3. Run: start_process("./bot "+source, cwd=strat_dir, timeout_s=360).watch(stream=True)
4. Save log to {source}.log file
5. logger.info record loop progress like f'[{i+1}/20] run and fix bug...'
6. Analyze and fix: run(fix_tip), check for <promise>DONE</promise> then break and exit

Test the following scenarios in order, **use a separate run() to modify run_policy in config.local.yml** before each scenario:
- Limit order test: run("Please help me modify the run_policy in data/config.local.yml to tmp:limit") + run_and_fix("trade", "banstrats/tmp/limit_order.go")
- Trigger entry test: run("Please help me modify the run_policy in data/config.local.yml to tmp:trigger") + run_and_fix("trade", "banstrats/tmp/trigger_ent.go")
- Moving average backtest: run("Please help me modify the run_policy in data/config.local.yml to ma:demo") + run_and_fix("backtest", "banstrats/ma/demo.go")

## Directory Structure
- base_dir: Script directory
- Working directory (cwd): Script parent directory (banexg project root)
- strat_dir: banstrats subdirectory under two levels up from script (defined before live trading verification phase)
- Log files: {source}.log (trade.log / backtest.log)
