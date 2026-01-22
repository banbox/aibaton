from aibaton import run, set_default, start_process, setup_logger, logger
import os

"""
此工作流是用于banexg（一个加密货币交易所库）中对接bybit交易所的自动化脚本。
您可根据aibaton SKILL以及work_flow_prompt.md提示词生成当前脚本。
"""

base_dir = os.path.dirname(os.path.abspath(__file__))

setup_logger(filepath="aibaton.log")
set_default(provider="codex", dangerous_permissions=True, cwd=os.path.dirname(base_dir))

gen_doc_index = """
请帮我创建 docs/bybit_index.md 文件，这个文件应该用于描述 docs/bybit_v5 下的所有文件的路径和每个文件的简单单行介绍。
用于AI通过此文件快速定位所需功能的路径。请使用最少的文本简要概括每个markdown文件的主要作用。
bybit_v5下的每个文件只需读取前20~40行左右即可，每个文件都是一个接口，了解其大概作用即可。
最后创建bybit_index.md ，减少冗余的文字，每个文件都要介绍到。应当分批进行，阅读一批，更新一次文件，然后继续阅读下一批。
"""

pick_plan_step = """
@docs/help.md @docs/contribute.md  @docs/bybit_dev.md  @docs/bybit_index.md 
目前需要对接bybit交易所，实现banexg中需要的相关接口。请根据bybit_dev.md 这个实施计划，帮我挑选下一步需要实现的部分。
在响应的末尾以<option>这是要实现的部分标题</option>格式输出。如果所有部分都已完成，则不要输出<option>部分。
"""

run_plan_step = """
@docs/help.md @docs/contribute.md  @docs/bybit_dev.md  @docs/bybit_index.md 
目前需要对接bybit交易所，实现banexg中需要的相关接口。请根据bybit_dev.md 这个详细的实施计划，帮我开始逐步对接bybit交易所。
根据banexg的已有接口规范和币安、okx的参考，从bybit_index中查找需要的接口，实现接口时根据接口路径从docs/bybit_v5 下阅读详细文档。
对接过程中始终遵循DRY准则，检查是否有冗余或相似代码，有则提取公共部分，方便维护。
确保始终遵循banexg的规范要求，和根结构体的相关规范，如果有几个交易所共同的逻辑，则提取到外部公共包的代码文件中。
现在需要对接的部分是：{section}
"""

run_plan_check = """
@contribute.md @help.md @docs/bybit_dev.md @docs/bybit_index.md 
目前正在对接bybit，目前已完成 {section} 部分，需要检查是否实现有错误或不完善的地方。
请阅读banexg接口和参数要求，适当参考binance中的处理，了解哪些参数和逻辑需要处理。
然后根据bybit_index定位接口文件路径，阅读docs/bybit_v5下的详细接口文档。
注意一些常见重要的参数都需要支持，但部分不常用的，交易所特有的参数无需支持。可参考binance/okx等接口相关方法。
最后把发现的需要修改或完善的地方总结给我。如果此部分的实现均正确且无缺漏，则在响应最后输出<promise>DONE</promise>。
请注意只关注 {section} 部分。
"""

run_code_refactor = """
使用`git status -s`查看当前修改的文件，重点对这些文件进行代码审查并优化。
发现冗余代码时，提取为子函数，确保遵循DRY原则，减少重复或相似的代码片段；
* 核心原则是尽量减少冗余或相似代码逻辑，方便维护。
* 保持业务逻辑不变。保持样式整体不变可细微调整。
* 当某些部分可能和其他文件中的某些重合时，考虑提取公共部分复用；
* 如果某函数body只有一行且参数不超过3个，则应该删除，在引用地方直接改为简短代码。
* 对于大部分相似但细微不同的，提取为带参数的可复用函数、组件或片段
"""

run_plan_test = """
@docs/contribute.md @docs/help.md @docs/bybit_dev.md @docs/bybit_index.md 
目前正在对接bybit，目前已完成 {section} 部分，现在需要对这部分完善单元测试用例并确保测试通过。
单元测试需要两类：一类是简单的函数测试（不发出接口请求）；另一类是实际提交到交易所的接口测试（统一使用`TestApi_`前缀）。可参考binance中的相关单元测试；
然后根据bybit_index定位接口文件路径，阅读docs/bybit_v5下的详细接口文档。
首先确保第一类测试完整并全部通过，如果有错误自行分析解决，重复测试直到通过。
然后开始第二类测试，这些测试应该使用local.json中配置的apiKey和apiSecret创建一个有效的交易所对象，然后调用实际的接口方法和交易所生产环境接口进行交互。
第二类测试有些需要提前有仓位，可以先执行某个单元测试下单创建仓位，然后测试相关的接口。
请注意只关注 {section} 部分。如果确信测试全部通过且均无缺漏和错误，则在响应最后输出<promise>DONE</promise>。
"""

run_plan_mark = """
@docs/bybit_dev.md 请帮我把此文档中 {section} 部分的实现标记为完成
"""


logger.info('开始生成文档索引...')
run(gen_doc_index)

logger.info('开始生成实施计划...')
run("""
@docs/help.md @docs/contribute.md  @docs/bybit_v5  @docs/bybit_index.md 
目前需要对接bybit交易所，实现banexg中需要的相关接口。
请先阅读help.md和contribute.md，了解banexg的架构和实现规范。明确需要实现的接口。
然后阅读bybit_index.md 了解bybit提供的所有接口；
然后随意挑选7个bybit_v5下的接口文档阅读，了解bybit接口参数和返回数据的格式特点。明确需要如何处理接口数据解析。
目前主要有两种：binance是接口返回数据完全不同，直接每个接口定义结构体解析即可；okx是接口返回的数据有部分一致，比如全都嵌套在data字段中，这种可以通过泛型传入不同部分，减少代码冗余。
然后根据banexg中的其他要求，自行从bybit文档中阅读所需信息概要。
最后综合所有信息，制定一份实施计划；此计划中禁止大段代码，以简洁凝练的风格逐步描述分阶段的实施步骤，但任务粒度要足够小尽可能详细。
注意banexg中涉及的所有交易所接口和各种参数都需要实现，尽可能从接口文档中找到对应的，整理到bybit_dev中
各个部分的预估耗时应该接近。应该按互相之间依赖关系按顺序排列。已完成的部分标记为完成。
输出计划内容到docs/bybit_dev.md
""")

def run_plan_steps():
    while True:
        logger.info('选择下一个要处理的计划步骤...')
        pick_res = run(pick_plan_step)
        section = pick_res.select()
        if not section:
            break
        for i in range(5):
            logger.info(f'运行计划步骤: {section}')
            run(run_plan_step.format(section=section))
            logger.info(f'检查计划步骤: {section}')
            check_res = run(run_plan_check.format(section=section))
            if check_res.select("promise") == "DONE":
                break
        
        logger.info(f'运行代码重构优化: {section}')
        run(run_code_refactor)
        logger.info(f'运行单元测试: {section}')
        run(run_plan_test.format(section=section), loop_max=5)
        logger.info(f'标记计划完成: {section}')
        run(run_plan_mark.format(section=section))

# 第一次按计划逐步实施
run_plan_steps()

logger.info('开始整体检查是否接口有错误...')
run("""
@docs/help.md @docs/contribute.md  @docs/bybit_dev.md @docs/bybit_v5  @docs/bybit_index.md 
目前正在对接bybit交易所，现在已初步实现大部分必要的接口。但仍可能有很多潜在问题或bug或遗漏。
请先阅读help.md和contribute.md，了解banexg的架构和实现规范。明确需要实现的接口。
然后阅读bybit_dev.md 了解初步的实施计划方案。再阅读bybit_index.md 了解bybit提供的所有接口；
注意banexg中涉及的所有交易所接口和各种参数都需要实现，尽可能从接口文档中找到对应的，整理到bybit_dev中
然后以banexg接口为单位，逐个根据参数，适当参考binance中的参数实现；再阅读bybit中涉及的相关接口文档，对于必要的所有参数都需要支持，检查是否有遗漏或错误。
将发现的所有错误或遗漏等更新到bybit_dev.md中，简述即可，不需要详细代码描述。把需要修改的部分状态改为待完成。
""")

# 针对第二次发现的问题，重新逐步实施
run_plan_steps()

set_default(cwd=os.path.realpath(os.path.join(base_dir, "../..")))

logger.info('修改测试yaml...')
run("""
banbot/go.mod
banstrats/go.mod
请帮我在上面2个项目的mod文件中，对依赖的banexg和banbot启用replace指令，确保直接使用本地代码编译。
data/config.local.yml
然后在这个yaml配置中，确保进行如下修改：
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
然后帮我在banstrats下执行`go build -o bot`编译，然后启动一个单独的可视化终端（使用gnome-terminal）异步持续运行`./bot spider`。
""")

strat_dir = os.path.realpath(os.path.join(base_dir, "../../banstrats"))

def get_fix_bug(source: str, path: str):
    tpl = """
banbot/doc/help.md
banexg/docs/help.md
banexg/docs/bybit_index.md
{holder}
请阅读相关banbot和banexg中bybit的相关代码。帮我分析解决；
如果需要bybit接口文档，先阅读bybit_index定位接口文件路径，阅读docs/bybit_v5下的详细接口文档。
深入思考，了解根本原因并解决。只需修复代码即可，可允许单元测试和编译测试，禁止重新执行实盘。
如果日志符合预期没有异常，无需修复，则在响应最后输出<promise>DONE</promise>。
"""
    hold_text = f"""
{path}
banstrats/trade.log 这是banexg新对接bybit在banbot中的实盘测试日志。
日志中预期应该有一个下单记录，然后晚些有订单成交日志。如果不符合预期，则可能有bug。"""
    if source == "backtest":
        hold_text = f"""
{path}
banstrats/backtest.log 这是banexg新对接bybit在banbot中的量化回测日志。
日志中预期应该有很多订单，在最后的统计中BarNum应该至少>100，如果不符合预期，则可能有bug。"""
    elif source == "compile":
        hold_text = f"""
{path}
这是banbot编译错误日志，请根据相关代码帮我排查修复。修复后可在banstrats下执行`go build -o bot`编译测试。"""
    return tpl.format(holder=hold_text)

# 运行实盘测试并自动修复bug
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

# 运行限价单测试
logger.info('运行限价单测试...')
run("帮我把 data/config.local.yml 中的 run_policy 修改为 tmp:limit")
run_and_fix("trade", "banstrats/tmp/limit_order.go")
# 运行触发入场测试
logger.info('运行触发入场测试...')
run("帮我把 data/config.local.yml 中的 run_policy 修改为 tmp:trigger")
run_and_fix("trade", "banstrats/tmp/trigger_ent.go")


# 运行双均线回测
logger.info('运行双均线回测...')
run("帮我把 data/config.local.yml 中的 run_policy 修改为 ma:demo")
run_and_fix("backtest", "banstrats/ma/demo.go")
