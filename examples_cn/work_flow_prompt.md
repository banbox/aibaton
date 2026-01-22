
帮我编写一个aibaton自动化脚本，用于将bybit交易所对接到banexg项目中。

## 项目背景
- banexg是一个Go语言的加密货币交易所统一接口库，已支持binance、okx等交易所
- 现在需要新增bybit交易所的支持
- 相关文档位于docs/目录：help.md（架构说明）、contribute.md（贡献规范）、bybit_v5/（bybit API文档）

## 工作流程要求

### 1. 准备阶段

#### gen_doc_index提示词（生成索引文件）
创建docs/bybit_index.md文件，用于描述docs/bybit_v5下所有文件的路径和每个文件的简单单行介绍。
用于AI通过此文件快速定位所需功能的路径。使用最少的文本简要概括每个markdown文件的主要作用。
bybit_v5下的每个文件只需读取前20~40行左右即可，每个文件都是一个接口，了解其大概作用即可。
减少冗余的文字，每个文件都要介绍到。应当分批进行，阅读一批，更新一次文件，然后继续阅读下一批。

#### 生成实施计划提示词（直接内联到run调用）
引用 @docs/help.md @docs/contribute.md @docs/bybit_v5 @docs/bybit_index.md
先阅读help.md和contribute.md，了解banexg的架构和实现规范，明确需要实现的接口。
然后阅读bybit_index.md了解bybit提供的所有接口；
然后随意挑选7个bybit_v5下的接口文档阅读，了解bybit接口参数和返回数据的格式特点，明确需要如何处理接口数据解析。
目前主要有两种：binance是接口返回数据完全不同，直接每个接口定义结构体解析即可；okx是接口返回的数据有部分一致，比如全都嵌套在data字段中，这种可以通过泛型传入不同部分，减少代码冗余。
然后根据banexg中的其他要求，自行从bybit文档中阅读所需信息概要。
最后综合所有信息，制定一份实施计划；此计划中禁止大段代码，以简洁凝练的风格逐步描述分阶段的实施步骤，但任务粒度要足够小尽可能详细。
注意banexg中涉及的所有交易所接口和各种参数都需要实现，尽可能从接口文档中找到对应的，整理到bybit_dev中。
各个部分的预估耗时应该接近。应该按互相之间依赖关系按顺序排列。已完成的部分标记为完成。
输出计划内容到docs/bybit_dev.md

### 2. 迭代开发阶段（计划驱动）
封装为 `run_plan_steps()` 函数以便复用，循环执行直到所有步骤完成：

#### pick_plan_step提示词
引用 @docs/help.md @docs/contribute.md @docs/bybit_dev.md @docs/bybit_index.md
目前需要对接bybit交易所，实现banexg中需要的相关接口。请根据bybit_dev.md这个实施计划，帮我挑选下一步需要实现的部分。
在响应的末尾以<option>这是要实现的部分标题</option>格式输出。如果所有部分都已完成，则不要输出<option>部分。

#### run_plan_step提示词（带{section}占位符）
引用 @docs/help.md @docs/contribute.md @docs/bybit_dev.md @docs/bybit_index.md
目前需要对接bybit交易所，实现banexg中需要的相关接口。请根据bybit_dev.md这个详细的实施计划，帮我开始逐步对接bybit交易所。
根据banexg的已有接口规范和币安、okx的参考，从bybit_index中查找需要的接口，实现接口时根据接口路径从docs/bybit_v5下阅读详细文档。
对接过程中始终遵循DRY准则，检查是否有冗余或相似代码，有则提取公共部分，方便维护。
确保始终遵循banexg的规范要求，和根结构体的相关规范，如果有几个交易所共同的逻辑，则提取到外部公共包的代码文件中。
现在需要对接的部分是：{section}

#### run_plan_check提示词（带{section}占位符，最多重试5次）
引用 @contribute.md @help.md @docs/bybit_dev.md @docs/bybit_index.md
目前正在对接bybit，目前已完成{section}部分，需要检查是否实现有错误或不完善的地方。
请阅读banexg接口和参数要求，适当参考binance中的处理，了解哪些参数和逻辑需要处理。
然后根据bybit_index定位接口文件路径，阅读docs/bybit_v5下的详细接口文档。
注意一些常见重要的参数都需要支持，但部分不常用的，交易所特有的参数无需支持。可参考binance/okx等接口相关方法。
最后把发现的需要修改或完善的地方总结给我。如果此部分的实现均正确且无缺漏，则在响应最后输出<promise>DONE</promise>。
请注意只关注{section}部分。

#### run_code_refactor提示词
使用`git status -s`查看当前修改的文件，重点对这些文件进行代码审查并优化。
发现冗余代码时，提取为子函数，确保遵循DRY原则，减少重复或相似的代码片段；
核心原则是尽量减少冗余或相似代码逻辑，方便维护。保持业务逻辑不变。保持样式整体不变可细微调整。
当某些部分可能和其他文件中的某些重合时，考虑提取公共部分复用；
如果某函数body只有一行且参数不超过3个，则应该删除，在引用地方直接改为简短代码。
对于大部分相似但细微不同的，提取为带参数的可复用函数、组件或片段。

#### run_plan_test提示词（带{section}占位符，loop_max=5）
引用 @docs/contribute.md @docs/help.md @docs/bybit_dev.md @docs/bybit_index.md
目前正在对接bybit，目前已完成{section}部分，现在需要对这部分完善单元测试用例并确保测试通过。
单元测试需要两类：一类是简单的函数测试（不发出接口请求）；另一类是实际提交到交易所的接口测试（统一使用`TestApi_`前缀）。可参考binance中的相关单元测试；
然后根据bybit_index定位接口文件路径，阅读docs/bybit_v5下的详细接口文档。
首先确保第一类测试完整并全部通过，如果有错误自行分析解决，重复测试直到通过。
然后开始第二类测试，这些测试应该使用local.json中配置的apiKey和apiSecret创建一个有效的交易所对象，然后调用实际的接口方法和交易所生产环境接口进行交互。
第二类测试有些需要提前有仓位，可以先执行某个单元测试下单创建仓位，然后测试相关的接口。
请注意只关注{section}部分。如果确信测试全部通过且均无缺漏和错误，则在响应最后输出<promise>DONE</promise>。

#### run_plan_mark提示词（带{section}占位符）
引用 @docs/bybit_dev.md 请帮我把此文档中{section}部分的实现标记为完成

### 3. 整体检查阶段
引用 @docs/help.md @docs/contribute.md @docs/bybit_dev.md @docs/bybit_v5 @docs/bybit_index.md
目前正在对接bybit交易所，现在已初步实现大部分必要的接口。但仍可能有很多潜在问题或bug或遗漏。
请先阅读help.md和contribute.md，了解banexg的架构和实现规范。明确需要实现的接口。
然后阅读bybit_dev.md了解初步的实施计划方案。再阅读bybit_index.md了解bybit提供的所有接口；
注意banexg中涉及的所有交易所接口和各种参数都需要实现，尽可能从接口文档中找到对应的，整理到bybit_dev中。
然后以banexg接口为单位，逐个根据参数，适当参考binance中的参数实现；再阅读bybit中涉及的相关接口文档，对于必要的所有参数都需要支持，检查是否有遗漏或错误。
将发现的所有错误或遗漏等更新到bybit_dev.md中，简述即可，不需要详细代码描述。把需要修改的部分状态改为待完成。
然后再次调用 run_plan_steps() 修复问题

### 4. 集成测试阶段
切换工作目录到上两级：`set_default(cwd=os.path.realpath(os.path.join(base_dir, "../..")))`

#### 集成测试配置提示词（直接内联到run调用）
引用 banbot/go.mod banstrats/go.mod
请帮我在上面2个项目的mod文件中，对依赖的banexg和banbot启用replace指令，确保直接使用本地代码编译。
引用 data/config.local.yml
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

### 5. 实盘验证阶段（编译-运行-修复循环）
在此阶段开始前定义strat_dir变量。封装为 `get_fix_bug(source, path)` 和 `run_and_fix(source, path)` 两个函数：

#### get_fix_bug函数（使用tpl模板+{holder}占位符）
tpl模板内容：
引用 banbot/doc/help.md banexg/docs/help.md banexg/docs/bybit_index.md
{holder}
请阅读相关banbot和banexg中bybit的相关代码。帮我分析解决；
如果需要bybit接口文档，先阅读bybit_index定位接口文件路径，阅读docs/bybit_v5下的详细接口文档。
深入思考，了解根本原因并解决。只需修复代码即可，可允许单元测试和编译测试，禁止重新执行实盘。
如果日志符合预期没有异常，无需修复，则在响应最后输出<promise>DONE</promise>。

根据source类型生成不同hold_text（默认是trade）：
- trade：{path}\nbanstrats/trade.log 这是banexg新对接bybit在banbot中的实盘测试日志。\n日志中预期应该有一个下单记录，然后晚些有订单成交日志。如果不符合预期，则可能有bug。
- backtest：{path}\nbanstrats/backtest.log 这是banexg新对接bybit在banbot中的量化回测日志。\n日志中预期应该有很多订单，在最后的统计中BarNum应该至少>100，如果不符合预期，则可能有bug。
- compile：{path}\n这是banbot编译错误日志，请根据相关代码帮我排查修复。修复后可在banstrats下执行`go build -o bot`编译测试。

#### run_and_fix函数
循环最多20次：
1. 编译：start_process("go build -o bot", cwd=strat_dir, timeout_s=360).watch(stream=True)
2. 编译失败时：logger.error记录，直接将res.output传给get_fix_bug("compile", res.output)，continue继续循环
3. 运行：start_process("./bot "+source, cwd=strat_dir, timeout_s=360).watch(stream=True)
4. 保存日志到{source}.log文件
5. logger.info记录循环进度如 f'[{i+1}/20] run and fix bug...'
6. 分析修复：run(fix_tip)，检测<promise>DONE</promise>则break退出

依次测试以下场景，每个场景前**单独用一个run()修改config.local.yml中的run_policy**：
- 限价单测试：run("帮我把 data/config.local.yml 中的 run_policy 修改为 tmp:limit") + run_and_fix("trade", "banstrats/tmp/limit_order.go")
- 触发入场测试：run("帮我把 data/config.local.yml 中的 run_policy 修改为 tmp:trigger") + run_and_fix("trade", "banstrats/tmp/trigger_ent.go")
- 双均线回测：run("帮我把 data/config.local.yml 中的 run_policy 修改为 ma:demo") + run_and_fix("backtest", "banstrats/ma/demo.go")

## 目录结构
- base_dir：脚本所在目录
- 工作目录(cwd)：脚本父目录（banexg项目根）
- strat_dir：脚本上两级目录下的banstrats子目录（在实盘验证阶段前定义）
- 日志文件：{source}.log（trade.log / backtest.log）