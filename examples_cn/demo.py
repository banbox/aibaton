from aibaton import run, set_default, setup_logger

setup_logger(level="debug")
set_default(provider="codex", dangerous_permissions=False)

res = run("如果你有一个机会回到5年前，你最想做的事情是什么？", options=["尽早享受生活", "买比特币", "持续学习精进"])
print(res.select())

res = run("你认为对于个人一生非常重要的，但99%的人都忽略的事情是什么？")
print(res.parse(["持续学习", "随时保持好奇心", "允许一切发生", "经常审视内心"]))

articles = ["旅行者卫星的信号没有出现。"]
prompt = "\n\n这是一篇震撼人心的作品，请继续续写约200字（只输出续写部分）。"
out = run(articles[0]+prompt)
for i in range(30):
    articles.append(out.text)
    all_text = "\n".join(articles)
    out = run(all_text+prompt)
