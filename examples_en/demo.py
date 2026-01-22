from aibaton import run, set_default, setup_logger

setup_logger(level="debug")
set_default(provider="codex", dangerous_permissions=False)

res = run("If you had a chance to go back 5 years, what would you most want to do?", options=["Enjoy life early", "Buy Bitcoin", "Keep learning and improving"])
print(res.select())

res = run("What do you think is extremely important for a person's life but ignored by 99% of people?")
print(res.parse(["Continuous learning", "Stay curious at all times", "Allow everything to happen", "Regularly reflect inward"]))

articles = ["The signal from the Voyager satellite did not appear."]
prompt = "\n\nThis is a stunning work, please continue writing about 200 words (output only the continuation)."
out = run(articles[0]+prompt)
for i in range(30):
    articles.append(out.text)
    all_text = "\n".join(articles)
    out = run(all_text+prompt)
