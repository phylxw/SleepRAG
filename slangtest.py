import openai

# SGLang 通常兼容 OpenAI 的 API 格式
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1",  # 指向你的端口
    api_key="EMPTY"  # 本地部署通常不需要 key
)

response = client.chat.completions.create(
    model="default",  # 或者填你的模型名 Qwen/Qwen3-4B-Instruct-2507
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "我目前现在这个方法涨点特别不明显，几乎约等于不涨，使用优化后的记忆库只比最初的记忆库多做对1道题。你能帮我构思一个更好的而且适合ICML的记忆优化方法吗？你也知道，我这个现在基本代码框架已经搭好了，从初次测试到聚类到优化到最终测试。而且现在距离ICML投稿日已经很近了，现在这个idea不work我也有点着急了 ，这怎么办呀？而且我这个只是一个设计，也没有什么理论支撑。现在必须要进一步构思一些更好的更高级的方法了"},
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)