from openai import OpenAI

API_KEY = 'sk-1fG8XDohxQh3hJYclbuDsEkA5MdTAI0e355V4oNv1u7WrQ7E'

def chat_gpt(utterance,model_type="gpt-3.5-turbo-ca"):
    """
    与gpt模型进行对话
    """
    #与gpt模型进行对话
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=API_KEY,
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.cn/v1"
    )
    completion = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": "You are a helpful assistant.你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。"},
            {"role": "user", "content": utterance}
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    chat_gpt("你好")