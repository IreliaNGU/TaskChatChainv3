from openai import OpenAI

class OllamaProvider:
    def __init__(self, model_name:str, base_url:str, api_key: str="ollama"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    def __call__(self, text: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role":"user","content":text}],
        )
        result = response.choices[0].message.content

        return result

    def process(self, text: str):
        response_text = self(text)
        return {"text": response_text}

if __name__ == "__main__":
    provider = OllamaProvider(model_name="qwen2.5:latest",
                              base_url="http://10.249.50.13:11434/v1")
    print(provider.process("你好"))

