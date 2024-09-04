from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# GLM-4-9B-Chat-1M
# max_model_len, tp_size = 1048576, 4

# pip install vllm
from vllm import LLM, SamplingParams


prompts = [
    "Funniest joke ever:",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=200)
llm = LLM(model="/home/hzl/work/TaskChatChainv3/models/Qwen-14B-chat",trust_remote_code=True)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    print(prompt)
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")