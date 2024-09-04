# CUDA_VISIBLE_DEVICES="1,2" \
# python -m vllm.entrypoints.openai.api_server \
# --model "/disk0/fin_group/zyn/hgmodels/chinese-alpaca-2-13b" \
# --gpu-memory-utilization 0.5

# CUDA_VISIBLE_DEVICES="2" \
# python -m vllm.entrypoints.api_server \
# --port 8012 \
# --model "/disk0/fin_group/zyn/hgmodels/Qwen-7B-Chat" \
# --trust-remote-code \
# --gpu-memory-utilization 0.5 \
# --tensor-parallel-size 1

model=/home/zyn/langchain_test/chinese-alpaca-2-13b
port=8011
# model=/home/zyn/langchain_test/0820_sft2_5ep_merged
port=8012
model=/home/zyn/hgmodels/Qwen-14B-chat
# model=/home/zyn/hgmodels/Qwen-7B-Chat
port=8014
port=8033

model=/home/hzl/work/TaskChatChainv3/Chinese_LLAMA3/models/llama-3-chinese-8b-instruct-v2
port=8040

CUDA_VISIBLE_DEVICES="0,1,2" \
python -m vllm.entrypoints.api_server \
--model ${model} \
--port ${port} \
--host "0.0.0.0" \
--trust-remote-code \
--max-logprobs 2000 \
--tensor-parallel-size 1 \
--gpu-memory-utilization 0.95 

