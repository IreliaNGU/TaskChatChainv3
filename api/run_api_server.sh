# POSTMAN
# 10.249.45.41:8908/qa
# {
#     "name": "多轮对话",
#     "history":[["你好","你好"],["请介绍一下你自己"]],
#     "stream":false
# }
# {
#     "name": "单轮对话",
#     "history":[["请介绍一下你自己"]],
#     "stream":false
# }

# Qwen

# model=/new_disk/models_for_all/icrc_lambergpt/round3/lambergpt_13b_warmup_3/sft_3_warmup
# port=8906
# model=/new_disk/models_for_all/icrc_lambergpt/round3/lambergpt_13b_sft_3/sft_3_v3
# model=/raid/hgmodels/sft_3

model=/home/hzl/work/TaskChatChainv3/models/Chinese_LLAMA3/models/llama-3-chinese-8b-instruct-v2
# model=/home/hzl/work/TaskChatChainv3/models/chinese-alpaca-2-13b
# model=/disk0/fin_group/hzl/ZHAI/models/chatglm3-6b
model=/home/hzl/work/TaskChatChainv3/models/Qwen-14B-chat
# model=/home/hzl/work/TaskChatChainv3/models/glm-4-9b-chat
model=/home/hzl/work/TaskChatChainv3/models/Qwen2-7B-Instruct
# model=/home/hzl/work/TaskChatChainv3/models/Qwen2-7B-instruct_lora_sft

# model=/raid/hgmodels/Baichuan2-13B-Chat
port=8040

modelname=${model##*/}

# nohup python -m vllm.entrypoints.api_server \
CUDA_VISIBLE_DEVICES="0" \
nohup python -u /home/hzl/work/TaskChatChainv3/api/api_server.py \
--model ${model} \
--port ${port} \
--host "0.0.0.0" \
--gpu-memory-utilization 0.30 \
--dtype half \
--trust-remote-code \
--served-model-name "Qwen2-7B-Instruct" \
--max-logprobs 2000 \
--tensor-parallel-size 1 > /disk0/fin_group/hzl/TaskChatChainv3/api/logs/${modelname}.txt 2>&1 & 
