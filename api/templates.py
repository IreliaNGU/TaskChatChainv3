DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。"""
TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

CNllama3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


Qwen_TEMPLATE = (
    "<|im_start|>system\n{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

def get_prompt_Qwen(system_prompt,instruction):
    return Qwen_TEMPLATE.format_map({'system_prompt':system_prompt,'instruction': instruction})

BaiChuan_TEMPLATE = (
    "<reserved_102>{instruction}<reserved_103>"  # <reserved_102>=user <reserved_103>=assistant
)

def get_prompt_Baichuan(instruction):
    return BaiChuan_TEMPLATE.format_map({'instruction': instruction})

BaiChuan2_TEMPLATE = (
    "<reserved_106>{instruction}<reserved_107>"  # <reserved_106>=user <reserved_107>=assistant
)
def get_prompt_Baichuan2(instruction):
    return BaiChuan2_TEMPLATE.format_map({'instruction': instruction})

def get_prompt_CNllama(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    # if '[/INST]' in instruction:
    #     TEMPLATE_ = TEMPLATE.replace(' [/INST]', '')
    # else:
    #     TEMPLATE_ = TEMPLATE
    # return TEMPLATE_.format_map({'instruction': instruction,'system_prompt': system_prompt})
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})

def get_prompt_CNllama3(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return CNllama3_TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})