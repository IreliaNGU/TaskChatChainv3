#使用大模型进行单轮对话，目前支持cnllama2,cnllama3,Qwen,Baichuan,Baichuan2
#请确保模型已经通过vllm的方式运行在端口上

import requests
import argparse
import json
from typing import *
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from loguru import logger

import sys
sys.path.append("/home/hzl/work/TaskChatChainv3")
from api.templates import get_prompt_CNllama, get_prompt_CNllama3, get_prompt_Qwen, get_prompt_Baichuan, get_prompt_Baichuan2,\
    DEFAULT_SYSTEM_PROMPT

class Model_Client():
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.api_url = f"http://{self.host}:{self.port}/generate"

    def post_http_request(self, prompt: str,
                          stream: bool = False,
                          **kwargs) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        sampling_params = kwargs
        # sampling_params
        # n: int = 1,
        # best_of: Optional[int] = None,
        # presence_penalty: float = 0.0,
        # frequency_penalty: float = 0.0,
        # temperature: float = 1.0,
        # top_p: float = 1.0,
        # top_k: int = -1,
        # use_beam_search: bool = False,
        # stop: Union[None, str, List[str]] = None,
        # ignore_eos: bool = False,
        # max_tokens: int = 16,
        # logprobs: Optional[int] = None,
        pload = {
            "prompt": prompt,
            "stream": stream,
        }
        pload.update(**sampling_params)
        response = requests.post(
            self.api_url, headers=headers, json=pload, stream=True)  # 此stream不是sampling_params的stream
        return response

    def generate(self, prompt:str, more_dict=None, **kwargs) -> List[str]:
        # SamplingParams kwargs
        # prompt = get_prompt(prompt)
        # logger.info('input:{}', prompt)
        response = self.post_http_request(prompt, **kwargs)
        data = json.loads(response.content)
        # print(data)
        if more_dict is not None:
            more_dict['ret'] = data
        output = data["text"]  # list of str. 多少由传入的n决定
        if len(output) == 1:
            output = output[0]
        # logger.info('output:{}', output)
        return output

class Model(LLM):
    model: Model_Client = None
    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        max_tokens=500,
        presence_penalty=1.0,
    )
    model_type: str = 'cnllama'  # cnllama qwen baichuan baichuan2
    @property
    def _llm_type(self) -> str:
        return "Chinese-LLaMA-Alpaca-2"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"generation_config":self.generation_config}
    
    def load_model(self, host, port, model_type=None):
        self.model = Model_Client(host, port)
        if model_type is not None:
            self.model_type = model_type

    def get_next_tokens_score(self, prompt, tokens_lst, **kwargs):
        temperature = kwargs.get("temperature", 0.2)
        self.generation_config.update({'temperature': temperature})
        if self.generation_config['temperature'] == 0.:  # no random
            self.generation_config['top_k'] = -1
            self.generation_config['top_p'] = 1

        if self.model is None:
            raise RuntimeError("Must call `load_model()` to load model and tokenizer!")
    
        # 根据不同模型构造模板
        if self.model_type == 'qwen':
            input_text = get_prompt_Qwen(instruction=prompt)
            self.generation_config.update(stop='<|im_end|>')
        elif self.model_type == 'baichuan':
            input_text = get_prompt_Baichuan(instruction=prompt)
        elif self.model_type == 'baichuan2':
            input_text = get_prompt_Baichuan2(instruction=prompt)
        elif self.model_type == 'cnllama':
            system_prompt = kwargs.get("system_prompt", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT
            input_text = get_prompt_CNllama(instruction=prompt, system_prompt=system_prompt)
        elif self.model_type == 'cnllama3':
            system_prompt = kwargs.get("system_prompt", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT
            input_text = get_prompt_CNllama3(instruction=prompt, system_prompt=system_prompt)
        else:  # default cnllama
            system_prompt = kwargs.get("system_prompt", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT
            input_text = get_prompt_CNllama(instruction=prompt, system_prompt=system_prompt)

        generation_config = dict(self.generation_config)
        generation_config.update(max_tokens=1,logprobs=2000)
        more_dict = {}
        response = self.model.generate(input_text, only_output=False,more_dict=more_dict, **generation_config)
        # print(more_dict)
        logprobs = json.loads(more_dict['ret']['logprobs'][0])

        predict_token = logprobs['tokens'][0]

        if predict_token in tokens_lst:
            return predict_token
        else:
            return None

        # top_logprobs = logprobs['top_logprobs'][0]
        # print(top_logprobs)

        # ret_scores = []
        # for tok in tokens_lst:
        #     ret_scores.append(top_logprobs.get(tok, -1e5))  # 相当于[/INST] A 中间有个空格
        # print(ret_scores)

        # min_rank = 2000
        # target_token = ""
        # for ret in ret_scores:
        #     if min_rank > ret['rank']:
        #         min_rank = ret['rank']
        #         target_token = ret['decoded_token']

        # max_idx = ret_scores.index(max_value)
        # logger.info('input>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>:\n{}', input_text)
        # logger.info('output<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<:\n{} {}', target_token, ret_scores)
        return target_token


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
        ) -> str:

        temperature = kwargs.get("temperature", 0.2)
        self.generation_config.update({'temperature': temperature})
        if self.generation_config['temperature'] == 0.:  # no random
            self.generation_config['top_k'] = -1
            self.generation_config['top_p'] = 1

        if self.model is None:
            raise RuntimeError("Must call `load_model()` to load model and tokenizer!")
    
        # 根据不同模型构造模板
        if self.model_type.startswith('qwen'):
            system_prompt = kwargs.get("system_prompt", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT
            input_text = get_prompt_Qwen(system_prompt=system_prompt,instruction=prompt)
            self.generation_config.update(stop='<|im_end|>')
        elif self.model_type == 'baichuan':
            input_text = get_prompt_Baichuan(instruction=prompt)
        elif self.model_type == 'baichuan2':
            input_text = get_prompt_Baichuan2(instruction=prompt)
        elif self.model_type == 'cnllama':
            system_prompt = kwargs.get("system_prompt", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT
            input_text = get_prompt_CNllama(instruction=prompt, system_prompt=system_prompt)
        elif self.model_type == 'cnllama3':
            system_prompt = kwargs.get("system_prompt", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT
            input_text = get_prompt_CNllama3(instruction=prompt, system_prompt=system_prompt)
        elif self.model_type == 'glm4':
            system_prompt = kwargs.get("system_prompt", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT
        else:  # default cnllama
            system_prompt = kwargs.get("system_prompt", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT
            input_text = get_prompt_CNllama(instruction=prompt, system_prompt=system_prompt)

        # response = self.model.generate(input_text, only_output=False, **self.generation_config)
        response = self.model.generate(input_text, only_output=False,**self.generation_config)
        # # 根据不同模型截取输出
        if self.model_type.startswith('qwen'):
            response = response.split("<|im_start|>assistant\n")[-1].strip()
            response = response.split("<|im_end|>")[0].strip()
        elif self.model_type == 'baichuan':
            response = response.split("<reserved_103>")[-1].strip()
        elif self.model_type == 'baichuan2':
            response = response.split("<reserved_107>")[-1].strip()
        elif self.model_type == 'cnllama':
            response = response.split("[/INST]")[-1].strip()
        elif self.model_type == 'cnllama3':
            response = response.split("<|end_header_id|>")[-1].strip()
        else:
            response = response.split("[/INST]")[-1].strip()
            
        logger.info('input>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>:\n{}', input_text)
        logger.info('output<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<:\n{}', response)
        return response

#外部调用接口1:生成式
def chat(utterance,model_type='cnllama3',host='127.0.0.1',port=8040):
    llm = Model()
    llm.load_model(host,port,model_type)
    return llm(utterance)

#外部调用接口2：判别式
def chat_prob_classify(utterance,tokens_lst,model_type='cnllama3',host='127.0.0.1',port=8040):
    llm = Model()
    llm.load_model(host,port,model_type)
    return llm.get_next_tokens_score(utterance,tokens_lst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--model_type", type=str, default="qwen")

    args = parser.parse_args()

    llm = Model()
    llm.load_model(args.host, args.port, args.model_type)
    llm("请问1+1等于多少？计算结果只返回数字。答案：")
