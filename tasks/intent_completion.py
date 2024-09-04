import json
import logging
import yaml
from typing import Optional,Dict,List

from api.vllm_model import chat
from utils.common import extract_json_from_str,extract_json_list_from_str
from intent_recognition.template.prompt import DEFAULT_IF_COMPLETION_TEMPLATE,FEW_SHOTS

intent_framework_path = 'intent_recognition/config/intent_framework.yaml'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_intent_framework(framework_name="tb"):
    with open(intent_framework_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    if framework_name not in data:
        logger.error("没有找到对应的framework")

    return data[framework_name]

class BaseIntentFrameworkCompletionTask:

    def __init__(self,framework: str,repeat_count: int=10, if_iteration:bool=False,iterate_count: int=5):

        self._framework_name = framework
        self._framework_dict:Optional[Dict] = None
        self.if_iter = if_iteration
        if self.if_iter:
            self.repeat_count = 1
            self.iterate_count = iterate_count
            self._new_framework_str = ""
            self._new_intent_statistics = {}
        else:
            self.repeat_count = repeat_count
            self.iterate_count = 0
        self._framework_str = self.construct_framework_markdown_by_name(framework)

    def construct_prompt(self) -> str:

        if not self._framework_str:
            return ""
        
        #若有迭代出的新的框架则使用
        if self._new_framework_str and self.if_iter: 
            framework_str = self._new_framework_str
        else:
            framework_str = self._framework_str

        prompt = DEFAULT_IF_COMPLETION_TEMPLATE.format(intent_framework=framework_str)
        return prompt

    #将framework转成markdown格式(markdown)
    def construct_framework_markdown_by_name(self, framework:str) -> str:
        intent_framework_str = ""
        return intent_framework_str
    
    #统计新增的intent
    #输入为完善后的intent框架的列表，输出为这些框架中在原始框架中新增得到的intent统计（键为intent名称，值为新增的次数）
    def statistics_new_intent(self, new_framework_list: List[Dict]) -> Dict:
        new_intent_dict = {}
        for new_framework in new_framework_list:
            for intent in new_framework:
                if intent['name'] not in self._framework_dict:
                    if intent['name'] not in new_intent_dict:
                        new_intent_dict[intent['name']] = 1
                    else:
                        new_intent_dict[intent['name']] += 1
        return new_intent_dict

    #单轮迭代
    def run(self):
        prompt = self.construct_prompt()
        if not prompt:
            return
        count = self.repeat_count
        new_framework_list = []
        for i in range(count):
            reply = chat(prompt)
            new_framework = extract_json_from_str(reply)

            #如果没有解析出framework则直接返回
            if not new_framework:
                return
            #构造新的framework_str
            self._new_framework_dict = new_framework
            self._new_framework_str = ""
            for intent in new_framework:
                self._new_framework_str += "- {intent} description:{desc}\n".format(intent=intent['name'],desc=intent['description'])

            new_framework_list.append(new_framework)
            with open('intent_recognition/output/intent_framework_completion_{framework_name}.out'.format(framework_name=self._framework_name), "a", encoding='utf8') as f:
                f.write("*" * 25 + str(i) + "*" * 25)
                f.write('\n')
                f.write(reply)
                f.write('\n')
        new_intent_dict = self.statistics_new_intent(new_framework_list)

        #若 self._new_intent_statistics 为空则直接赋值，否则将新的统计结果加入
        if not self._new_intent_statistics:
            self._new_intent_statistics = new_intent_dict
        else:
            for intent in new_intent_dict:
                if intent not in self._new_intent_statistics:
                    self._new_intent_statistics[intent] = new_intent_dict[intent]
                else:
                    self._new_intent_statistics[intent] += new_intent_dict[intent]

    #多轮迭代，即将每一轮迭代的输出返回给模型作为下一次的输入
    def run_iterate(self):
        if not self.if_iter:
            return
        
        for i in range(self.iterate_count):
            logger.info("第{count}轮迭代".format(count=i+1))
            self.run()
            logger.info(json.dumps(self.get_last_framework(),ensure_ascii=False,indent=2))

        logger.info("迭代完成")
        logger.info(json.dumps(self.get_new_intent_statistics(),ensure_ascii=False,indent=2))
    
    def get_last_framework(self) -> Dict:
        if not self._new_framework_dict:
            return {}
        return self._new_framework_dict

    def get_new_intent_statistics(self) -> Dict:
        if not self._new_intent_statistics:
            return {}
        return self._new_intent_statistics

class TbCompletionTask(BaseIntentFrameworkCompletionTask):


    def construct_framework_markdown_by_name(self, framework: str) -> str:

        intent_framework_str = ""
        self._framework_dict = load_intent_framework(framework)

        intent_lst = self._framework_dict
        for i in intent_lst:
            intent_framework_str += "- {intent}\n".format(intent=i)

        return intent_framework_str


Tbtask = TbCompletionTask(framework='tb',if_iteration=True,iterate_count=5)

#从文件中读取并统计新增的intent
# path = '/home/hzl/work/TaskChatChainv3/intent_recognition/output/gpt4_output'
# with open(path, 'r', encoding='utf-8') as f:
#     s = f.read()
# print(Tbtask.statistics_new_intent(extract_json_list_from_str(s)))

#多轮迭代调用
Tbtask.run_iterate()

# Tbtask.run()
# print(Tbtask.get_new_intent_statistics())


# def intent_framework_completion_test(fewshots=False):

#     intent_lst = load_intent_framework(framework_name='tb')

#     intent_framework_str=""
#     for i in intent_lst:
#         intent_framework_str += "- {intent} description:{desc}\n".format(intent=i,desc="desc" )

#     prompt = DEFAULT_IF_COMPLETION_TEMPLATE.format(intent_framework = intent_framework_str)

#     count = 10
#     for i in range(count):
#         reply = chat(prompt)

#         with open('intent_recognition/output/intent_framework_completion_tb.out',"a",encoding='utf8') as f:
#             f.write("*"*25+str(i)+"*"*25)
#             f.write('\n')
#             f.write(reply)
#             f.write('\n')

# intent_framework_completion_test()
# print(json.dumps(load_intent_framework(framework_name='NLUpp'),ensure_ascii=False,indent=2))