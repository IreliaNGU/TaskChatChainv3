# from data.tb_data.dataset import ACTION
from core.nlu.nlu import DomainRelavant,CoarseGrainedIntent,FineGrainedIntent
from constants.nlu import COARSE_GRAINED_INTENT as Coa, Fill_TYPE
from typing import Dict
from loguru import logger
import copy

import json

class BaseParser:
    
    def __init__(self) -> None:
        pass

    def wash(self) -> None:
        self.input = self.input.strip().replace(" ","").replace("\n","")
    
    def parse(self,input) -> Dict:
        self.input = input
        self.wash()
        parsed = {
            "output": self.input
        }
        return parsed

class IntentParser(BaseParser):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.intents =  [intent.value for intent in ACTION]
    
    def wash(self):
        #去除空格
        input = self.input.replace(" ","")
        #去除最后的句号
        if len(input)!=0 and input[-1] == "。":
            input = input[:-1]
        #去除“/:”等表情符号的干扰
        input = input.replace("/:","")
        #取“：”后的内容
        if "：" in input:
            input = input.split("：")[-1]
        # print("清洗后:",input)
        self.input = input

    def parse(self):
        #看字符串中是否有唯一出现的意图
        self.wash()
        has_intent_cnt = 0
        has_intent = []
        for intent in self.intents:
            if intent in self.input:
                has_intent_cnt += 1
                has_intent.append(intent)
        
        if has_intent_cnt == 1:
            self.output = has_intent[0]
        else:
            self.output = self.input

        parsed = {
            "output": self.output
        }
        
        return parsed

class DomainFilterParser(BaseParser):
    
    def parse(self, input) -> Dict:
        self.input = input
        self.wash()

        if "有关" in self.input:
            self.output = DomainRelavant(if_relavant=True)
        elif "无关" in self.input:
            self.output = DomainRelavant(if_relavant=False)
        else:
            logger.warning("无法解析的领域关联性描述: %s" % self.input)

        parsed = {
            "output": self.output
        }
            
        return parsed
    
class CoarseGrainedNluParser(BaseParser):

    def __init__(self, intents, **kwargs) -> None:
        super().__init__(**kwargs)
        self.intents = intents
    
    def parse(self,input) -> Dict:
        self.input = input
        self.wash()

        output = []
        for intent in self.intents:
            if intent.id in self.input:
                output.append(copy.deepcopy(intent))

        parsed = {
            "output": output
        }
        return parsed

class FineGrainedNluParser(BaseParser):

    def __init__(self, mapping, scene_slots, scene_tasks, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mapping = mapping
        self.scene_slots = scene_slots
        self.scene_tasks = scene_tasks
    
    def parse(self,input,input_type:Coa) -> Dict:
        self.input = input
        self.wash()
        #如果被包裹，则提取包裹在```json```中的内容
        if "```json" in self.input and "```" in self.input:
            self.input = self.input.split("```json")[-1].split("```")[0]
        #从字符串转化为python元素
        try:
            json_input = json.loads(self.input)
        except Exception as e:
            raise ValueError("无法解析的json字符串: %s" % self.input)
        
        output = []

        if input_type==Coa.INFO_REQUIRING:

            if 'id' not in json_input:
                raise ValueError("INFO_REQUIRING意图识别结果缺少id字段: %s" % self.input)
            
            fine_intents = self.mapping[Coa.INFO_REQUIRING]
            ids = [intent.id for intent in fine_intents]
            if json_input['id'] not in ids:
                raise ValueError("INFO_REQUIRING意图识别结果不在可选意图中: %s" % self.input)
            
            intent_id = json_input['id']
            match_intent = [intent for intent in fine_intents if intent.id==intent_id][0]
            
            if intent_id=='SQL_QUERY' and 'sql' not in json_input:
                raise ValueError("INFO_REQUIRING意图识别结果缺少sql字段: %s" % self.input)
            
            sql = json_input['sql'] if 'sql' in json_input else None

            copy_ = copy.deepcopy(match_intent)
            copy_.set_sql(sql)
            output.append(copy_)

        elif input_type==Coa.INFO_PROVIDING:
            fine_intents = self.mapping[Coa.INFO_PROVIDING]
            for item in json_input:
                if 'id' not in item :
                    raise ValueError("INFO_PROVIDING意图识别结果缺少id字段: %s" % self.input)
                if item['id'] not in [intent.id for intent in fine_intents]:
                    raise ValueError("INFO_PROVIDING意图识别结果不在可选意图中: %s" % self.input)
                #检查item中除了id外的字段是否与fine_intents中的schema一一匹配
                match_intent = [intent for intent in fine_intents if intent.id==item['id']][0]
                #将所有除了id外的keys放入列表
                keys = list(key for key in item.keys() if key != 'id')
                #检查keys是否与schema一一匹配
                if set(keys) != set([s['name'] for s in match_intent.schema]):
                    raise ValueError("INFO_PROVIDING意图识别结果与schema不匹配: %s" % self.input)
                
                copy_ = copy.deepcopy(match_intent)

                for k,v in item.items():
                    if k=='id': continue
                    if k=='slot_value':
                        #slot合法性检查
                        #检查v是不是字典
                        if not isinstance(v,dict):
                            raise ValueError("INFO_PROVIDING意图识别结果中的slot_value字段的内容不为字典: %s" % self.input)
                        #检查v中的key是否只有一个
                        if len(v.keys())!=1:
                            raise ValueError("INFO_PROVIDING意图识别结果中的slot_value字段中的键值对不唯一: %s" % self.input)
                        copy_.set_slots(v,self.scene_slots)
                    elif k=='fill_type':
                        copy_.set_fill_type(v)
                    elif k=='reply_utterance_id':
                        copy_.set_reply_utterance_id(v)
                    else:
                        raise NotImplementedError


                output.append(copy_)
                    
        elif input_type==Coa.ACTION_REQUESTING:
            fine_intents = self.mapping[Coa.ACTION_REQUESTING]
            for item in json_input:
                if 'id' not in item :
                    raise ValueError("ACTION_REQUESTING意图识别结果缺少id字段: %s" % self.input)
                if item['id'] not in [intent.id for intent in fine_intents]:
                    raise ValueError("ACTION_REQUESTING意图识别结果不在可选意图中: %s" % self.input)
                
                #检查item中除了id外的字段是否与fine_intents中的schema一一匹配
                match_intent = [intent for intent in fine_intents if intent.id==item['id']][0]
                #将所有除了id外的keys放入列表
                keys = list(key for key in item.keys() if key != 'id')
                #检查keys是否与schema一一匹配
                if set(keys) != set([json.loads(s)['name'] for s in match_intent.schema]):
                    raise ValueError("ACTION_REQUESTING意图识别结果与schema不匹配: %s" % self.input)
                
                copy_ = copy.deepcopy(match_intent)

                for k,v in item.items():
                    if k=='id': continue
                    if k=='task_name':
                        copy_.set_task_name(v,self.scene_tasks)

                output.append(copy_)

        else:
            raise NotImplementedError

        parsed = {
            "output": output
        }
        return parsed