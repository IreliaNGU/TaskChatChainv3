from pydantic import BaseModel,field_validator
from constants.nlu import COARSE_GRAINED_INTENT,Fill_TYPE
from core.domain import Slot,Task
import json
from loguru import logger
from typing import List,Dict

class DomainRelavant(BaseModel):
    if_relavant: bool
    description: str = "输入是否与场景相关"

class Intent(BaseModel):
    id: str
    description: str
    #intent_type只能为coarse-grained或fine-grained
    intent_type: str

    def get_meta_info(self):
        return {
            'id': self.id,
            'desc': self.description,
            'intent_type': self.intent_type
        }

class CoarseGrainedIntent(Intent):
    intent_type:str = 'coarse-grained'

    @field_validator("id")
    @classmethod
    def id_must_valid(cls, v):
        """检查id是否合法取值"""
        try:
            _ = COARSE_GRAINED_INTENT(v)
        except Exception as e:
            raise ValueError("coarse grained intent is not a valid value.")
        return v

    def to_enum_type(self):
        return COARSE_GRAINED_INTENT(self.id)
    
class FineGrainedIntent(Intent):
    intent_type: str = 'fine-grained'
    schema: List[str] # 指定这个意图需要附带的附加信息
    sql:str = None #sql查询语句
    slots:Dict[str,str] = {} #槽值对信息,键为槽名，值为槽值
    fill_type:Fill_TYPE = Fill_TYPE.COVER #填充类型，可选值为补充、覆盖，默认为覆盖
    reply_utterance_id: str = None # 如果这个意图是在回复前文某句话，则这个字段为前文的句子id
    task_name: str = None # 任务名字

    @field_validator("schema")
    @classmethod
    def check_schema(cls, v):
        """检查附加信息是否合法"""
        for s in v:
            schema = json.loads(s)
            if schema['name'] not in ['sql','slot_value','fill_type','reply_utterance_id','task_name']:
                raise ValueError(f"schema {s} is not valid.")
        return v
    
    def set_sql(self, sql:str):
        self.sql = sql
    
    def set_slots(self, slots:Dict[str,str], scene_slots:List[Slot]):
        for k,v in slots.items():
            if k not in [slot.name for slot in scene_slots]:
                raise ValueError("slot名不在预定义的场景slot中,因此为fine-grained intent设置slot失败: %s" % k)   
            slot = [slot for slot in scene_slots if slot.name == k][0]
            if not slot.check_value(v):
                raise ValueError("value值不合法，因此为fine-grained intent设置slot失败: %s" % self.input)
        self.slots = slots

    def set_fill_type(self, fill_type:str):
        if fill_type not in [mem.value for mem in Fill_TYPE]:
            raise ValueError("fill_type is not valid.")
        self.fill_type = Fill_TYPE(fill_type)
    
    def set_task_name(self, task_name:str, scene_tasks:List[Task]):
        #检查task_name是否在任务列表中
        if task_name not in [task.name for task in scene_tasks]:
            raise ValueError("ACTION_REQUESTING意图识别结果中的task_name字段不在任务列表中: %s" % self.input)
        self.task_name = task_name

    def set_reply_utterance_id(self, reply_utterance_id:str):
        #检查reply_utterance_id是否为数字
        try:
            _ = int(reply_utterance_id)
        except Exception as e:
            raise ValueError("reply_utterance_id的值不是一个数字: %s." % reply_utterance_id)
        self.reply_utterance_id = reply_utterance_id

    def check_slots_validation(self, valid_slots:List[Slot]):
        """检查槽是否存在以及槽位值是否合法"""
        for slot_name,slot_value in self.slots.items():
            if slot_name not in [slot.name for slot in valid_slots]:
                raise ValueError(f"slot {slot_name} is not valid.")
            slot = [slot for slot in valid_slots if slot.name == slot_name][0]
            if not slot.check_value(slot_value):
                raise ValueError(f"slot {slot_name} value {slot_value} is not valid.")

    def get_meta_info(self):
        return {
            'id': self.id,
            'description': self.description,
            'intent_type': self.intent_type,
            'sql': self.sql,
            'slots': self.slots,
            'fill_type': self.fill_type,
            'reply_utterance_id': self.reply_utterance_id,
            'task_name': self.task_name
        }

