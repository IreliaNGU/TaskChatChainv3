import os
from datetime import datetime
from typing import List,Dict,Optional

from api.vllm_model import chat
from template.intent_classification.IR_template import IntentClassficationTemplate
from models.intent_classification.intent_factory import IntentFramework, load_intent_from_config
from models.intent_classification.dialog import Utterance,Dialog,DialogRound
from data.tb_data.dataset import TBdataset
from config import config

class BaseIntentClassficationTask:

    def __init__(self, intent_config: str, history_round: int = 1, only_classify_user_intent: bool = True):
        """
        history_round: 对话历史轮数,用户一句系统一句为一轮，0为不传入对话历史，-1传入全部对话历史
        """
        self.intent_framework = IntentFramework(load_intent_from_config(intent_config))
        self.history_round = history_round
        self.only_classify_user_intent = only_classify_user_intent
    
    def predict(self, prompt:str) :
        return chat(prompt)

    def get_prompt(self, utterance:Utterance, history:List[Utterance]=None) -> Optional[str]:
        
        #保证一定拼接上history_round轮的历史对话，否则返回None
        if self.history_round > len(history):
            return None,None
        final_prompt = ""

        final_prompt += IntentClassficationTemplate.BASE_INTENT_RECOGNITION_PREFIX.format(candidate_intent=str(self.intent_framework))

        if self.history_round != 0:
            history_str = ""
            for i in range(self.history_round,0,-1):   
                history_str += str(history[-i])

            return IntentClassficationTemplate.with_history_intent_recognition_template(
                candidate_intent=str(self.intent_framework),
                history=history_str,
                utterance=str(utterance)
                ),history_str
        else:
            return IntentClassficationTemplate.no_history_intent_recognition_template(
                candidate_intent=str(self.intent_framework),
                utterance= str(utterance)
                ),None

def main():
    
    #加载数据集
    data_path = config['dataset']['src']
    dataset = TBdataset(window_size=-1,sample_mode=1,src=data_path,annoymous_speaker=False,summarize=False,completion=False)
    dataset.combine_continuous_sentence()
    dataset.split_train_test(ratio=0.8)

    task = BaseIntentClassficationTask(config['intent_framework']['src'],history_round=1)

    res = []
    f = open(os.path.join(config['output']['intent_classification'],"output{}".format(datetime.now())),'w',encoding='utf-8')
    f.write("模型:{model}\n历史对话轮次:{history_round}\n意图框架:{framework}\n".format(
        model="qwen2-7b",
        framework=str(task.intent_framework),
        history_round=task.history_round))
    for dialog_idx,dialog_struct in enumerate(dataset.test_set):
        f.write("******************dialog {}******************\n\n".format(dialog_idx))
        dialog = Dialog(dialog_struct['dialogID'])
        dialog.init_history(dialog_struct['content'])

        if task.only_classify_user_intent:
            for i,tup in enumerate(dialog.get_user_utterances()):
                utterance,history = tup[0],tup[1]
                f.write("******************sentence {}******************\n".format(i))
                prompt,history_str = task.get_prompt(utterance,history)
                if prompt:
                    ans = task.predict(prompt)
                    f.write("输入：{input}\n历史对话:{history}\n识别结果：{result}\n".format(input=str(utterance),history=str(history_str),result=ans))
                    res.append((utterance,history_str,ans))
                f.write("\n\n")
        f.write("\n\n")
    
    f.write("数据集样例数:{}".format(len(res)))



if __name__ == "__main__":
    main()