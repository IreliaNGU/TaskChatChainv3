#根据上下文窗口建立数据集
import jsonlines
from pathlib import Path
import random
from loguru import logger
from enum import Enum
from tqdm import tqdm
import sys
# sys.path.append("/home/hzl/work/TaskChatChainv3")
from api.vllm_model import chat
from template.intent_classification.IR_template import HISTORY_WITH_LABEL_TEMPLATE, SUMMARIZE_HISTORY_TEMPLATE,COMPLETION_TEMPLATE
from api.templates import DEFAULT_SYSTEM_PROMPT
from constants.dialog import sentence_segment,intent_segment,Role
from config import config

data_path = '/home/hzl/work/TaskChatChainv3/intent_recognition/tb_data/processed/tb_dialogue.jsonl'

class ACTION(Enum):
    GREETING = "打招呼"
    ASK = "询问"
    REQUEST = "请求"
    INFORM = "告知"
    RESPONSE = "回复"
    CONFIRM = "确认"
    EXPLAIN = "解释"
    ACCEPT = "接受"
    UNKNOWN = "未知"

intent_lst = ['Greeting','Ask','Request','Inform','Response','Confirm','Explain','Accept','NULL']
cand_intent_cn = [intent.value for intent in ACTION]

intent_cn2en = {
    "打招呼":"Greeting",
    "询问":"Ask",
    "请求":"Request",
    "告知":"Inform",
    "回复":"Response",
    "确认":"Confirm",
    "解释":"Explain",
    "接受":"Accept",
    "未知":"NULL"
}

intent_description = {
    ACTION.GREETING.value:{
        "desc":"表示礼貌",
        "example":"你好"
    },
    ACTION.ASK.value:{
        "desc":"向对方发出询问",
        "example":"我什么时候收到？"
    },
    ACTION.REQUEST.value:{
        "desc":"根据需要向对方发出请求",
        "example":"请尽快发货"
    },
    ACTION.INFORM.value:{
        "desc":"告知对方当前状况或者事实",
        "example":"我地址变了"
    },
    ACTION.RESPONSE.value:{
        "desc":"对前述问题或请求的回复",
        "example":"今晚发货"
    },
    ACTION.CONFIRM.value:{
        "desc":"确认对方的问题或者请求",
        "example":"是的，我们在搞价格促销"
    },
    ACTION.EXPLAIN.value:{
        "desc":"针对问题或者事实给出解释",
        "example":"节假日成交量大，所以不能及时发货"
    },
    ACTION.ACCEPT.value:{
        "desc":"接受\同意对方的提议或者告知",
        "example":"好的"
    },
    ACTION.UNKNOWN.value:{
        "desc":"闲谈等其他行为",
        "example":"/:^$^"
    }
}


def action_transform(action):
    if action=="Greeting": return ACTION.GREETING.value
    elif action=="Ask": return ACTION.ASK.value
    elif action=="Request": return ACTION.REQUEST.value
    elif action=="Inform": return ACTION.INFORM.value
    elif action=="Response": return ACTION.RESPONSE.value
    elif action=="Confirm": return ACTION.CONFIRM.value
    elif action=="Explain": return ACTION.EXPLAIN.value
    elif action=="Accept": return ACTION.ACCEPT.value
    else:
        return ACTION.UNKNOWN.value

class TBdataset:

    def __init__(self,window_size,src,model_type='qwen',
                 sample_mode=1,
                 save2jsonl=True,
                 only_user=True,
                 summarize=False,
                 completion=False,
                 annoymous_speaker=False,
                 load_from_json=False) -> None:

        #window_size指定了每次选取的对话历史窗口，当为-1时选择全部，当window_size大于该段对话轮数则选择全部轮数
        self.window_size = window_size
        self.sample_mode = sample_mode
        self.summerize = summarize
        self.completion = completion
        self.model_type = model_type
        self.annoymous_speaker = annoymous_speaker
        self.only_user = only_user

        if load_from_json:
            with open(src,"r") as reader:
                self.test_set = json.load(reader)
            return

        self.dialogs = self.load_data(src,annoymous_speaker=self.annoymous_speaker)

        self.total_len_stat = {}

        for dialog in self.dialogs:
            length = len(dialog['content'])
            self.total_len_stat[length] = self.total_len_stat.get(length,0) + 1

        #合并连续的同一说话者的句子
        self.combine_continuous_sentence()

        #根据窗口大小调整数据集
        self.dialogs = self.adjust_to_window(config['dataset']['save_dir'],
                                             save2jsonl=save2jsonl,
                                             only_user=only_user,
                                             summarize=self.summerize,
                                             completion=self.completion)
        self.train_set = None
        self.test_set = None
    
    #三种模式:
        # 1、取一个上下文窗口，比如当窗口大小为3时，句子ID 123为一个样本，234为一个样本...。这种取样方式的样本长度固定。当窗口大小为-1时，选择全部样本
        # 2、对于每一个大小的上下文窗口，比如当窗口大小为3时，样本为 1、12、123，2、23、234、3、34、345...。这种取样方式的样本量为原来的size倍，对few-shots的例子数不敏感。
        # 3、将数据集中显式指定了predist和condist关系的具有关联信息的句子单独抽离出来。
    def adjust_to_window(self,
                        save_dir,
                        save2jsonl=True,
                        only_user=True,
                        summarize=True,
                        completion=True):
        if self.dialogs == None:
            logger.error("dialog为空")
            return None
        
        adjusted_content = []
        #每段对话
        sampleID = 0

        if self.sample_mode==1:
            for dialog in tqdm(self.dialogs):
                sentences = dialog['content']
                sent_num = len(sentences)
                chat_history = ""
                completion_utterance = ""

                if not only_user:
                    #对所有对话历史作采样 
                    if self.window_size==-1:
                        # if (summarize or completion) and sent_num>1:
                        #     speaker = sentences[-1]['speakerID']
                        #     if speaker==1: other_speaker = 2
                        #     else: other_speaker = 1
                        #     role_dict = {speaker:'小明',other_speaker:'小红'}
                        #     history = ""
                        #     for idx,sent in enumerate(sentences[:-1]):
                        #         history += role_dict[sent['speakerID']]+":"+sent['sentence']
                        #         if idx != len(sentences)-2:
                        #             history += "\n"

                        #     #对除了最后一句话外的对话历史做摘要
                        #     if summarize:
                        #         chat_history = chat(SUMMARIZE_HISTORY_TEMPLATE.format(history=history),model_type=self.model_type)
                            
                        #     #根据历史对最后一句话作上下文补全
                        #     if completion:
                        #         completion_utterance = chat(COMPLETION_TEMPLATE.format(history=history,sentence=sentences[-1]['sentence']),model_type=self.model_type)

                        adjusted_content.append({"sampleID":sampleID,
                                                "dialogID":dialog['dialogID'],
                                                "content":sentences,
                                                "history_summarization":chat_history,
                                                "completion":completion_utterance})
                        sampleID += 1
                        continue
                    
                    #对不足窗口大小的对话历史做采样（则对全部轮次做采样）
                    if self.window_size >= sent_num:

                        if (summarize or completion) and sent_num>1:
                            speaker = sentences[-1]['speakerID']
                            if speaker==1: other_speaker = 2
                            else: other_speaker = 1
                            role_dict = {speaker:'小明',other_speaker:'小红'}
                            history = ""
                            for idx,sent in enumerate(sentences[:-1]):
                                history += role_dict[sent['speakerID']]+":"+sent['sentence']
                                if idx != len(sentences)-2:
                                    history += "\n"


                            #对除了最后一句话外的对话历史做摘要
                            if summarize:
                                chat_history = chat(SUMMARIZE_HISTORY_TEMPLATE.format(history=history),model_type=self.model_type)
                            
                            #根据历史对最后一句话作上下文补全
                            if completion:
                                completion_utterance = chat(COMPLETION_TEMPLATE.format(history=history,sentence=sentences[-1]['sentence']),model_type=self.model_type)


                        adjusted_content.append({"sampleID":sampleID,
                                                "dialogID":dialog['dialogID'],
                                                "content":sentences,
                                                "history_summarization":chat_history,
                                                "completion":completion_utterance})
                        sampleID += 1

                    #对大于窗口大小的对话历史做采样
                    else:
                        for i in range(0,sent_num-self.window_size+1):

                            if (summarize or completion) and sent_num>1:
                                speaker = sentences[-1]['speakerID']
                                if speaker==1: other_speaker = 2
                                else: other_speaker = 1
                                role_dict = {speaker:'小明',other_speaker:'小红'}
                                history = ""
                                for idx,sent in enumerate(sentences[i:i+self.window_size-1]):
                                    history += role_dict[sent['speakerID']]+":"+sent['sentence']

                                    if idx != self.window_size-2:
                                        history += "\n"
                                    

                                if summarize:
                                    chat_history = chat(SUMMARIZE_HISTORY_TEMPLATE.format(history=history),model_type=self.model_type)
                                
                                if completion:
                                    completion_utterance = chat(COMPLETION_TEMPLATE.format(history=history,sentence=sentences[i+self.window_size-1]['sentence']),model_type=self.model_type)

                            adjusted_content.append({"sampleID":sampleID,
                                                    "dialogID":dialog['dialogID'],
                                                    "content":sentences[i:i+self.window_size],
                                                    "history_summarization":chat_history,
                                                    "completion":completion_utterance})
                            sampleID += 1
                else:
                    #只对用户的对话历史做采样
                    for i,sent in enumerate(sentences):
                        if sent['speakerID'] == Role.System:
                            continue

                        user_utterance = sentences[i]
                        if self.window_size==-1:
                            history = sentences[:i]
                        else:
                            history = sentences[i-self.window_size:i] if i>=self.window_size else sentences[:i]

                        adjusted_content.append({"sampleID":sampleID,
                                                "dialogID":dialog['dialogID'],
                                                "target":user_utterance,
                                                "history":history}) 
                        sampleID += 1

        elif self.sample_mode==2:
            raise NotImplementedError("wrong")
        elif self.sample_mode==3:
            raise NotImplementedError("Wrong")
        
        if save2jsonl:

            save_path = Path(save_dir) / Path("tb_dataset_mode"+str(self.sample_mode)+"_size"+str(self.window_size)+"_only_user.jsonl") if only_user \
                else Path("tb_dataset_mode"+str(self.sample_mode)+"_size"+str(self.window_size)+".jsonl")
            with jsonlines.open(save_path,"w") as writer:
                for cont in adjusted_content:
                    writer.write(cont)

        return adjusted_content

    def load_data(self,src,annoymous_speaker=False):
        dialog = []
        single_dialog = {}
        with jsonlines.open(src,"r") as reader:
            dialog_id_temp = -1
            for obj in reader:

                if obj['sentence'].strip() == "":
                    continue

                if obj['dialogID'] == dialog_id_temp:
                    if annoymous_speaker:
                        speakerID = 1 if obj['speaker']=='SELLER' else 2
                    else:
                        speakerID = obj['speaker']
                    single_dialog['content'].append({'sentID':obj['sentID'],
                                                     "speakerID":speakerID,
                                                     'sentence':obj['sentence'],
                                                     'intent':action_transform(obj['action'])})
            
                else:
                    if dialog_id_temp != -1: 
                        dialog.append(single_dialog)
                        single_dialog = {}

                    dialog_id_temp = obj['dialogID']
                    single_dialog['dialogID'] = dialog_id_temp
                    single_dialog['content'] = []

                    if annoymous_speaker:
                        speakerID = 1 if obj['speaker']=='SELLER' else 2
                    else:
                        speakerID = obj['speaker']
                    single_dialog['content'].append({'sentID':obj['sentID'],
                                                     "speakerID":speakerID,
                                                     'sentence':obj['sentence'],
                                                     'intent':action_transform(obj['action'])})
            
            if len(single_dialog)!=0:
                dialog.append(single_dialog)

        return dialog

    def combine_continuous_sentence(self):
        if not self.dialogs:
            logger.error("数据集尚未建立")
            return None
        
        for dialog in self.dialogs:
            sentences = dialog['content']
            merged_list = [sentences[0]]
            
            for current_sentence in sentences[1:]:
                last_dict = merged_list[-1]
                
                if current_sentence['speakerID'] == last_dict['speakerID']:
                    last_dict['sentence'] += sentence_segment + current_sentence['sentence']
                    last_dict['intent'] += intent_segment + current_sentence['intent']
                else:
                    merged_list.append(current_sentence)
            
            dialog['content'] = merged_list


    def split_train_test(self,ratio=0.8,save_dir=False,save2jsonl=False):
        if self.dialogs == None:
            logger.error("数据集尚未建立")
        random.shuffle(self.dialogs)
        split_index = int(len(self.dialogs)*ratio)
        self.train_set = self.dialogs[:split_index]
        self.test_set = self.dialogs[split_index:]

        if save2jsonl:
            save_path = Path(save_dir) / Path("tb_trainset_mode"+str(self.sample_mode)+"_size"+str(self.window_size)+".jsonl")
            with jsonlines.open(save_path,"w") as writer:
                for cont in self.train_set:
                    writer.write(cont)

            save_path = Path(save_dir) / Path("tb_testset_mode"+str(self.sample_mode)+"_size"+str(self.window_size)+".jsonl")
            with jsonlines.open(save_path,"w") as writer:
                for cont in self.test_set:
                    writer.write(cont)

    def construct_sft_trainset(self):
        if self.annoymous_speaker:
            logger.error("目前只有非匿名化的说话者才能构建SFT训练集")
            return None
        if self.window_size != -1:
            logger.error("目前只有窗口大小为-1时才能构建SFT训练集")
            return None
        if not self.train_set:
            logger.error("训练集尚未建立")
            return None
        
        sft_trainset=[]
        for sample in self.train_set:
            sft_sample = {}

            history=""
            for i in range(0,len(sample['content'])-1):
                history += HISTORY_WITH_LABEL_TEMPLATE.format(ROLE=str(i+1)+ " "+sample['content'][i]['speakerID'],
                                                SENTENCE=sample['content'][i]['sentence'],
                                                INTENT=sample['content'][i]['intent'])
            cur_role = sample['content'][-1]['speakerID']
            other_role = "BUYER" if cur_role=="SELLER" else "SELLER"

            candidate_intent_str = ""
            for idx,intent in enumerate(cand_intent_cn):

                candidate_intent_str += "- {intent}{description}。\n".format(intent=intent,
                                                                                    description="\t含义：{}".format(intent_description[intent]['desc']))

            examples_str = "## Examples\n"
            for idx,intent in enumerate(cand_intent_cn):
                examples_str += "### Example{}\n".format(idx+1)
                examples_str += "句子：{}\t意图:{}".format(intent_description[intent]['example'],intent)

                if idx != len(cand_intent_cn)-1:
                    examples_str += "\n"
                    
            candidate_intent_str += examples_str

            inst = WITH_HISTORY_INTENT_RECOGNITION_TEMPLATE.format(history=history,
                                                                   cur_role=cur_role,
                                                                   other_role=other_role,
                                                                   candidate_intent = candidate_intent_str,
                                                                   sentence=sample['content'][-1]['sentence'])
            sft_sample['instruction'] = inst
            sft_sample['input'] = ""
            sft_sample['output'] = sample['content'][-1]['intent']
            sft_sample['system'] = DEFAULT_SYSTEM_PROMPT
            sft_trainset.append(sft_sample)
        
        return sft_trainset

    def statiscal(self):
        #对训练集和数据集统计每段对话的长度分布，键为不同的长度，值为该长度的对话数
        train_len_stat = {}
        test_len_stat = {}

        for dialog in self.train_set:
            length = len(dialog['content'])
            train_len_stat[length] = train_len_stat.get(length,0) + 1
        
        for dialog in self.test_set:
            length = len(dialog['content'])
            test_len_stat[length] = test_len_stat.get(length,0) + 1
        
        return self.total_len_stat,train_len_stat,test_len_stat

import json
import numpy as np

def cal_euclidean_distance(dict1,dict2):
    # 存储平方差的和
    sum_of_squares = 0

    # 遍历字典的键
    for key in dict1:
        # 获取对应键的值
        value1 = dict1[key]
        value2 = dict2[key]

        # 计算差值的平方并累加
        square_diff = (value1 - value2) ** 2
        sum_of_squares += square_diff

    # 计算欧氏距离
    euclidean_distance = np.sqrt(sum_of_squares)

    return euclidean_distance

def TB_role_intent_analysis():
    with jsonlines.open(data_path,"r") as reader:
        role_intent = {}
        for obj in reader:
            if obj['sentence'].strip() == "":
                continue
            if obj['speaker'] not in role_intent:
                role_intent[obj['speaker']] = {}
            if obj['action'] not in role_intent[obj['speaker']]:
                role_intent[obj['speaker']][obj['action']] = 1
            else:
                role_intent[obj['speaker']][obj['action']] += 1

        with open("/home/hzl/work/TaskChatChainv3/intent_recognition/tb_data/processed/role_intent.json","w") as writer:
            json.dump(role_intent,writer,ensure_ascii=False,indent=4)
        
        print(cal_euclidean_distance(role_intent['BUYER'],role_intent['SELLER']))

import random
def TB_same_role_intent_analysis():
    buyer = []
    seller = []
    with jsonlines.open(data_path,"r") as reader:
        for obj in reader:
            if obj['sentence'].strip() == "": continue
            if obj['speaker'] == 'BUYER':
                buyer.append((obj['dialogID'],obj['sentID'],obj['action']))
            else:
                seller.append((obj['dialogID'],obj['sentID'],obj['action']))
    
    sample_size = 5000

    for i in range(10):
        print("buyer第{}次采样".format(i+1))
        buyer_intent_dict1 = {}
        buyer_intent_dict2 = {}
        buyer_sample1 = random.sample(buyer,sample_size)
        buyer_sample2 = random.sample([item for item in buyer if item not in buyer_sample1],sample_size)

        for sam in buyer_sample1:
            if sam[2] not in buyer_intent_dict1:
                buyer_intent_dict1[sam[2]] = 1
            else:
                buyer_intent_dict1[sam[2]] += 1
        for intent in intent_lst:
            if intent not in buyer_intent_dict1:
                buyer_intent_dict1[intent] = 0

        for sam in buyer_sample2:
            if sam[2] not in buyer_intent_dict2:
                buyer_intent_dict2[sam[2]] = 1
            else:
                buyer_intent_dict2[sam[2]] += 1
        for intent in intent_lst:
            if intent not in buyer_intent_dict2:
                buyer_intent_dict2[intent] = 0

        print("第一个buyer意图分布",buyer_intent_dict1)
        print("第二个buyer意图分布",buyer_intent_dict2)
        print(cal_euclidean_distance(buyer_intent_dict1,buyer_intent_dict2))

        print("seller第{}次采样".format(i+1))
        seller_intent_dict1 = {}
        seller_intent_dict2 = {}
        seller_sample1 = random.sample(seller,sample_size)
        seller_sample2 = random.sample([item for item in seller if item not in seller_sample1],sample_size)

        for sam in seller_sample1:
            if sam[2] not in seller_intent_dict1:
                seller_intent_dict1[sam[2]] = 1
            else:
                seller_intent_dict1[sam[2]] += 1
        
        for sam in seller_sample2:
            if sam[2] not in seller_intent_dict2:
                seller_intent_dict2[sam[2]] = 1
            else:
                seller_intent_dict2[sam[2]] += 1
        
        print("第一个seller意图分布",seller_intent_dict1)
        print("第二个seller意图分布",seller_intent_dict2)
        print(cal_euclidean_distance(seller_intent_dict1,seller_intent_dict2))

        print("buyer、seller欧式距离",cal_euclidean_distance(buyer_intent_dict1,seller_intent_dict2))


if __name__ == "__main__":
    # dataset = TBdataset(window_size=-1,sample_mode=1,src=data_path,annoymous_speaker=False,summarize=False,completion=False)
    # dataset.split_train_test(ratio=0.8)
    # sft_set = dataset.construct_sft_trainset()
    # with open("/home/hzl/work/TaskChatChainv3/intent_recognition/tb_data/processed/tb_sft_trainset.json","w") as writer:
    #     json.dump(sft_set,writer,ensure_ascii=False,indent=4)
    # with open("/home/hzl/work/TaskChatChainv3/intent_recognition/tb_data/processed/tb_sft_testset.json","w") as writer:
    #     json.dump(dataset.test_set,writer,ensure_ascii=False,indent=4)
    # print(dataset.test_set)
    # # TB_role_intent_analysis()
    # # TB_same_role_intent_analysis()

    # #加载数据集
    # data_path = config['dataset']['src']
    # dataset = TBdataset(window_size=-1,
    #                     src=data_path,
    #                     only_user=True)


    import json
    data_path = "/disk0/fin_group/hzl/TaskChatChainv3/data/tb_data/processed/tb_dataset_mode1_size-1_only_user_renumber.jsonl"
    save_path = "/disk0/fin_group/hzl/TaskChatChainv3/data/tb_data/processed/tb_dataset_mode1_size-1_only_user_renumber_v2.jsonl"
    with open(save_path, 'w',encoding='utf-8') as f:
        for item in jsonlines.open(data_path):
            #将列表倒序
            item['history'] = item['history'][::-1]
            f.write(json.dumps(item,ensure_ascii=False)+'\n')