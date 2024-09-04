import sys
from pathlib import Path
import json
import jsonlines
from tqdm import tqdm
from datetime import datetime
sys.path.append("/disk0/fin_group/hzl/TaskChatChainv3")
from loguru import logger
from api.vllm_model import chat,chat_prob_classify
from data.tb_data.dataset import TBdataset,ACTION,intent_description,intent_cn2en,cand_intent_cn
from core.output_parser import IntentParser
from api.openai_api import chat_gpt

from intent_recognition.template.IR_template import HISTORY_WITH_LABEL_TEMPLATE,HISTORY_NO_LABEL_TEMPLATE,NO_HISTORY_INTENT_RECOGNITION_TEMPLATE,WITH_HISTORY_INTENT_RECOGNITION_TEMPLATE,CROSSWOZ_NLU_TEMPLATE,STEP_INTENT_RECOGNITION_TEMPLATE,STEP_EXAMPLE

CAP_lst = ["A","B","C","D","E","F","G","H","I"]


def test_on_tb(model_type,
         sample_mode,
         window_size,
         ratio,
         source_data,
         prob_mode=True,
         few_shots=True,
         history_with_label=False,
         history_summarize=False,
         query_completion=False,
         load_from_json=False,
         log_case_dir = "intent_recognition/logs"):
    """
    这个函数用于测试在TB数据集上的意图识别任务，前提是对应model_type的llm已经通过vllm的方式运行在了服务器上。

    Args:
        model_type: str,模型的类型，支持'cnllama3','cnllama','qwen','gpt'
        sample_mode: int,采样模式，1表示根据窗口大小进行采样，将对话切分成多个相同大小的窗口，对话长度不足则全采样
        window_size: int,窗口大小，当-1时表示全采样。当1时表示没有对话历史
        ratio: float,训练集和测试集的比例
        source_data: str,数据集的路径
        prob_mode: bool,是否使用判别式
        few_shots: bool,是否给意图加例子
        history_with_label: bool,是否给每个历史中加入意图识别的标签；仅当history_summarize为False时有效
        history_summarize: bool,是否对对话历史进行摘要
        query_completion: bool,是否对需要判断的句子根据上下文进行补全
        log_case_dir: str,日志的保存根路径
    
    """
    save_dir = Path.cwd() / Path(log_case_dir) / "{}_sampleMode{}_windowSize{}_ratio{}_probMode{}_labelHistory{}_sumHistory{}_queryCom{}_{}".format(model_type,
                                                                                                sample_mode,
                                                                                                window_size,
                                                                                                ratio,
                                                                                                1 if prob_mode else 0,
                                                                                                1 if history_with_label else 0,
                                                                                                1 if history_summarize else 0,
                                                                                                1 if query_completion else 0,
                                                                                                datetime.now().strftime("%m%d_%H:%M"))
    if not save_dir.exists():
        save_dir.mkdir()

    if load_from_json:
        mydataset = TBdataset(window_size=window_size,sample_mode=sample_mode,src=source_data,load_from_json=True)
    else:
        mydataset = TBdataset(window_size=window_size,sample_mode=sample_mode,src=source_data,save2jsonl=False,annoymous_speaker=False)
        mydataset.split_train_test(ratio=ratio,save2jsonl=False)

    test_set_statistical = {intent:0  for intent in cand_intent_cn}
    res_statistical = {intent:0  for intent in cand_intent_cn}

    if prob_mode:
        res_statistical['FALSE'] = 0
        Cap2Intent = {}
        Intent2Cap = {}
        candidate_intent_str = ""
        for idx,intent in enumerate(cand_intent_cn):
            candidate_intent_str += CAP_lst[idx] + ":" + intent+"\n"
            Cap2Intent[CAP_lst[idx]] =  intent
            Intent2Cap[intent] = CAP_lst[idx]
    else:
        candidate_intent_str = ""
        for idx,intent in enumerate(cand_intent_cn):
            # candidate_intent_str += "- {intent}{description}。{example}\n".format(intent=intent,
            #                                                                       description="\t含义：{}".format(intent_description[intent]['desc']),
            #                                                                       example="例子：{}".format(intent_description[intent]['example']) if one_shot else "")

            candidate_intent_str += "- {intent}{description}。\n".format(intent=intent,
                                                                                  description="\t含义：{}".format(intent_description[intent]['desc']))

        if few_shots:
            examples_str = "## Examples\n"
            for idx,intent in enumerate(cand_intent_cn):
                examples_str += "### Example{}\n".format(idx+1)
                examples_str += "句子：{}\t意图:{}".format(intent_description[intent]['example'],intent)

                if idx != len(cand_intent_cn)-1:
                    examples_str += "\n"
                
            candidate_intent_str += examples_str

    llm_res = []
    positive_case = []
    negative_case = []
    unrecog_case = []
    y_true = []
    y_predict = []

    if history_summarize: history_with_label = False

    for sample in tqdm(mydataset.test_set):
        test_set_statistical[sample['content'][-1]['intent']] += 1
        cur_sent = sample['content'][-1]['sentence']
        cur_role = sample['content'][-1]['speakerID']

        if window_size>1 or window_size==-1:
            history = ""
            for i in range(0,len(sample['content'])-1):
                if history_with_label:
                    history += HISTORY_WITH_LABEL_TEMPLATE.format(ROLE=sample['content'][i]['speakerID'],
                                                    SENTENCE=sample['content'][i]['sentence'],
                                                    INTENT=Intent2Cap[sample['content'][i]['intent']] if prob_mode else sample['content'][i]['intent'])
                else:
                    history += HISTORY_NO_LABEL_TEMPLATE.format(ROLE=sample['content'][i]['speakerID'],
                                SENTENCE=sample['content'][i]['sentence'])
                    
            # if history_summarize:
            #     summarize_inst = SUMMARIZE_HISTORY_TEMPLATE.format(history=history)
            #     history = chat(summarize_inst)
            
            inst = WITH_HISTORY_INTENT_RECOGNITION_TEMPLATE.format(candidate_intent=candidate_intent_str,
                                                                   sentence=cur_sent,
                                                                   history=history,
                                                                   cur_role=cur_role,
                                                                   other_role="SELLER" if cur_role=="BUYER" else "BUYER",)
        else: #无对话历史
            inst = NO_HISTORY_INTENT_RECOGNITION_TEMPLATE.format(candidate_intent=candidate_intent_str,sentence=cur_sent)

        if prob_mode:
            llm_intent_cap = chat_prob_classify(utterance=inst,tokens_lst=CAP_lst,model_type=model_type)

            if llm_intent_cap == None:
                #没有预测出一个明确的结果
                llm_res.append("FALSE")
                res_statistical['FALSE'] += 1
                unrecog_case.append({"Prompt":inst,"predict":"None","target":sample['content'][-1]['intent']})
                # logger.warning("没有回答选项")
            else:
                llm_res.append(llm_intent_cap)
                res_statistical[Cap2Intent[llm_intent_cap]] += 1

                if  Cap2Intent[llm_intent_cap] != sample['content'][-1]['intent']:
                    # logger.warning("不一致的意图："+Cap2Intent[llm_intent_cap]+","+sample['content'][-1]['intent'])
                    negative_case.append({"Prompt":inst,"predict":Cap2Intent[llm_intent_cap],"target":sample['content'][-1]['intent']})
                else:
                    # logger.info("一致的意图："+Cap2Intent[llm_intent_cap])
                    positive_case.append({"Prompt":inst,"predict":Cap2Intent[llm_intent_cap],"target":sample['content'][-1]['intent']})
        else:

            #gpt系列的模型使用该接口
            if model_type.startswith("gpt"):
                llm_intent = chat_gpt(utterance=inst)
                logger.info(llm_intent)
            else: #部署在vllm上的模型使用该接口
                llm_intent = chat(utterance=inst,model_type=model_type)

            #对结果进行解析（后处理）
            parser = IntentParser(llm_intent)
            llm_intent = parser.parse_intent() 

            llm_res.append(llm_intent)

            
            if  llm_intent!= sample['content'][-1]['intent']:
                # logger.warning("不一致的意图："+llm_intent+","+sample['content'][-1]['intent'])
                if llm_intent in cand_intent_cn:
                    y_predict.append(cand_intent_cn.index(llm_intent))
                    y_true.append(cand_intent_cn.index(sample['content'][-1]['intent']))
                res_statistical[llm_intent] = res_statistical.get(llm_intent,0) + 1
                negative_case.append({"Prompt":inst,"predict":llm_intent,"target":sample['content'][-1]['intent']})
            else:
                y_predict.append(cand_intent_cn.index(llm_intent))
                y_true.append(cand_intent_cn.index(sample['content'][-1]['intent']))
                # logger.info("一致的意图："+llm_intent)
                res_statistical[llm_intent] = res_statistical.get(llm_intent,0) + 1
                positive_case.append({"Prompt":inst,"predict":llm_intent,"target":sample['content'][-1]['intent']})


    if prob_mode:
        accuracy = sum( a==b for a,b in zip(llm_res,[Intent2Cap[sample['content'][-1]['intent']] for sample in mydataset.test_set])) / len(mydataset.test_set)
    else:
        accuracy = sum( a==b for a,b in zip(llm_res,[sample['content'][-1]['intent'] for sample in mydataset.test_set])) / len(mydataset.test_set)
    
     #混淆矩阵
    cm = confusion_matrix(y_true, y_predict)
    plot_confusion_matrix(cm, [intent_cn2en[intent] for intent in cand_intent_cn], "confusion_matrix", save_dir)

    #准确率
    accuracy_sk = accuracy_score(y_true,y_predict)
    accuracy_all = len(positive_case) / len(mydataset.test_set)

    #F1 score
    f1_score_macro = f1_score(y_true, y_predict, average='macro')

    logger.info("准确率:{}",accuracy)
    logger.info("测试集的意图分布:{}",json.dumps(test_set_statistical,ensure_ascii=False))
    logger.info("预测的结果分布:{}",json.dumps(res_statistical,ensure_ascii=False))

    if prob_mode:
        with jsonlines.open(Path(save_dir) / "unrecog_case.jsonl","w") as writer:
            for obj in unrecog_case:
                writer.write(obj)
    
    with jsonlines.open(Path(save_dir) / "pos_case.jsonl","w") as writer:
        for obj in positive_case:
            writer.write(obj)

    with jsonlines.open(Path(save_dir) / "neg_case.jsonl","w") as writer:
        for obj in negative_case:
            writer.write(obj)
    
    with open(Path(save_dir) / "analysis.txt","w") as f:
        f.write("测试集大小:{}\n".format(len(mydataset.test_set)))
        all_stat,train_stat,test_stat = mydataset.statiscal()
        f.write("原数据集中对话长度分布:{}\n".format(json.dumps(all_stat,ensure_ascii=False)))
        f.write("测试集中对话长度分布:{}\n".format(json.dumps(test_stat,ensure_ascii=False)))
        f.write("测试集的意图分布:{}\n".format(json.dumps(test_set_statistical,ensure_ascii=False)))
        f.write("预测的意图分布:{}\n".format(json.dumps(res_statistical,ensure_ascii=False)))
        f.write("成功解析出意图的话数量为{}\nAccuracy（分类正确/可解析的样本数）:{}\nAccuracy（分类正确/总共识别的样本数）:{}\n".format(len(y_predict),accuracy_sk,accuracy_all))
        f.write("F1_score_macro：{}\n".format(f1_score_macro))

        pos_case_statistic =  {}
        for case in positive_case:
            pos_case_statistic[case['predict']] = pos_case_statistic.get(case['predict'],0) + 1
        f.write("其中分类正确的部分，意图分布为:{}\n".format(json.dumps(pos_case_statistic,ensure_ascii=False)))

        neg_case_statistic =  {}
        for case in negative_case:
            neg_case_statistic[case['predict']] = neg_case_statistic.get(case['predict'],0) + 1
        f.write("其中分类错误的部分，意图分布为:{}\n".format(json.dumps(neg_case_statistic,ensure_ascii=False)))

        if prob_mode:
            unrecog_case_statistic = {}
            for case in unrecog_case:
                unrecog_case_statistic[case['target']] = unrecog_case_statistic.get(case['target'],0) + 1
            f.write("其中未产生选项的概率为{}，其意图分布为:{}\n".format(len(unrecog_case)/len(mydataset.test_set),json.dumps(unrecog_case_statistic,ensure_ascii=False)))

def construct_crosswoz_inst(messages):
    single_turn_template = "{role}：“{content}”"
    role = ['小明','小红']
    history=""
    for idx,message in enumerate(messages):
        single_turn = single_turn_template.format(role=role[idx%2],content=message)
        history += single_turn + "\n"
    
    inst = CROSSWOZ_NLU_TEMPLATE.format(history=history)
    return inst

def test_on_crosswoz(model_type='gpt',test_path='/home/hzl/work/TaskChatChainv3/other/CrossWOZ/data/crosswoz/test.json'):
    for _,dia_content in json.load(open(test_path,mode='r')).items():
        messages = []
        for turn in dia_content['messages']:
            messages.append(turn['content'])
            inst = construct_crosswoz_inst(messages)
            print(chat_gpt(utterance=inst))
        
        break

from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,accuracy_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans','KaiTi']
import numpy as np
def plot_confusion_matrix(cm, labels_name, title, save_dir, colorbar=False, cmap=None):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)    # 在特定的窗口上显示图像
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.title(title)    # 图像标题
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    plt.savefig(Path(save_dir) / Path(title+".png"))


def test_on_tb_relation(model_type,
         window_size,
         ratio,
         source_data,
         top_n,
         few_shots=True,
         log_case_dir = "intent_recognition/logs"):
    """
    这个函数用于测试用关系牵引的方法在TB数据集上的意图识别任务，前提是对应model_type的llm已经通过vllm的方式运行在了服务器上。

    Args:
        model_type: str,模型的类型，支持'cnllama3','cnllama','qwen','gpt'
        window_size: int,窗口大小，当-1时表示全采样。当1时表示没有对话历史
        ratio: float,训练集和测试集的比例
        source_data: str,数据集的路径
        few_shots: bool,是否给意图加例子
        history_with_label: bool,是否给每个历史中加入意图识别的标签
        top_n: int,关系牵引的时候，需要选择的最相近的句子的数量
    
    """
    save_dir = Path('/home/hzl/work/TaskChatChainv3') / Path(log_case_dir) / "{}_windowSize{}_ratio{}_topn{}_labelHistory{}_fewshot{}_{}".format(model_type,
                                                                                                window_size,
                                                                                                ratio,
                                                                                                top_n,
                                                                                                1 if history_with_label else 0,
                                                                                                1 if few_shots else 0,
                                                                                                datetime.now().strftime("%m%d_%H:%M"))
    if not save_dir.exists():
        save_dir.mkdir()
    
    mydataset = TBdataset(window_size=window_size,sample_mode=1,src=source_data,save2jsonl=False,annoymous_speaker=False)
    mydataset.split_train_test(ratio=ratio,save2jsonl=False)

    candidate_intent_str = ""
    for idx,intent in enumerate(cand_intent_cn):
        candidate_intent_str += "- {intent}{description}。\n".format(intent=intent,
                                                                                description="\t含义：{}".format(intent_description[intent]['desc']))

    buyer_recog_cnt = 0
    seller_recog_cnt = 0
    negative_case = []
    positive_case = []
    bad_case_cnt = 0
    y_true = []
    y_predict = []
    extra_intent_set = set()

    for sample in tqdm(mydataset.test_set):
        cur_sent = sample['content'][-1]['sentence']
        cur_role = sample['content'][-1]['speakerID']

        if cur_role == "SELLER":
            seller_recog_cnt+=1
        else:
            buyer_recog_cnt+=1

        if window_size>1 or window_size==-1:
            history = ""
            for i in range(0,len(sample['content'])-1):
                if history_with_label:
                    history += HISTORY_WITH_LABEL_TEMPLATE.format(ROLE=str(i+1)+ " "+sample['content'][i]['speakerID'],
                                                    SENTENCE=sample['content'][i]['sentence'],
                                                    INTENT=sample['content'][i]['intent'])
                else:
                    history += HISTORY_NO_LABEL_TEMPLATE.format(sample['content'][i]['speakerID'],
                                SENTENCE=sample['content'][i]['sentence'])
                    
            # if history_summarize:
            #     summarize_inst = SUMMARIZE_HISTORY_TEMPLATE.format(history=history)
            #     history = chat(summarize_inst)
            if few_shots:
                example = STEP_EXAMPLE
            else:
                example = ""
            inst = STEP_INTENT_RECOGNITION_TEMPLATE.format(candidate_intent=candidate_intent_str,
                                                           sentence=cur_sent,history=history,
                                                           top_n=top_n,
                                                           Example=example,
                                                           role=cur_role,
                                                           roles="与".join(["BUYER","SELLER"]))
        else: #无对话历史
            logger.error("window_size必须大于1或为-1")

        #gpt系列的模型使用该接口
        if model_type.startswith("gpt"):
            llm_output = chat_gpt(utterance=inst)
            logger.info(llm_output)
        else: #部署在vllm上的模型使用该接口
            llm_output = chat(utterance=inst,model_type=model_type)
            logger.info(llm_output)


        # #对结果进行解析（后处理)
        try:
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[-1].split("```")[0]
            logger.info(llm_output)
            llm_output_json = json.loads(llm_output)
        except Exception as e:
            logger.error("解析json失败")
            bad_case_cnt+=1
            continue
        
        try:
            llm_intent = llm_output_json['step2']['query_intent']
        except Exception as e:
            logger.error("json格式不对导致解析意图失败")
            bad_case_cnt+=1
            continue
        
        #后处理
        parser = IntentParser(llm_intent)
        llm_intent = parser.parse_intent() 

        if llm_intent not in cand_intent_cn:
            logger.error("意图无法解析成可选意图中的一个")
            extra_intent_set.add(llm_intent)
            continue
        
        y_predict.append(cand_intent_cn.index(llm_intent))
        y_true.append(cand_intent_cn.index(sample['content'][-1]['intent']))
        if  llm_intent!= sample['content'][-1]['intent']:
            negative_case.append({"Prompt":inst,"llm_output":llm_output,"predict":llm_intent,"target":sample['content'][-1]['intent']})
        else:
            positive_case.append({"Prompt":inst,"llm_output":llm_output,"predict":llm_intent,"target":sample['content'][-1]['intent']})
            
    #评价指标

    #混淆矩阵
    cm = confusion_matrix(y_true, y_predict)
    plot_confusion_matrix(cm, [intent_cn2en[intent] for intent in cand_intent_cn], "confusion_matrix", save_dir)

    #准确率
    accuracy = len(positive_case) / (len(positive_case) + len(negative_case))
    accuracy_sk = accuracy_score(y_true,y_predict)
    accuracy_all = len(positive_case) / (buyer_recog_cnt+seller_recog_cnt)

    #精确率
    precision_macro = precision_score(y_true, y_predict, average='macro')
    precision_micro = precision_score(y_true, y_predict, average='micro')
    precision_weighted = precision_score(y_true, y_predict, average='weighted')

    #召回率
    recall_macro = recall_score(y_true, y_predict, average='macro')
    recall_micro = recall_score(y_true, y_predict, average='micro')
    recall_weighted = recall_score(y_true, y_predict, average='weighted')

    #F1 score
    f1_score_macro = f1_score(y_true, y_predict, average='macro')
    f1_score_micro = f1_score(y_true, y_predict, average='micro')
    f1_score_weighted = f1_score(y_true, y_predict, average='weighted')

    with jsonlines.open(Path(save_dir) / "pos_case.jsonl","w") as writer:
        for obj in positive_case:
            writer.write(obj)

    with jsonlines.open(Path(save_dir) / "neg_case.jsonl","w") as writer:
        for obj in negative_case:
            writer.write(obj)
    
    with open(Path(save_dir)/"analysis.txt","w") as fw:
        fw.write("测试集中总共需要识别的话数量为:{}\n".format(buyer_recog_cnt+seller_recog_cnt))
        fw.write("成功解析出意图的话数量为{}\nAccuracy（分类正确/可解析的样本数）:{}\nAccuracy（分类正确/总共识别的样本数）:{}\n".format(len(positive_case) + len(negative_case),accuracy_sk,accuracy_all))
        fw.write("Precision_macro：{}\n".format(precision_macro))
        fw.write("Recall_macro：{}\n".format(recall_macro))
        fw.write("F1_score_macro：{}\n".format(f1_score_macro))
        fw.write("F1_score_micro：{}\n".format(f1_score_micro))
        fw.write("F1_score_weighted：{}\n".format(f1_score_weighted))
        
        #识别出其他意图的比例
        fw.write("识别出其他意图的样本占全部json成功解析的比例:{}\n".format(len(extra_intent_set)/(buyer_recog_cnt-bad_case_cnt)))
        fw.write("识别出的其他意图为:{}".format(extra_intent_set))

if __name__ == "__main__":
    # test_zero_shot('cnllama3')

    # inst=NO_HISTORY_INTENT_RECOGNITION_TEMPLATE.format(candidate_intent="A:询问\nB:请求\nC:回复\n",sentence="我修改好了")
    # tokens_lst = ["A","B","C"]
    # _,ans = chat_prob_classify(utterance=inst,tokens_lst=tokens_lst)
    # print(ans)

    # test_few_shots('cnllama3')
    window_size = 1
    sample_mode = 1
    ratio = 0.8
    prob_mode = False
    few_shots = True
    history_with_label = True
    history_summarize =  False
    query_completion = False
    source_data = '/home/hzl/work/TaskChatChainv3/intent_recognition/tb_data/processed/tb_dialogue.jsonl'
    source_data = '/home/hzl/work/TaskChatChainv3/intent_recognition/tb_data/processed/tb_sft_testset.json'
    # test_on_tb(model_type='qwen2-7b-instruct_lora_sft',
    #            sample_mode=sample_mode,
    #            window_size=window_size,
    #            ratio=ratio,
    #            source_data=source_data,
    #            prob_mode=prob_mode,
    #            few_shots=few_shots,
    #            history_with_label=history_with_label,
    #            history_summarize=history_summarize,
    #            query_completion=query_completion,
    #            load_from_json=True)
    # test_on_crosswoz()

    # test_on_tb_relation(model_type='qwen2-7b',
    #         window_size=20,
    #         ratio=0.8,
    #         source_data='/home/hzl/work/TaskChatChainv3/intent_recognition/tb_data/processed/tb_dialogue.jsonl',
    #         top_n=1,
    #         few_shots=False,
    #         log_case_dir = "intent_recognition/logs_relation")

    print(chat(utterance="你好，你是谁？",model_type='qwen'))
    
    # json_str = '''
    # {
    #     "step1": [
    #         {
    #             "id": "3",
    #             "intent": "询问",
    #             "reason": "买家在询问卖家下次的批次日期，与query中询问下次批次日期的意图一致。"
    #         },
    #         {
    #             "id": "8",
    #             "intent": "询问",
    #             "reason": "买家在询问错过批次是否就不能购买，与query中询问下次批次日期的意图一致。"
    #         },
    #         {
    #             "id": "13",
    #             "intent": "询问",
    #             "reason": "买家在询问错过优惠码后是否就不能享受优惠，与query中询问下次批次日期的意图一致。"
    #         }
    #     ],
    #     "step2": {
    #         "analysis": "从第一步得到的三个句子中，可以看出买家都在询问关于批次日期的信息。这些句子都涉及到对未来的批次日期进行询问，因此可以推断出买家的意图是询问下次的批次日期。",
    #         "query_intent": "询问"
    #     }
    # }
    # '''

    # print(json.loads(json_str))

