import re
import json

def extract_json_from_str(s:str):
    """
    Extract json from string
    将用```json及```包裹的json字符串提取出来,并转换为json格式
    """
    json_str = re.findall(r'```json(.*?)```', s, re.S)
    if not json_str:
        return ""
    try:
        res = json.loads(json_str[0])
        return res
    except:
        return []

def extract_json_list_from_str(s:str):
    """
    Extract json list from string
    将用```json及```包裹的json字符串提取出来,并转换为json的列表。字符串中可能有多个json
    """
    json_str = re.findall(r'```json(.*?)```', s, re.S)
    if not json_str:
        return ""
    try:
        res = [json.loads(i) for i in json_str]
        return res
    except:
        return []

if __name__ == "__main__":
    path = '/home/hzl/work/TaskChatChainv3/intent_recognition/output/gpt4_output'
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read()
    print(extract_json_list_from_str(s)[0][0]['name'])