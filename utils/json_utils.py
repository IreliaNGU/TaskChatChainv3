import json
import orjson
from tqdm import tqdm


def read_json(data_path):
    with open(data_path, 'rb') as f:
        data = orjson.loads(f.read())
    return data


def write_json(data_path, data):
    with open(data_path, 'wb') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False).encode('utf-8'))


def read_jsonl(data_path):
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def write_jsonl(dest, data):
    with open(dest, "r") as f:
        for d in tqdm(data):
            j = orjson.dumps(d).decode('utf-8')
            f.write(j + "\n")