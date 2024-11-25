import yaml

def read_yml(data_path):
    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)
    return data