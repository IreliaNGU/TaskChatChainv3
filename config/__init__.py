import yaml

# 假设有一个名为 config.yaml 的 YAML 文件
yaml_file = 'config/ic_config.yaml'

# 读取 YAML 文件
with open(yaml_file, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# 打印读取的配置
print(config)
