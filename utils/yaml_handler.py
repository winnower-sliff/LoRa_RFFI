# utils/yaml_handler.py
import yaml
import os


def update_nested_yaml_entry(file_path, section_path, entry_data):
    """
    更新YAML嵌套字典中的特定条目（覆盖模式）

    :param file_path: YAML文件路径
    :param section_path: 嵌套路径列表，如 ['models', 'pruned', 'classification_results', 'epoch1']
    :param entry_data: 条目数据
    """
    # 读取现有数据
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = yaml.safe_load(f) or {}
    else:
        existing_data = {}

    # 按路径逐层访问并更新数据
    current = existing_data
    for key in section_path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # 更新最终条目
    current[section_path[-1]] = entry_data

    # 如果是epoch相关的字典，保持键的排序
    if 'epoch' in section_path[-2] if len(section_path) > 1 else False:
        # 对包含epoch的字典按键的数字部分排序
        if isinstance(current, dict):
            sorted_keys = sorted(current.keys(),
                                 key=lambda x: int(x.replace('epoch', '')) if x.startswith('epoch') else x)
            sorted_dict = {k: current[k] for k in sorted_keys}
            # 替换原字典内容
            current.clear()
            current.update(sorted_dict)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(existing_data, f, default_flow_style=False, allow_unicode=True)

    print(f"已更新YAML文件：{file_path}")