# report_generator.py
import os
import yaml
from datetime import datetime

def generate_markdown_report(exp_record_path: str, output_dir: str = "./reports"):
    """生成Markdown格式的实验报告"""
    os.makedirs(output_dir, exist_ok=True)

    with open(exp_record_path, 'r', encoding='utf-8') as f:
        record = yaml.safe_load(f)

    exp_id = record["experiment_id"]
    date = record["date"]
    config = record["config"]
    results = record["results"]

    report_content = f"""### **实验版本** [{exp_id}]  
**日期**: {date}  
**数据集**:  
- 名称: LoRa_RFFI  
- 设备数量: {config['dataset']['devices']}  
- 训练/验证/测试集划分: 依据配置  
- 数据增强: {config['dataset'].get('augmentation', 'None')}  

**模型架构**:  
- 基础模型: {config['model']['architecture']}  
- 超参数: 学习率={config['model']['parameters']['learning_rate']}, 批量大小={config['model']['parameters']['batch_size']}  

**硬件平台**:  
- 训练: {config['hardware']['device']}  

**实验结果**:  
- 最终准确率: {results.get('accuracy', 'N/A')}  
- 训练耗时: {results.get('training_time', 'N/A')} 秒
"""

    report_filename = f"{exp_id}_report.md"
    report_filepath = os.path.join(output_dir, report_filename)

    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return report_filepath
