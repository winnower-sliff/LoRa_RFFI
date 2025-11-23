# experiment_logger.py
import os
import yaml
from datetime import datetime
from typing import Dict, Any

class ExperimentLogger:
    def __init__(self, log_dir: str = "./experiments"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def create_experiment_record(self, experiment_config: Dict[str, Any]):
        """创建新的实验记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"exp_{timestamp}"

        record = {
            "experiment_id": exp_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "config": experiment_config,
            "results": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "status": "running"
            }
        }

        # 保存到文件
        filename = f"{exp_id}.yaml"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(record, f, default_flow_style=False, allow_unicode=True, width=float("inf"))

        return filepath, exp_id

    def update_experiment_result(self, exp_id: str, results: Dict[str, Any]):
        """更新实验结果"""
        filepath = os.path.join(self.log_dir, f"{exp_id}.yaml")

        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                record = yaml.safe_load(f)

            record["results"].update(results)
            record["metadata"]["updated_at"] = datetime.now().isoformat()
            record["metadata"]["status"] = "completed"

            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(record, f, default_flow_style=False, allow_unicode=True)
