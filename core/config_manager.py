import yaml
import os

class Config:
    def __init__(self, config_path="config/config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self._cfg = yaml.safe_load(f)
            
    @property
    def recognition(self):
        # 兼容旧配置 key 'model' -> 'recognition'
        return self._cfg.get('recognition', self._cfg.get('model', {}))
        
    @property
    def detector(self):
        return self._cfg.get('detector', {})
        
    @property
    def preprocess(self):
        return self._cfg.get('preprocess', {})

# 全局单例
try:
    global_config = Config()
except Exception as e:
    print(f"Warning: Failed to load config: {e}")
    global_config = None
