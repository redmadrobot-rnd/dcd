import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    def __init__(self, config_path: str = "metrics_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.data: Dict[str, Any] = yaml.safe_load(f)
        
        self._validate_config()
    
    def _validate_config(self):
        required_sections = ['input', 'output', 'openai', 'required_columns']
        for section in required_sections:
            if section not in self.data:
                raise ValueError(f"Missing required section in config: {section}")
    
    @property
    def naive_rag_dataset(self) -> str:
        return self.data['input']['naive_rag_dataset']
    
    @property
    def dcd_dataset(self) -> str:
        return self.data['input']['dcd_dataset']
    
    @property
    def results_dir(self) -> Path:
        return Path(self.data['output']['results_dir'])
    
    @property
    def dcd_metrics_file(self) -> str:
        return self.data['output']['dcd_metrics_file']
    
    @property
    def naive_rag_metrics_file(self) -> str:
        return self.data['output']['naive_rag_metrics_file']
    
    @property
    def detailed_naive_rag(self) -> str:
        return self.data['output']['detailed_naive_rag']
    
    @property
    def detailed_dcd(self) -> str:
        return self.data['output']['detailed_dcd']
    
    @property
    def base_url(self) -> str:
        return self.data['openai']['base_url']
    
    @property
    def api_key(self) -> str:
        return self.data['openai']['api_key']
    
    @property
    def model_name(self) -> str:
        return self.data['openai']['model_name']
    
    @property
    def timeout(self) -> float:
        return self.data['openai']['timeout']
    
    @property
    def temperature(self) -> float:
        return self.data['openai']['temperature']
    
    @property
    def context_temperature(self) -> float:
        return self.data['context_eval']['temperature']
    
    @property
    def required_columns(self) -> list:
        return self.data['required_columns']
