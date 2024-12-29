import yaml
from functools import lru_cache

@lru_cache()
def get_config():
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file) 