from pydantic import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    config: Dict[str, Any] = {
        'service': {
            'chroma': {
                'host': '10.0.1.10',
                'port': 8001,
                'collections': {
                    'initial': 'supplements_initial',
                    'interaction': 'supplements_interaction',
                    'adjustment': 'supplements_adjustment'
                }
            }
        },
        'data_sources': {
            'pubmed': {
                'api_key': None,
                'batch_size': 100
            }
        }
    }

CONFIG = Settings() 