from modules.input_data_manager import add_new_condition_to_db
from modules.pubmed import PubMedClient
from chromadb import HttpClient
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "../config/config.yaml")

with open(config_path, 'r') as config_file:
    config_dict = yaml.safe_load(config_file)

def main():
    chroma_client = HttpClient(host=config_dict['chroma']['server_host'],
                               port=config_dict['chroma']['server_port'])
    pubmed_client = PubMedClient(config_dict)
    condition_name = input("추가할 질병 또는 바이러스 이름을 입력하세요: ")

    add_new_condition_to_db(condition_name, pubmed_client, chroma_client)

if __name__ == "__main__":
    main()
