from langchain.embeddings.base import Embeddings
import requests
import langchain

class YandexEmbeddings(Embeddings):
    def __init__(self, api_key=None, folder_id=None):
        self.api_key = api_key
        self.folder_id = folder_id
        self.headers = {'Authorization': 'Api-key ' + self.api_key,
                             "x-folder-id" : self.folder_id }
        self.doc_uri = f"emb://{self.folder_id}/text-search-doc/latest"
        self.query_uri = f"emb://{self.folder_id}/text-search-query/latest"
        self.embed_url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"


    def embed_documents(self, text):
        json = {
            "modelUri": self.doc_uri,
            "text": text
        }
        vec = requests.post(self.embed_url, json=json, headers=self.headers)
        return vec.text
    def embed_query(self, text):
        json = {
            "modelUri": self.query_uri,
            "text": text
        }
        vec = requests.post(self.embed_url, json=json, headers=self.headers)
        return vec.text