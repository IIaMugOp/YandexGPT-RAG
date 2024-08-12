import streamlit as st
import os
import glob
import json
import chromadb

from YandexLLModule import *
from YandexEmbeddingsModule import *

from langchain_community.document_loaders import TextLoader
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document


YA_FOLDER_ID = "b1g4g4a90jln1j686ud9"
YAGPT_API_KEY = "AQVNxR5vPv-B0uzNJTtvnc28vZLbP2TKu6k9fPTg"

embeding_model = YandexEmbeddings(YAGPT_API_KEY, YA_FOLDER_ID)
llm_model = YandexLLM(api_key=YAGPT_API_KEY, folder_id=YA_FOLDER_ID)

doc_files = glob.glob('C:\LLM\data_texts\*.txt')
doc_texts = []
for file_name in doc_files:
    data = open(file_name, "r", encoding='utf-8').read()
    doc_texts.append(data)

for doc in doc_texts:
    result = embeding_model.embed_documents(doc)

doc_embeddings = []
for doc in doc_texts:
    doc_embeddings.append(json.loads(embeding_model.embed_documents(doc))["embedding"])


client = chromadb.Client()
collection = client.get_or_create_collection(name="InternalDocs")

collection.add(
    embeddings = doc_embeddings,
    documents = doc_texts,
    metadatas = [{"source": "first document"},{"source": "second document"},{"source": "third document"}, {"source": "fourth document"}],
    ids = ["id1", "id2", "id3", "id4"]
)



def generate_response(query):
    model = YandexLLM(
                      api_key=YAGPT_API_KEY,
                      folder_id=YA_FOLDER_ID,
                      temperature=st.session_state["interface_temperature"],
                      max_tokens=st.session_state["interface_max_tokens"]
                      )

    my_query_embedding = json.loads(embeding_model.embed_query(query))["embedding"]

    relevant_doc = collection.query(
        query_embeddings=my_query_embedding,
        n_results=1
    )

    relevant_doc_lgch =  [Document(page_content=relevant_doc['documents'][0][0], metadata=relevant_doc['metadatas'][0][0])]

    document_template = PromptTemplate(
        input_variables=["page_content"], #A list of the names of the variables.
        template="{page_content}" # The document template.
    )

    document_variable_name = "context"

    template_override = """
    Представь что ты собеседуешься в компанию к Ярославу.
    Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста.
    Текст:
    -----
    {context}
    -----
    Вопрос:
    {query}
    """

    prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=template_override
    )

    llm_chain = LLMChain(llm=llm_model, prompt=prompt)

    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_template,
        document_variable_name=document_variable_name
    )
    st.info(model.invoke(my_query_embedding))





st.title("🦜🔗 Test RAG-system interface")

if "interface_temperature" not in st.session_state:
    st.session_state["interface_temperature"] = 0.3
interface_temperature = st.slider("Temperature", 0.0, 1.0, 0.3, key="interface_temperature")
llm_model.temperature = interface_temperature

if "interface_max_tokens" not in st.session_state:
    st.session_state["interface_max_tokens"] = 2000
interface_max_tokens = st.slider("Max tokens", 0, 5000, 2000, key="interface_max_tokens")
llm_model.max_tokens = interface_max_tokens


print("test1")


with st.form("my_form"):
    text = st.text_area(
        "Введите Ваш запрос:",
        "вот сюда",
    )
    submitted = st.form_submit_button("Задать вопрос")
    print("test2")
    generate_response(text)


st.write(st.session_state)