#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA



import streamlit as st
#from watsonxlangchain import LangChainInterface
from wxai_langchain.llm import LangChainInterface

creds={
    'apikey': 'mUsdu8M5yzGM1j2LqHENsoAgjBza_SUxA6QHczGLklYI',
    'url': 'https://us-south.ml.cloud.ibm.com'
}

llm = LangChainInterface(
    credentials = creds,
    model = 'meta-llama/llama-2-70b-chat',
    params = {
        'decoding_method': 'sample',
        'max_new_tokens': 200,
        'temperature': 0.5
        },
    project_id='5157b2fc-0115-4bcd-af3c-5016227723ca')


@st.cache_resource
def load_pdf():
  pdf_name = 'what is generative ai.pdf'
  loaders = [PyPDFLoader(pdf_name)]
  index = VectorstoreIndexCreator(
      embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
      text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  ).from_loaders(loaders)
  return index

index = load_pdf()

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question')


st.title("Ask watsonx")

if 'messages' not in st.session_state:
  st.session_state.message = []

for message in st.session_state.messages:
  st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Pass Your Prompt here')

if prompt:
  st.chat_message('user').markdown(prompt)
  st.session_state.messages.append({'role': 'user', 'content': prompt})
  response = chain.run(prompt)
  st.chat_message('assistant').markdown(response)
  st.session_state.messages.append({'role': 'assistant', 'content': response})

