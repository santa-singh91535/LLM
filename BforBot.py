import streamlit as st
import os
import pickle
import time
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

from dotenv import load_dotenv
load_dotenv()

st.title('News Research Tool📊')
st.sidebar.title("News Urls")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f'URL {i+1}')
    urls.append(url)

process_url_clicked =st.sidebar.button("Process URLs")
file_path = 'faiss_store_openai.pkl'

main_placeholder= st.empty()
llm = ChatOpenAI(temperature= 0.8 , max_tokens=500)

if process_url_clicked:
    # load data
    loader =UnstructuredURLLoader(urls = urls)
    main_placeholder.text("Data Loading...Started")
    data = loader.load()
    # splitting into chunks
    text_splitter= RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size= 1000
        )
    main_placeholder.text("Text Spitter....Started")
    docs= text_splitter.split_documents(data)
    # create embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore_openai =FAISS.from_documents(docs , embeddings)
    main_placeholder.text("Embedding Vector Started Buliding")
    time.sleep(2)

    # # save faiss index
    with open(file_path,'wb') as f:
        pickle.dump(vectorstore_openai,f)

query= main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path ,'rb') as f:
            vectorstore=pickle.load(f)
            chain =RetrievalQAWithSourcesChain.from_llm(llm = llm  , retriever= vectorstore.as_retriever())
            result = chain({'question':query} , return_only_outputs=True)
            st.header("Answer")
            st.write(result['answer'])

            # display sources 
            sources= result.get('sources','')
            if sources:
                st.subheader("Sources:")
                sources_list= sources.split("\n")
                for source in sources_list:
                    st.write(source)
