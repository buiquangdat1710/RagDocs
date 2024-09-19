import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

# Memory imports
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


def load_document(file):
    name, extension = os.path.splitext(file)
    if extension == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == ".csv":
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(file)
    elif extension == ".txt":
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    elif extension == ".xlsx":
        from langchain_community.document_loaders import UnstructuredExcelLoader
        loader = UnstructuredExcelLoader(file)
    else:
        return None
    data = loader.load()
    return data


def load_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()

        # Wrap the text content in a Document object
        return [Document(page_content=text)]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to retrieve the webpage: {e}")
        return None


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    vector_store.persist()
    return vector_store

def ask_and_get_answer(vector_store, query, k=3, memory=None):
    if memory is None:
        memory = ConversationBufferMemory(memory_key="chat_history")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, memory=memory)
    
    answer = chain.invoke(query)
    return answer


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    cost = total_tokens * 0.0004 / 1000
    return total_tokens, cost

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    
    st.image('img.png')
    st.subheader('LLM Question Answering Application 🤖')

    with st.sidebar:
        uploaded_file = st.file_uploader('Choose a file:', type=['pdf', 'docx', 'csv', 'txt', 'xlsx'])
        url_input = st.text_input('Or enter a webpage URL:')
        
        chunk_size = st.number_input('Chunk size', min_value=100, max_value=2048, value=256, on_change=clear_history)
        k = st.number_input('K', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add data', on_click=clear_history)

        if (uploaded_file or url_input) and add_data:
            with st.spinner('Reading, chunking, and creating embeddings...'):
                if uploaded_file:
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join('./', uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)
                    data = load_document(file_name)
                elif url_input:
                    data = load_from_url(url_input)
                
                if data:
                    chunks = chunk_data(data, chunk_size)
                    st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                    print(chunks[0])
                    tokens, embeddings_cost = calculate_embedding_cost(chunks)
                    st.write(f'Embedding cost: ${embeddings_cost:.4f} USD')
                    vector_store = create_embeddings(chunks)
                    st.session_state.vs = vector_store
                    st.session_state.memory = ConversationBufferMemory()
                    st.success('Data processed, chunked, and embedded successfully!')

    q = st.text_input('Enter your question about the content:')
    
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            memory = st.session_state.memory
            answer = ask_and_get_answer(vector_store, q, k, memory=memory)
            
            st.text_area('LLM Answer', value=answer['result'])
            st.divider()

            if "history" not in st.session_state:
                st.session_state.history = ""

            value = f'Q: {q}\nA: {answer["result"]}'
            st.session_state.history = f'{value}\n{"-"*100}\n{st.session_state.history}'

            h = st.session_state.history
            st.text_area('Chat History', value=h, height=400)
