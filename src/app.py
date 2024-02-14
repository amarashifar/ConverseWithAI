from cgitb import text
from unittest import loader

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()



def get_vectorstore_from_url(url):
    # Initialize a loader object to fetch content from the provided URL
    loader = WebBaseLoader(url)
    document = loader.load()
    # Initialize a text splitter object to split the document into smaller chunks

    text_splitter = RecursiveCharacterTextSplitter()
    # Split the document into smaller chunks

    documents_chucks = text_splitter.split_documents(document)

    #create vector stores from chunks
    vector_store = Chroma.from_documents(documents_chucks, OpenAIEmbeddings())

    return vector_store



def get_retreiver_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"), ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"
         )
    ])

    #retriever
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain



def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's questions based on the below context:\n\n{context}"
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)




def get_response(user_input):
    retriever_chain = get_retreiver_chain(st.session_state.vector_store)

    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversational_rag_chain.invoke({
        "chat_history":
        st.session_state.chat_history,
        "input":
        user_query
    })

    return response['answer']


#application Configuration
st.set_page_config(page_title="Chat with any website", page_icon="🍑")
st.title("Ask Away")

#sidebar
with st.sidebar:
    st.header("Setting")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter URL")

else:
    #session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    #user input
    user_query = st.chat_input("Type your message here.....")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    #CONVERSTATION
    #looping through the elements in chat history. The chat message will be formatted accordingly as an "AI" or "Human"
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

#disable if there doesnt
