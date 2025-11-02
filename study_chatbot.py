
import streamlit as st
import os
import bs4

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


# Environment setup
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("groq_api_KEY"))
# Session memory
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# UI
st.set_page_config(page_title="📚 AI Study Chatbot with RAG")
st.title("📚 AI Study Chatbot")
st.caption("Paste a blog/article URL and ask questions about its content.")

# Input: URL
url = st.text_input("Paste a blog/article URL:", value="https://lilianweng.github.io/posts/2023-06-23-agent/")
start = st.button("Load & Start Chat")

# Memory-safe component
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None

# Load and prepare chain
if start:
    try:
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
            )
        )
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say you don't know. Keep the answer short.\n\n{context}"),
            ("human", "{input}"),
        ])
        question_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_chain)

        chat_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        st.session_state.chat_chain = chat_chain
        st.session_state.chat_history = []
        st.success("Article loaded successfully. Ask your questions below.")

    except Exception as e:
        st.error(f"Failed to load article: {e}")

# Chat logic
if st.session_state.chat_chain:
    for msg in st.session_state.chat_history:
        speaker = "You" if isinstance(msg, HumanMessage) else "Bot"
        st.markdown(f"**{speaker}:** {msg.content}")

    user_question = st.chat_input("Ask your question here...")

    if user_question:
        session_id = "user_rag_study"
        st.session_state.chat_history.append(HumanMessage(content=user_question))

        response = st.session_state.chat_chain.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": session_id}}
        )
        st.session_state.chat_history.append(AIMessage(content=response["answer"]))
        st.rerun()

