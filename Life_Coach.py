

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os 
from dotenv import load_dotenv
load_dotenv()

# 1. Set your Groq API Key

groq_api_key = os.getenv("groq_api_KEY")
model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)


# 2. Set up Session Store

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 3. Prompt Setup

system_prompt = (
    "You are a personal productivity coach. "
    "Help users build routines, focus on goals, and stay productive. "
    "Remember what each user tells you during the session."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# 4. Chain with Memory

chain = prompt | model
with_memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)


# 5. Streamlit App

st.set_page_config(page_title="🧠 Productivity Coach", page_icon="✅")
st.title("🧠 Personal Productivity Coach")

session_id = st.text_input("Enter your name to begin:", value="amal")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for msg in st.session_state.chat_history:
    role = "You" if isinstance(msg, HumanMessage) else "Coach"
    st.markdown(f"**{role}:** {msg.content}")

user_input = st.chat_input("Ask me about productivity, goals, routines...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    response = with_memory_chain.invoke(
        {"messages": st.session_state.chat_history},
        config={"configurable": {"session_id": session_id}}
    )

    st.session_state.chat_history.append(response)
    st.rerun()

