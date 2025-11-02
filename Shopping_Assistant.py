
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Model Setup (Using Groq)
# -----------------------------
groq_api_key = os.getenv("groq_api_KEY") 
model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# -----------------------------
# Session Store Setup
# -----------------------------
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# -----------------------------
# Prompt Template
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all the questions to the best of your ability."),
        MessagesPlaceholder(variable_name="messages")
    ]
)



chain = prompt | model
with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")


st.set_page_config(page_title="Shopping Assistant", page_icon="🛍️")
st.title("Shopping Assistant")

# Get session/user name
session_id = st.text_input("Enter your name to start:", value="shop_user1")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    role = "You" if isinstance(msg, HumanMessage) else "Assistant"
    st.markdown(f"**{role}:** {msg.content}")

# Chat input box
user_input = st.chat_input("What are you looking to buy?")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Invoke model with memory
    response = with_message_history.invoke(
        {"messages": st.session_state.chat_history},
        config={"configurable": {"session_id": session_id}}
    )

    # Append response to history and rerun
    st.session_state.chat_history.append(response)
    st.rerun()

