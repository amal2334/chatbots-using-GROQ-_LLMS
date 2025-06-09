
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()
import os
# -----------------------------
# 1. Setup Groq Model
# -----------------------------
groq_api_key = os.getenv("groq_api_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# -----------------------------
# 2. Manual Message Trimmer
# -----------------------------
def trim_messages_last_n(messages, n=6):
    # Always keep the system message, trim rest
    system = [msg for msg in messages if isinstance(msg, SystemMessage)]
    conversation = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    return system + conversation[-n:]

# -----------------------------
# 3. Prompt Setup
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant."),
    MessagesPlaceholder(variable_name="messages")
])

chain = (
    RunnablePassthrough.assign(messages=lambda d: trim_messages_last_n(d["messages"]))
    | prompt
    | model
)

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Customer Support Chatbot", page_icon="📞")
st.title("Customer Support Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful customer support assistant.")
    ]

# Display history
for msg in st.session_state.chat_history:
    role = "You" if isinstance(msg, HumanMessage) else ("Support" if isinstance(msg, AIMessage) else "System")
    st.markdown(f"**{role}:** {msg.content}")

# Chat input
user_input = st.chat_input("Describe your issue...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    response = chain.invoke({"messages": st.session_state.chat_history})

    st.session_state.chat_history.append(response)
    st.rerun()
