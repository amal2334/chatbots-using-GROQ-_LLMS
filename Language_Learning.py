
#Language learning Assistant with Memory
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os
load_dotenv()


# -----------------------------
# 1. Groq Model Setup
# -----------------------------
groq_api_key = os.getenv("groq_api_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# -----------------------------
# 2. Prompt for Language Tutor
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a supportive language tutor. Respond in {language}. Encourage and correct the student. Remember what they learned."),
    MessagesPlaceholder(variable_name="messages")
])

chain = prompt | model

# -----------------------------
# 3. Message History Store
# -----------------------------
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Language Learning Tutor",page_icon="🎓")
st.title("Language Learning Assistant")

# Input: user session and language
session_id = st.text_input("Enter your name:", value="english_learning_amal")
language = st.selectbox("Choose your learning language:", ["English", "Arabic", "Spanish", "French","Hindi","Turkish","Telugu"])

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    role = "You" if isinstance(msg, HumanMessage) else "Tutor"
    st.markdown(f"**{role}:** {msg.content}")

# Input: user message
user_input = st.chat_input("Ask a question or practice a phrase...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    response = with_message_history.invoke(
        {
            "messages": st.session_state.chat_history,
            "language": language
        },
        config={"configurable": {"session_id": session_id}}
    )

    st.session_state.chat_history.append(response)
    st.rerun()
