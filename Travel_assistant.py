

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
import os
# -----------------------------
# Set your Groq API key here
# -----------------------------
groq_api_key = os.getenv("groq_api_KEY")

# Initialize the model
model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# Streamlit app setup
st.set_page_config(page_title="Travel Assistant 🌍", page_icon="✈️")
st.title("🌍 Smart Travel Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a smart travel assistant who helps people plan their vacations."),
        HumanMessage(content="Hi, I’m Sarah and I’d like to plan a trip to Italy next month."),
        AIMessage(content="Hi Sarah! That sounds exciting. I'd love to help you plan your trip to Italy. Do you have any cities or activities in mind?")
    ]

# Display previous messages
for msg in st.session_state.chat_history[1:]:
    role = "You" if isinstance(msg, HumanMessage) else "Assistant"
    st.markdown(f"**{role}:** {msg.content}")

# Chat input
user_input = st.chat_input("Where would you like to travel next?")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get assistant response using Groq model
    response = model.invoke(st.session_state.chat_history)

    # Add response to history
    st.session_state.chat_history.append(response)

    # Rerun the app to show the updated conversation
    st.rerun()




