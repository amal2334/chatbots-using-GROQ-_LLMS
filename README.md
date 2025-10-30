 collection of 6 intelligent chatbots, each designed for a unique task using Streamlit, LangChain, Hugging Face, and Groq.

---

## Live Demos

| Chatbot Name         | Description                             | Live Demo Link |
|----------------------|-----------------------------------------|----------------|
| Customer Support     | Helps users make product decisions      | [Launch](https://nlp-and-llm-project-4c7d6ojrm62hm7n5py5bik.streamlit.app/) |
| Language Learning    | Language tutor chatbot                  | [Launch](https://nlp-and-llm-project-koxdsnsvyjg79vajwedwrc.streamlit.app/) |
| Life Coach           | Motivational chatbot for personal growth| [Launch](https://nlp-and-llm-project-8rejon7nticykp5588aru5.streamlit.app/) |
| Shopping Assistant   | Recommends shopping items               | [Launch](https://nlp-and-llm-project-hoo4bttzzpip9hgizkxsbh.streamlit.app/) |
| Travel Assistant     | Builds custom travel itineraries        | [Launch](https://nlp-and-llm-project-qvx3vx4gizk7yvylzpwkzk.streamlit.app/) |
| Study Chatbot        | Study helper with summarizing ability   | [Launch](https://nlp-and-llm-project-ndrbmhxappbsm9v7ngy52as.streamlit.app/) |

------

<p align="center">
  <h1> Specialized Chatbots </h1>
</p>

### Executive Summary
- This project showcases the creation of six distinct chatbots, each designed with unique conversational AI capabilities. Using LangChain for framework orchestration and Groq Gemini 2-9b-It model for fast inference, the chatbots range from a Travel Assistant to an AI Study Companion, demonstrating state-of-the-art natural language processing and human-computer interaction techniques.

### Business Problem
- With increasing demand for customizable AI assistants, the project addresses the challenge of building modular chatbots specialized in different domains, improving user engagement and task efficiency in applications such as productivity, customer support, language learning, and shopping.

### Methodology
- Developed chatbot modules using LangChain to orchestrate complex conversational flows.

- Leveraged Groq LLMs for high-performance language understanding and generation.

- Implemented prompt engineering, semantic embeddings via HuggingFace’s all-MiniLM-L6-v2, and retrieval-augmented generation (RAG) techniques for contextual accuracy.

- Deployed interactive interfaces with Streamlit for end-user testing and usability feedback.

### Skills Demonstrated
- Expertise in Natural Language Processing (NLP) and AI model integration.

- Proficiency with LangChain and cutting-edge LLM architectures (Groq Gemini).

- Applied semantic embeddings, prompt engineering, and RAG.

- Experience in interactive application deployment using Streamlit.

### Results & Recommendations
- Successfully built and demonstrated six specialized chatbots with targeted functionalities.

- Achieved robust, context-aware interactions validated through user tests.

- Future scope includes expanding chatbot adaptability and integrating further multi-modal capabilities.

### Next Steps
- Enhance chatbot modules with real-time learning and feedback adaptation.

- Explore multi-language support and voice-enabled interfaces.

- Integrate with enterprise systems for scalable deployment.


##  How to Run Locally

1. Clone the repo:
```bash
git clone https://github.com/amal2334/NLP-and-LLM-Project.git
cd NLP-and-LLM-Project

install the dependencies:
pip install -r requirements.txt
## run any chatbot now (for example Customer_Support)
streamlit run Customer_Support.py
