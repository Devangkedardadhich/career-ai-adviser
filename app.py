import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# Load env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HISTORY_FILE = "history.json"

# Load history from file
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

# Save history to file
def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-7:], f)

# Setup Streamlit config
st.set_page_config(
    page_title="💼 Career AI Advisor For All",
    page_icon="💡",
    layout="centered"
)

# Session state for history
if "search_history" not in st.session_state:
    st.session_state.search_history = load_history()
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Sidebar
with st.sidebar:
    st.markdown("### 👋 Welcome to the world of jobs!")
    st.markdown("Ask me anything like:")
    st.markdown("- What jobs can I get with Python?")
    st.markdown("- I want to work abroad, how should I prepare?")
    st.markdown("- Which skills are best for remote jobs?")
    st.markdown("## 🕓 Your Last 7 Searches")

    for i, query in enumerate(st.session_state.search_history[::-1], 1):
        if st.button(f"{i}. {query}", key=f"history_{i}"):
            st.session_state.user_input = query

    if st.button("🧹 Clear History"):
        st.session_state.search_history = []
        save_history([])
        st.success("History cleared!")

    st.markdown("----")
    st.markdown("Made with ❤️ using Groq + LangChain")
    st.markdown("### 👨‍💻 Created by: Devang Dadhich")

# Custom CSS
st.markdown("""<style>
/* Simplified UI CSS */
body { font-family: 'Segoe UI', sans-serif; background-color: #f4f7fb; }
.header-container { text-align: center; padding: 30px 0 20px; background-color: #fff;
    border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 30px; }
.header-title { font-size: 48px; font-weight: 700; color: #3b82f6; margin-bottom: 10px; }
.header-description { font-size: 20px; font-weight: 400; color: #444; margin-bottom: 25px; }
.header-line { width: 60px; height: 3px; background-color: #3b82f6; margin: 0 auto; border-radius: 2px; }
.response-card, .input-card {
    background-color: #ffffff; padding: 20px 25px; border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-top: 20px;
}
.footer {
    text-align: center; padding: 20px; background-color: #3b82f6;
    color: white; margin-top: 40px; font-size: 14px; border-radius: 12px;
}
.stTextArea>textarea {
    font-family: 'Segoe UI', sans-serif; font-size: 16px; padding: 12px;
    border-radius: 8px; border: 1px solid #ddd; width: 100%;
    height: 150px; box-sizing: border-box;
}
</style>""", unsafe_allow_html=True)

# Header Section
st.markdown("""<div class="header-container">
    <div class="header-title">💼 Your Career AI Advisor</div>
    <div class="header-description">
        Get instant career advice based on your skills, interests, or doubts. Powered by Devang Career Ai.
    </div>
    <div class="header-line"></div>
</div>""", unsafe_allow_html=True)

# Input Box
st.markdown('<div class="input-card">', unsafe_allow_html=True)
user_input = st.text_area(
    "🔍 Type your career question below:",
    value=st.session_state.user_input,
    placeholder="e.g. I live in India studying in a government university. My field of interest is Full stack development..."
)
st.markdown('</div>', unsafe_allow_html=True)

# Prompt template
system_prompt = (
    "You are a professional career advisor who understands global job markets, especially India or any country user mentions. "
    "When a user asks about jobs, skills, career growth, or salary:\n"
    "1. Understand their background, education, and location.\n"
    "2. Recommend learning paths, missing skills, and certifications.\n"
    "3. Suggest real job roles, companies hiring, and salary expectations.\n"
    "4. Provide actionable next steps with resources if possible.\n"
    "Keep your advice simple, clear, and actionable."
)


# Format Output
def format_response(text):
    lines = text.split("\n")
    formatted = []
    for line in lines:
        line = line.strip()
        if line.lower().startswith("1.") or line.lower().startswith("step 1"):
            formatted.append(f"### 🧭 {line}")
        elif any(line.startswith(f"{i}.") for i in range(2, 10)):
            formatted.append(f"- {line}")
        elif line:
            formatted.append(line)
    return "\n".join(formatted)

# Get Career Advice from Groq API
def get_career_advice_with_langchain(query):
    llm = ChatOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )

    tools = [
        Tool(
            name="Career Advisor Tool",
            func=lambda q: llm.invoke(system_prompt + "\nUser: " + q),
            description="Get personalized career advice from Groq AI."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    response = agent.invoke(query)
    return format_response(response.get("output", response))

# Button Click Action
if st.button("🚀 Get Career Advice") and user_input.strip():
    with st.spinner("🤖 Thinking..."):
        try:
            reply = get_career_advice_with_langchain(user_input)

            # Save history
            if user_input not in st.session_state.search_history:
                st.session_state.search_history.append(user_input)
                st.session_state.search_history = st.session_state.search_history[-7:]
                save_history(st.session_state.search_history)

            st.session_state.user_input = user_input

            st.markdown(f"""
                <div class="response-card">
                    <h4 style="color:#10b981;margin-top:0;">🌟 Here's what I recommend:</h4>
                    <div style="white-space:pre-wrap; color:#111827; font-size:16px; line-height:1.6;">
                        {reply}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
elif user_input.strip() == "":
    st.info("✍️ Start typing your question and click **Get Career Advice**.")

# Footer
st.markdown("""<div class="footer">
✍️ Thanks for using | Developed by Devang Dadhich | Powered by Groq + LangChain
</div>""", unsafe_allow_html=True)
