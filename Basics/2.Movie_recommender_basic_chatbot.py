# app.py

import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Set up LLM
llm_text = ChatGroq(model="qwen-2.5-32b")

# Define the state for LangGraph
class State(TypedDict):
    genre: str
    language: str
    country: str
    recommendation : str

# Recommendation logic node
def recommendation(state: State):
    prompt = (
        f"Based on the user's preferences:",
        f"Genre: {state['genre']}, Language: {state['language']}, "
        f"Country: {state['country']}, recommend one good movie with a short description."
    )
    response = llm_text.invoke(prompt)
    return {"recommendation": response.content}  # Use .content (not .content()) for LLM responses

# Build the LangGraph
builder = StateGraph(State)
builder.add_node("get_recommendation", recommendation)
builder.add_edge(START, "get_recommendation")
builder.add_edge("get_recommendation", END)
graph = builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"].content)

# --- Streamlit UI ---
st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("Movie Recommendation Assistant")
st.markdown("Get a personalized movie suggestion based on your mood and preferences.")

# Input fields
with st.form("preferences_form"):
    genre = st.selectbox(
    "Genre",
    ["Happy", "Sad", "Romantic", "Adventurous","Thriller","Bollywood","Crime","Horror","War"])
    language = st.selectbox("Language",
                            ["English","French", "Hindi"])
    country = st.selectbox("Country",
                            ["India","Germany", "Italy","Spanish"])
    

    submitted = st.form_submit_button("Recommend a Movie")

# Handle form submission
if submitted:
    if not genre or not language or not country:
        st.warning("Please fill in all required fields:genre,language and country ")
    else:
        user_state = {
            "genre": genre,
            "language": language,
            "country": country
        }

        with st.spinner("Fetching the perfect movie for you..."):
            result = graph.invoke(user_state)

        st.success("Here's your movie recommendation:")
        st.markdown(result["recommendation"])
