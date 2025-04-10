import os
from typing import TypedDict
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import threading
import uvicorn

# FastAPI app initialization
app = FastAPI()

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TMDB_API_KEY"] = os.getenv("TMDB_API_KEY")

# Set up LLM
llm_text = ChatGroq(model="qwen-2.5-32b")

# Define the state for LangGraph
class State(TypedDict):
    genre: str
    year: int
    recommendation: str
    source: str

class Preferences(BaseModel):
    genre: str
    year: int

response_schemas = [
    ResponseSchema(
        name="answer",
        description=( 
            "List of 5 movie recommendations. Each should be in this format:\n\n"
            "Title: <movie name>\n \n "
            "Description: <short description>\n"
        )
    ),
    ResponseSchema(
        name="source",
        description="A trusted website link used as the source for the recommendation."
    ),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

# LangChain Prompt
prompt = PromptTemplate(
    template=( 
        "You are a movie expert. Answer the following question in a structured format.\n\n"
        "{format_instructions}\n\n"
        "Question: {question}"
    ),
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | llm_text | output_parser

# Genre mapping to TMDB genre IDs
genre_mapping = {
    "Happy": 35,      # Comedy
    "Sad": 18,        # Drama
    "Romantic": 10749,# Romance
    "Adventurous": 12,# Adventure
    "Thriller": 53,   # Thriller
    "Bollywood": 10402,# Music
    "Crime": 80,      # Crime
    "Horror": 27,     # Horror
    "War": 10752,     # War
}

# Fetch real-time movie data from TMDB API
def fetch_movies_from_tmdb(genre: str, year: int):
    genre_id = genre_mapping.get(genre, "")
    
    if not genre_id:
        return []
    
    url = f"https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": os.getenv("TMDB_API_KEY"),
        "with_genres": genre_id,
        "primary_release_year": year,
        "sort_by": "popularity.desc",
    }
    response = requests.get(url, params=params)
    print(f"TMDB API Response Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"TMDB Response Data: {data}")
        return data['results']
    else:
        print(f"Error fetching data from TMDB: {response.status_code}")
        return []

# Recommendation logic node
def recommendation(state: State):
    tmdb_movies = fetch_movies_from_tmdb(state['genre'], state['year'])
    
    if tmdb_movies:
        recommendations = []
        for movie in tmdb_movies[:5]:  # Get top 5 movies
            title = movie['title']
            description = movie['overview']
            recommendations.append(f"Title: {title}\nDescription: {description}\n")
        
        source = "TMDB API"
        return {
            "recommendation": "\n\n".join(recommendations),
            "source": source
        }
    else:
        # Mock recommendations for debugging purposes
        return {
            "recommendation": "Title: Example Movie\nDescription: This is a sample movie description.",
            "source": "Source: TMDB API"
        }

# Build the LangGraph
builder = StateGraph(State)
builder.add_node("get_recommendation", recommendation)
builder.add_edge(START, "get_recommendation")
builder.add_edge("get_recommendation", END)
graph = builder.compile()

# FastAPI endpoint to handle the movie recommendation requests
@app.post("/recommendation")
async def get_movie_recommendations(preferences: Preferences):
    print(f"Received preferences: {preferences}")  # Debugging line
    result = graph.invoke(preferences.dict())  # Ensure the preferences are passed as a dictionary
    print(f"Recommendation result: {result}")
    return result

# --- Streamlit UI ---
def run_streamlit():
    st.set_page_config(page_title="Movie Recommender", layout="centered")
    
    st.title("Movie Recommendation Assistant")
    st.markdown("Get a personalized movie suggestion based on your preferences.")
    
    with st.form("preferences_form"):
        genre = st.selectbox(
            "Genre",
            ["Happy", "Sad", "Romantic", "Adventurous", "Thriller", "Bollywood", "Crime", "Horror", "War"]
        )
        year = st.selectbox("Year", [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015])
    
        submitted = st.form_submit_button("Recommend a Movie")
    
    if submitted:
        if not genre or not year:
            st.warning("Please fill in all required fields: genre, year")
        else:
            user_state = {
                "genre": genre,
                "year": year
            }
    
            with st.spinner("Fetching the perfect movie for you..."):
                response = requests.post("http://127.0.0.1:8000/recommendation", json=user_state)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Here are your movie recommendations:")
                    recommendations = result["recommendation"].strip().split("\n\n")
                    # Loop through recommendations with serial number
                    for index, rec in enumerate(recommendations, 1):
                        # Split the title and description
                        rec_parts = rec.split("\nDescription: ")
                        title = rec_parts[0].replace("Title: ", "").strip()
                        description = rec_parts[1].strip() if len(rec_parts) > 1 else ""
                        
                        # Display the recommendation with serial number
                        st.markdown(f"**{index}. Title:** {title}")
                        st.markdown(f"**Description:** {description}")
                    st.markdown(f"**Source:** {result['source']}")
                else:
                    st.error(f"Failed to fetch recommendations: {response.status_code}")


# Start FastAPI server in a separate thread
def start_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    threading.Thread(target=start_fastapi, daemon=True).start()
    
    # Run Streamlit UI
    run_streamlit()
