import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Set up LLM
llm = ChatGroq(model="qwen-2.5-32b")

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define chatbot node
def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}

# Build the graph
builder = StateGraph(State)
builder.add_node("Chat-bot", chatbot)
builder.add_edge(START, "Chat-bot")
builder.add_edge("Chat-bot", END)
graph = builder.compile()

# Stream updates from the graph
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"].content)

# Chat loop
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break  # <-- this was missing
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User:", user_input)
        stream_graph_updates(user_input)
        break
