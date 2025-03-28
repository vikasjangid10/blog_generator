import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
import streamlit as st

st.title("📝 Blog Generator")
st.subheader("Just give a topic and gat the title for blog and whole content as well!")

class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

user_topic = st.text_input("Enter a topic for the blog:")
submit = st.button("Generate")

if user_topic and submit:
    def title_creator(state: State):
        """Generate a concise blog title (4 words or fewer) without any reasoning or extra text."""
        sys_prompt = SystemMessage(
            content="You are a blog title creator. Generate a catchy and relevant blog title for the given topic. "
                    "The title must be **exactly 4 words or fewer**. Do not include any thoughts, explanations, or extra text. "
                    "Your response must be **only the title**, nothing else."
        )
        human_message = HumanMessage(content=user_topic)
        
        title_response = llm.invoke([sys_prompt, human_message]) 
        title_text = title_response.content.strip()  
        
        
        if "<think>" in title_text:
            title_text = title_text.split("</think>")[-1].strip()

        return {"messages": [AIMessage(content=title_text)]} 

    def content_creator(state: State):
        """Generate blog content without any `<think>` or extra reasoning."""
        title_message = state["messages"][-1]  
        title_text = title_message.content.strip()  
        
        sys_prompt = SystemMessage(
            content="You are a blog content creator. Generate high-quality, engaging blog content for the given title. "
                    "Do not include any **thoughts, explanations, or reasoning** in your response. "
                    "Your response must be **only the blog content**, nothing else."
        )
        human_message = HumanMessage(content=title_text)  
        
        response = llm.invoke([sys_prompt, human_message])  
        content_text = response.content.strip()

       
        if "<think>" in content_text:
            content_text = content_text.split("</think>")[-1].strip()

        return {"messages": [AIMessage(content=content_text)]}  

    builder.add_node("title_maker", title_creator)
    builder.add_node("content_creator", content_creator)

    builder.add_edge(START, "title_maker")
    builder.add_edge("title_maker", "content_creator")
    builder.add_edge("content_creator", END)

    graph = builder.compile()
    messages = graph.invoke({"messages": []})  
    
    st.subheader("Generated Blog Title:")
    st.write(messages["messages"][0].content)  
    
    st.subheader("Generated Blog Content:")
    st.write(messages["messages"][1].content)  
