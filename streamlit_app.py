import streamlit as st
from openai import OpenAI

# Load ===========================================
from typing import Any, Optional, Sequence, List
# from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain_community.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain.tools import tool
from langchain.tools import BaseTool, StructuredTool
from langchain.tools.base import ToolException
from pydantic import BaseModel, Field

# lang graph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import TypedDict, List, Annotated


# for langgraph visualization of the graph diagram
from IPython.display import Image, display

# Setup API Keys =====================
import os
# from google.colab import userdata
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets['langsmith_new']
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com" # included or not does not matter
os.environ["LANGCHAIN_PROJECT"] = "lablab.ai hackathon Okt 2024"

# Setup LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI( model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=2000,
        timeout=None,
        max_retries=2,
        base_url="https://api.aimlapi.com/",
    )

# Prompt Configuration
from langchain_core.prompts import ChatPromptTemplate


system_prompt = """
      You are a helpful assistant that translates English sentences to a pandas python dataframe query.
      The dataframe df has info on films and TV series made in San Francisco
      The df columns are:
      'Title', 'Release Year', 'Locations', 'Fun Facts', 'Production Company',
            'Distributor', 'Director', 'Writer', 'Actor 1', 'Actor 2', 'Actor 3',
            'SF Find Neighborhoods', 'Analysis Neighborhoods',
            'Current Supervisor Districts'

      If you don't know the answer, just say that you don't know.
     """

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        ("human", "{input}"),
    ]
)

# Load the data =====================
import pandas as pd

file_path = './data/Film_Locations_in_San_Francisco_20241005.csv'

df = pd.read_csv(file_path)
columns = df.columns
print(columns)
# prompt: write a query for all films made in the 80s
print('\n')
eighties_films = df[(df['Release Year'] >= 1980) & (df['Release Year'] <= 1989)]
print(eighties_films['Title'].value_counts)

# =========================================
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode,tools_condition
# custom function definiton

# Define a pydantic model for the input
class PandasInput(BaseModel):
    command_string: str = Field(..., description="A valid pandas command string to execute.")

# Define a pydantic model for the output
class PandasOutput(BaseModel):
    result: dict = Field(..., description="The result of the pandas command as a dict.")



@tool(args_schema=PandasInput, return_direct=True)
def run_pandas(command_string: str, df: pd.DataFrame=df) -> PandasOutput:
    """
  Runs a pandas command string and returns the result as a pd.DataFrame.

  Args:
    pandas_string: The pandas command string to execute.

  Returns:
    The result of the pandas command as a pd.DataFrame
  """
    local_env = {
        'pd': pd,
        'df': df.copy(),
        'result': None
    }

    edited_string = "result = " + command_string
    command_string = edited_string + "\n"

    try:
        exec(command_string, {"__builtins__": {}}, local_env)
        # Extract the result if it was assigned in the command_string
        result = local_env.get('result', "Command executed successfully.")
        if isinstance(result, pd.DataFrame):  # Check if it's a DataFrame
            return PandasOutput(result=result.to_dict())  # Return the DataFrame as a dict object
        else:
            return PandasOutput(result=result)  # Return the result as is

    except Exception as e:
        # Wrap the exception in ToolException to be handled by the agent
        raise ToolException(f"Error executing command: {e}")


tools = [run_pandas]
tool_node = ToolNode(tools)
# ===============================================

# LangGraph setup ===============================
## Langgraph Application
from langgraph.graph.message import add_messages
class State(TypedDict):
  messages:Annotated[list,add_messages]
#################################################
from langgraph.graph import StateGraph,START,END

graph_builder= StateGraph(State)

llm_with_tools=llm.bind_tools(tools=tools)
chain_with_prompt = prompt | llm_with_tools

def chatbot(state:State):
  return {"messages":[chain_with_prompt.invoke(state["messages"])]}

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START,"chatbot")
graph=graph_builder.compile()

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

user_input="Hi there!, My name is John"

events=graph.stream(
     {"messages": [("user", user_input)]},stream_mode="values"
)

for event in events:
  event["messages"][-1].pretty_print()


st.title("GPT Clone")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"],base_url = "https://api.aimlapi.com/v1")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's up???????"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],   
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})