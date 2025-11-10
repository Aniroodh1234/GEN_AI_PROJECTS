## importing necessary tokens and environement keys
import os
from dotenv import load_dotenv
load_dotenv()

## Groq api key and setting hugging face environment
Groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")

## Langsmith tracking
os.environ["LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_PROJECT"]= "Simple Gen AI Q&A chatbot"


## importing necessay modules and library to build gen AI application
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from langchain_groq import ChatGroq
import streamlit as st


## developing the prompt
prompt = ChatPromptTemplate(
    [
        ("system","Answer all the questions accurately asked by the user"),
        ("human",'{question}')
    ]
)


def generate_response(question,api_key,engine,temperature,max_tokens):
    Groq_api_key = api_key
    ## initializing the llm model
    llm = ChatGroq(model=engine)

    ## initializing the output parser
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer


## #Title of the app
st.title("Enhanced Q&A Chatbot With Groq LLm")


## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Open AI API Key:",type="password")

## Select the groq model
engine=st.sidebar.selectbox("Select groq AI model",["openai/gpt-oss-120b","llama-3.3-70b-versatile","meta-llama/llama-4-maverick-17b-128e-instruct","openai/gpt-oss-safeguard-20b","qwen/qwen3-32b"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## MAin interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input and api_key:
    response=generate_response(user_input,api_key,engine,temperature,max_tokens)
    st.write(response)

elif user_input:
    st.warning("Please enter the Groq aPi Key in the sider bar")
else:
    st.write("Please provide the user input")
