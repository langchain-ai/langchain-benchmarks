import pandas as pd
import streamlit as st
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI

df = pd.read_csv("titanic.csv")


llm = ChatOpenAI(temperature=0)
agent = create_pandas_dataframe_agent(llm, df, agent_type=AgentType.OPENAI_FUNCTIONS)


from langsmith import Client

client = Client()


def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)


st.set_page_config(page_title="🦜🔗 Ask the CSV App")
st.title("🦜🔗 Ask the CSV App")
st.info(
    "Most 'question answering' applications run over unstructured text data. But a lot of the data in the world is tabular data! This is an attempt to create an application using [LangChain](https://github.com/langchain-ai/langchain) to let you ask questions of data in tabular format. For this demo application, we will use the Titanic Dataset. Please explore it [here](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) to get a sense for what questions you can ask. Please leave feedback on well the question is answered, and we will use that improve the application!"
)

query_text = st.text_input("Enter your question:", placeholder="Who was in cabin C128?")
# Form input and query
result = None
with st.form("myform", clear_on_submit=True):
    submitted = st.form_submit_button("Submit")
    if submitted:
        with st.spinner("Calculating..."):
            response = agent({"input": query_text}, include_run_info=True)
            result = response["output"]
            run_id = response["__run"].run_id
if result is not None:
    st.info(result)
    col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
    with col_text:
        st.text("Feedback:")
    with col1:
        st.button("👍", on_click=send_feedback, args=(run_id, 1))
    with col2:
        st.button("👎", on_click=send_feedback, args=(run_id, 0))
