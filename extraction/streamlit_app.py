import streamlit as st
from langsmith import Client
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain

st.set_page_config(page_title='🦜🔗 AutoGraph: Triple extraction')
client = Client()
def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)

st.title('🦜🔗 AutoGraph: Triple extraction')
st.info("LLMs are good at extracting structured output from natural lanaguge. This playground will extract knowledge graph triplets from the user input text using [OpenAI functions](https://openai.com/blog/function-calling-and-other-api-updates) and [LangChain](https://github.com/langchain-ai/langchain). Knowledge graph triplets help to represent relationships between entities in a structured manner.")

# Input text
with open("oppenheimer_short.txt", "r") as file:
    oppenheimer_text = file.read()

# LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Knowledge triplet schema
default_schema = {
    "properties": {
        "subject": {"type": "string"},
        "predicate": {"type": "string"},
        "object": {"type": "string"},
    },
    "required": ["subject", "predicate", "object"],
}

# Create a text_area, set the default value to oppenheimer_text
MAX_CHARS = 2000  # Maximum number of characters
user_input_text = st.text_area("Enter your text (<2000 characters):", value=oppenheimer_text, height=200)
if len(user_input_text) > MAX_CHARS:
    st.warning(f"Text is too long. Only {MAX_CHARS} characters allowed!")
    user_input_text = user_input_text[:MAX_CHARS]

# Output formatting of triples
def json_to_markdown_table(json_list):
    if not json_list:
        return "No data available."
    
    # Extract headers
    headers = json_list[0].keys()
    markdown_table = " | ".join(headers) + "\n"
    markdown_table += " | ".join(["---"] * len(headers)) + "\n"
    
    # Extract rows
    for item in json_list:
        row = " | ".join([str(item[header]) for header in headers])
        markdown_table += row + "\n"
    
    return markdown_table

# Form input and query
markdown_output = None
with st.form('myform', clear_on_submit=True):

    submitted = st.form_submit_button('Submit')
    if submitted:
		
        with st.spinner('Calculating...'):

            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            chain = create_extraction_chain(default_schema, llm)
            extraction_output = chain(user_input_text,include_run_info=True)
            markdown_output = json_to_markdown_table(extraction_output['text'])
            run_id = extraction_output["__run"].run_id
        
# Feeback
if markdown_output is not None:
    st.markdown(markdown_output)
    col_blank, col_text, col1, col2 = st.columns([10, 2,1,1])
    with col_text:
        st.text("Feedback:")
    with col1:
        st.button("👍", on_click=send_feedback, args=(run_id, 1))
    with col2:
        st.button("👎", on_click=send_feedback, args=(run_id, 0))
	
