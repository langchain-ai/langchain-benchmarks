import json
import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain

# Text for extraction
st.set_page_config(page_title='🦜🔗 OppenAI-mer: Text extraction on the Oppenheimer characters')

from langsmith import Client
client = Client()
def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)

st.title('🦜🔗 OppenAI-mer')
st.info("LLMs are good at extracting structured output from natural lanaguge. Given a short bio of all the characters from the movie Oppenheimer, you can extract a structured summary of charecters and associated information about them using OpenAI functions and [LangChain](https://github.com/langchain-ai/langchain)")

# Show text
with open("oppenheimer.txt", "r") as file:
    oppenheimer_text = file.read()
show_text_option = st.selectbox("Show Oppenheimer character bio?", ["No", "Yes"], index=0)
if show_text_option == "Yes":
    st.text_area("Oppenheimer Text:", value=oppenheimer_text, height=400, disabled=True)

# LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Schema
default_schema = {
    "properties": {
        "name": {"type": "string"},
        "political_affiliation": {"type": "string"},
        "scientific_contribution": {"type": "string"},
    },
    "required": ["name",],
}

# Convert the schema to a formatted string
default_schema_str = json.dumps(default_schema, indent=4)

# Create a text_area in Streamlit and set the default value to the schema string
# placeholder = 'Who was in cabin C128?'
user_input_schema = st.text_area("Modify the JSON schema:", value=default_schema_str, height=200)

# Output formatting
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

            # Convert the user input string back to a JSON object
            try:
                modified_schema = json.loads(user_input_schema)
                st.success("Successfully parsed the JSON schema!")
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please correct it and try again.")


            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            chain = create_extraction_chain(modified_schema, llm)
            extraction_output = chain.run(oppenheimer_text)
            # Convert the output to a string representation
            # output_str = json.dumps(extraction_output, indent=4)
            markdown_output = json_to_markdown_table(extraction_output)
            run_id=1
            # Display the output in a text box
            # st.text_area("Extraction Output:", value=output_str, height=400, disabled=True)
            # st.info(output_str)
        
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
	
