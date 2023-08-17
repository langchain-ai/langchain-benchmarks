import streamlit as st
from langsmith import Client
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain

st.set_page_config(page_title='🦜🔗 Text-to-graph extraction')
client = Client()
def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)

st.title('🦜🔗 Text-to-graph playground')
st.info("This playground explores the use of [OpenAI functions](https://openai.com/blog/function-calling-and-other-api-updates) and [LangChain](https://github.com/langchain-ai/langchain) to build knowledge graphs from user-input text. It breaks down the user input text into knowledge graph triples of subject (primary entities or concepts in a sentence), predicate (actions or relationships that connect subjects to objects), and object (entities or concepts that interact with or are acted upon by the subjects).")

# Input text (optional default)
oppenheimer_text=''''Julius Robert Oppenheimer, often known as Robert or "Oppie", is heralded as the father of the atomic bomb. Emerging from a non-practicing Jewish family in New York, he made several breakthroughs, such as the early black hole theory, before the monumental Manhattan Project. His wife, Katherine “Kitty” Oppenheimer, was a German-born woman with a complex past, including connections to the Communist Party. Oppenheimer\'s journey was beset by political adversaries, notably Lewis Strauss, chairman of the U.S. Atomic Energy Commission, and William Borden, an executive director with hawkish nuclear ambitions. These tensions culminated in the famous 1954 security hearing. Influential figures like lieutenant general Leslie Groves, who had also overseen the Pentagon\'s creation, stood by Oppenheimer\'s side, having earlier chosen him for the Manhattan Project and the Los Alamos location. Intimate relationships, like that with Jean Tatlock, a Communist and the possible muse behind the Trinity test\'s name, and colleagues like Frank, Oppenheimer\'s physicist brother, intertwined with his professional life. Scientists such as Ernest Lawrence, Edward Teller, David Hill, Richard Feynman, and Hans Bethe were some of Oppenheimer\'s contemporaries, each contributing to and contesting the atomic age\'s directions. Boris Pash\'s investigations, and the perspectives of figures like Leo Szilard, Niels Bohr, Harry Truman, and others, framed the broader sociopolitical context. Meanwhile, individuals like Robert Serber, Enrico Fermi, Albert Einstein, and Isidor Isaac Rabi, among many others, each played their parts in this narrative, from naming the atomic bombs to pivotal scientific contributions and advisory roles. All these figures, together with the backdrop of World War II, McCarthyism, and the dawn of the nuclear age, presented a complex mosaic of ambitions, loyalties, betrayals, and ideologies.oppenheimer_short.txt'''

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
user_input_text = st.text_area("Enter your text (<2000 characters):", height=200)
if len(user_input_text) > MAX_CHARS:
    st.warning(f"Text is too long. Processing only the first {MAX_CHARS} characters")
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
	
