
###############################################################
# +=========================================================+ #
# |     AI Powered by Openai Api - CSV Asking About         | #
# +=========================================================+ #
# | Author   : JOSE TEOTONIO DA SILVA NETO [TEO]            | #
# | Objective: Build a simple using openai Api              | #
# | Version  : 1.0.0.0                                      | #
# +=========================================================+ #
# | Name   | Changed At | Description                       | #
# +=========================================================+ #
# | Teo    | 24/09/2023 | Build Starter Version             | #
# +=========================================================+ #
###############################################################

# +=========================================================+ #
# | Libraries necessaries to execute current project        | #
# +=========================================================+ #
import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# +=========================================================+ #
# | Page Title of current project                           | #
# +=========================================================+ #
st.set_page_config(page_title='🦜🔗 Ask the Data App')
st.title('🦜🦜🦜🔗 AsCSV on Importedpppppp')

# +=========================================================+ #
# | Load CSV File Content                                   | #
# +=========================================================+ #


def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    with st.expander('See DataFrame'):
        st.write(df)
    return df

# +=========================================================+ #
# | Generate LLM response                                   | #
# +=========================================================+ #


def generate_response(csv_file, input_query):
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo-0613',
        temperature=0.1,
        verbose=True,
        openai_api_key=openai_api_key)

    df = load_csv(csv_file)

    # Create Pandas DataFrame Agent
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)

    # Perform Query using the Agent
    response = agent.run(input_query)
    return st.success(response)


# +=========================================================+ #
# | Imput Widgets                                           | #
# +=========================================================+ #
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
question_list = [
    'How many rows are there?',
    'Other']
query_text = st.selectbox('Select an example query:',
                          question_list, disabled=not uploaded_file)
openai_api_key = st.text_input(
    'OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))

# +=========================================================+ #
# | App Logic [Read, Interprete and Response]               | #
# +=========================================================+ #
if query_text is 'Other':
    query_text = st.text_input(
        'Enter your query:', placeholder='Enter query here ...', disabled=not uploaded_file)
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
if openai_api_key.startswith('sk-') and (uploaded_file is not None):
    st.header('Output')
    generate_response(uploaded_file, query_text)
