import os
import streamlit as st
import transformers
import torch
import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
import re
import PyPDF4
from typing import List
import os
import pdfplumber
import docx2txt
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredDocxLoader

st.set_page_config(page_title="Equinor Data Catalog", page_icon=":mag:")
# Display the image
st.image("equinor.png", width=200)


def qa(q, c):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    from langchain.llms import HuggingFaceHub
    from langchain.embeddings import HuggingFaceHubEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain import VectorDBQA
    from langchain.chains import VectorDBQA
    from langchain.document_loaders import UnstructuredPDFLoader
    from langchain.prompts import PromptTemplate
    

    loader = UnstructuredPDFLoader("C:\\Users\\hamse\\Downloads\\jodi_standard2.pdf")
    docs= loader.load()
    
    flan_ul2 = HuggingFaceHub(
        repo_id="google/flan-ul2", model_kwargs={"temprature": 0.9, "max_new_tokens": 500}
    )


    text_splitter = CharacterTextSplitter(chunk_size=520, chunk_overlap=100)
    texts = text_splitter.split_text(docs[0].page_content)

    embeddings = HuggingFaceHubEmbeddings()

    if len(texts) == 0:
        st.warning(f"No text was found in the document: {file.name}. Skipping this document.")
    docsearch = Chroma.from_texts(texts, embeddings)

    prompt_template = """
    Given the following context containing multiple subheaders and their corresponding information, make sure to extract and print out the information under a specific subheader when requested. 
    The user will provide the name of the subheader they are interested in, and the you should output the corresponding information. 
    You should also handle variations in the subheader names and document formatting. 
    If the requested subheader is not found, return an appropriate message. 
    Please ensure that each output is clearly labeled and presented in a user-friendly manner.

    It is very important to make sure that you include everything under each subheader. 
    -----------------------------------------------------------------------------------
    For example: 
    (if the input is:
    
    Timeliness
    • data is updated every month
    • New publications every week.

    Output will be:
    • data is updated every month
    • New publications every week.)
    ---------------------------------------------------------------------------------------
    Have this in mind when looking for answers to questions.
    ------------------------------------------------------------------------------------------------------------------------
    Use the following pieces of context to answer the question at the end. If the information can not be found in the context given, just say "i dont know".
    Don't try to make up an answer. Make sure to only use the context given in {summaries} to answer the question.

    {summaries}

    Question: {question}
    Answer in English:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["summaries", "question"])

    qa = VectorDBQA.from_chain_type(llm=flan_ul2, chain_type="stuff", vectorstore=docsearch)
    
    #docss = docsearch.similarity_search(query=q)
    query=q
    return qa.run(query)
    return chain({"input_documents": docss, "question": q}, return_only_outputs=True)


# Define a function to get a list of uploaded files in the current directory
def get_uploaded_files():
    files = []
    for item in os.listdir("."):
        if os.path.isfile(item):
            files.append(item)
    return files

# Define a function to read the contents of a file
def read_file(file):
    with open(file, "r") as f:
        content = f.read()
    return content


# Get a list of uploaded files
filess = get_uploaded_files()

# Create the search bar
def clean_text(text: str) -> str:
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Replace newline characters with a space
    text = text.replace('\n', ' ')
    return text


# Show live suggestions based on the search term as the user types
suggestions = [file for file in filess if file.endswith(('.pdf'))]
# Create a multiselect widget to select files based on suggestions
selected_files = st.multiselect("Search:", suggestions)

# Display the selected files and their contents
if selected_files:
    st.write("Selected files:")
    for file in selected_files:
        st.write(file)
        content = read_file(file)
        st.text(content)

# User inputs

files = st.file_uploader("Choose a file", accept_multiple_files=True, type=['pdf'])
question = st.text_input('Enter a question:')

# Define your list of questions
questions_list = ["Is the data from the IEA updated regularly each month?",
            "Does the application check for updates every hour?",
            "Is the time delay considered insignificant for the current use of the data?",
            "Is the data static or volatile?"]

# Check if the session state already has questions_and_answers attribute; if not, create an empty dictionary
if "questions_and_answers" not in st.session_state:
    st.session_state["questions_and_answers"] = {}

# Process the uploaded files in a loop and check for the document type to process each document
if files:
    for file in files:
        file_extension = file.name.split('.')[-1]
        text = ''
        if file_extension == 'txt':
            content = file.read().decode('utf-8')
            text += content
        elif file_extension == 'docx':
            content = docx2txt.process(file)
            text += content
        elif file_extension == 'pdf':
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()

    if file.name not in st.session_state["questions_and_answers"]:
            st.session_state["questions_and_answers"][file.name] = {}

        # Iterate through all predefined questions
    textt=clean_text(text)
    st.write(textt)
    for predefined_question in questions_list:
            # Get the answer for the current file
            answer = qa(predefined_question, file)

            # Store the answer in the session state
            st.session_state["questions_and_answers"][file.name][predefined_question] = answer

# Convert the results to a pandas dataframe
df_results = pd.DataFrame(st.session_state["questions_and_answers"]).T

# Display the results in a table
if not df_results.empty:
    st.write('Data Description:')
    st.table(df_results)