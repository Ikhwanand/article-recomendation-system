"""
This script contains the main function to run the Streamlit app.
"""

from pathlib import Path
import pandas as pd 
import numpy as np 

import streamlit as st 
import pickle
from dotenv import load_dotenv

import torch 
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from streamlit_feedback import streamlit_feedback
from trubrics.integrations.streamlit import FeedbackCollector


from constants import PATH_EMBEDDINGS, PATH_SENTENCES, PATH_DATA_BASE
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer


import requests
from io import BytesIO
from PyPDF2 import PdfReader
from urllib.parse import urlparse

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

from streamlit_chat import message
from chat import main as chat_page

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

try:
    HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
except:
    raise Exception('Set environment variable HOPSWORKS_API_KEY')



################ Helper functions ###################


# @st.cache_resource
# def init_trubrics() -> FeedbackCollector:
#     """
#     Initialize an environment and authenticate with the Trubrics platform account.
    
#     This function connects to the Trubrics platform using the project name stored in the `secrets.toml` file as an environment variable.
#     The Trubrics account authentication is assumed to be set with 'single_user' mode.
    
#     Environment Variables:
#         TRUBRICS_CONFIG_EMAIL: Email for Trubrics account, stored in the `secrets.toml` file.
#         TRUBRICS_CONFIG_PASSWORD: Password for Trubrics account, stored in the `secrets.toml` file.
#         TRUBRICS_PROJECT_NAME: Project name to initiate on the Trubrics platform, also stored in the `secrets.toml` file in a [general] section.
        
#     Returns:
#         FeedbackCollector: A FeedbackCollector object authenticated with Trubrics platform.

#     """

#     with st.spinner("Connecting to the Trubrics platform..."):
#         pass
#     return FeedbackCollector(project="paper-recommender-system")



def get_paper_abstracts() -> dict:
    """
    Fetch paper abstracts from a CSV file.

    Returns:
        dict: A dictionary mapping paper titles to their abstracts.
    """
    
    df = pd.read_csv(f"{PATH_DATA_BASE}/filtered_data.csv")
    paper_abstracts = dict(zip(df['titles'], df['abstracts']))
    
    return paper_abstracts


def get_paper_urls() -> dict:
    """
    Fetch paper URLs from a CSV file.

    Returns:
        dict: A dictionary mapping paper titles to their URLs.
    """
    
    df = pd.read_csv(f"{PATH_DATA_BASE}/filtered_data.csv")
    paper_urls = dict(zip(df['titles'], df['urls']))

    return paper_urls


def get_sentences_data(path: str) -> list:
    """
    Load sentence data from a pickle file.
    
    Args:
        path (str): The location of the pickle file to read from.

    Returns:
        list: A list of sentences loaded from the pickle file.
    """
    
    with open(path, 'rb') as f:
        sentences = pickle.load(f)
    
    return sentences


def get_embedding_data(path: str) -> np.array:
    """
    Load embedding data from a pickle file.
    
    Args:
        path (str): The location of the pickle file to read from.

    Returns:
        numpy.array: An array of embeddings loaded from the pickle file.
    """
    
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
    
    return embeddings


@st.cache_resource
def connect_to_hopsworks() -> tuple:
    """
    Connects to Hopsworks feature store using a provided API key.

    Returns:
        tuple: A tuple containing the Hopsworks project and feature store objects.
    """
    with st.spinner("Connecting to Hopsworks Feature Store..."):
        try:
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
            fs = project.get_feature_store()
            
            print("Connected to the Hopsworks Feature Store")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    return project, fs



@st.cache_resource
def get_data_from_hopsworks(project, feature_store) -> pd.DataFrame:
    """
    Fetches data from the 'papers_info' feature group.

    Returns:
        pd.DataFrame: A DataFrame containing the features from the 'papers_info'
                      feature group. If the connection or data fetch fails, it 
                      will return an empty DataFrame.
    """
    
    try:
        feature_store = project.get_feature_store()
        print("Connected to the Hopsworks Feature Store")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    with st.spinner("Fetching data from Hopsworks Feature Store..."):
        feature_group = feature_store.get_feature_group("paper_info", version=1)
    # Pull the feature group as a Pandas DataFrame
    df = feature_group.read()
    return df 


def get_model() -> SentenceTransformer:
    """
    This function creates a SentenceTransformer model using the 'all-MiniLM-L6-v2' configuration and returns it.

    Returns:
        SentenceTransformer: A SentenceTransformer model with 'all-MiniLM-L6-v2' configuration.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model



def generate_summary(abstract):
    """
    Generates a summary of a given abstract using a pre-trained BART model.

    If the length of the abstract is less than 100 characters, the function returns the original abstract.
    Otherwise, it processes the abstract through the tokenizer and model to produce a concise summary.

    Args:
        abstract (str): The input abstract to be summarized.

    Returns:
        str: The summarized text of the abstract.
    """

    if len(abstract) < 100:
        return abstract
    parser = PlaintextParser.from_string(abstract, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summarizer.stop_words = ["and", "the", "is", "in", "it", "of", "to", "for", "on", "with", "as", "by", "this", "that", "be", "are", "was", "were", "not", "can", "will", "have", "has", "had", "do", "does", "did", "if", "else", "while", "when", "where", "how", "who", "what", "which", "or", "but"]
    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    summary_text = " ".join(str(sentence) for sentence in summary)
    return summary_text


def get_paper_id(abstract_url):
    parsed = urlparse(abstract_url)
    path = parsed.path
    parts = path.split('/')
    paper_id = parts[-1]
    return paper_id 


def build_pdf_url(paper_id):
    return f"https://arxiv.org/pdf/{paper_id}.pdf"




def get_paper_content(abstract_url):
    paper_id = get_paper_id(abstract_url)
    pdf_url = build_pdf_url(paper_id)
    response = requests.get(pdf_url)
    if response.status_code != 200:
        return "Failed to fetch PDF content."
    try:
        pdf_content_bytes = response.content
        return pdf_content_bytes
    except Exception as e:
        return f"Failed to extract text from PDF: {str(e)}"


DB_FAISS_PATH = "./vectorstore/db_faiss"
class FileIngestor:
    def __init__(self, pdf_content_bytes):
        self.pdf_content_bytes = pdf_content_bytes

    def handle_file_and_ingest(self):
        # Create a temporary file to hold the PDF content
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(self.pdf_content_bytes)
            tmp_file_path = tmp_file.name

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(file_path=tmp_file_path)
        data = loader.load()

        # Create embeddings using Sentence Transformers
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)
        return db


############ Main Function #########


def main_page() -> None:
    html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """
  
    st.title("Paper Recommender System: AI-powered Paper Recommendations")
    
    # Create a feedback collector
    # collector = init_trubrics()
    
    # Connect to Hopsworks Feature Store
    # project, fs = connect_to_hopsworks()
    
    with st.sidebar:
        st.markdown(
            """
            # About
            Locating relevant scientific papers to read can be a daunting task, especially
            when delving into a new field.
            
            
            To address this, we have created a paper recommender system that leverages
            the power of AI to provide personalized recommendations based on your interests.
            
            You can also chat with the AI to ask questions about the papers you are interested in that support file PDF or Docs.
            """
        )
        st.markdown(html_temp.format('rgba(55, 53, 47, 0.16)'),
                    unsafe_allow_html=True)
        
        # Todo: add 'by abstracts' option to the sidebar
        option = st.selectbox(
            'How would you like to calculate similarity?',
            ('Using titles', 'Using abstracts'),
            index=0
        )
        
        if option == 'Using abstracts':
            st.info("This option is coming soon. 'Using titles' has been selected by default.")
            option = 'Using titles' # set to default or another option
            
        
        # You comment all code below if you don't want to use feature chat with AI for low cost of memory
        if st.button(f"ChatPDF with AI"):
            # Set query parameters for chat page
            st.query_params.from_dict({"page":"chat"})
                    # Prevent the rest of the page from being executed
            st.stop()
            
        st.markdown("""
                    # How it work
                    Simply enter a topic of interest, select the preferred method for calculating similarity, and the system will recommend relevant papers for you to read.
                    
                    For instance, if you want to read about Transformers in Computer Vision, type:
                    
                    "I'm interested in papers about Transformers in Computer Vision."
                    """)
        st.markdown(html_temp.format('rgba(55, 53, 47, 0.16)'),
                    unsafe_allow_html=True)
        # st.markdown("""
        #             Inspire by [yassine-rd](https://github.com/yassine-rd) make with ❤️
                    
        #             """, unsafe_allow_html=True) 
        
    
    # Load sentence and embedding data if the user chooses to calculate similarity by titles
    if option == 'Using titles':
        sentences = get_sentences_data(f"{PATH_SENTENCES}/sentences.pkl")
        embeddings = get_embedding_data(f"{PATH_EMBEDDINGS}/embeddings.pkl") 
        
    
    # Todo: load sentence and embedding data if the user chooces to calculate similarity by abstracts
    
    # Load the SentenceTransformer model
    model = get_model()
    
    # Fetch data from Hopsworks Feature Store
    
    # Load paper abstracts and URLs
    paper_abstracts = get_paper_abstracts()
    paper_urls = get_paper_urls()
    
    # Initializations for session state variables
    if 'paper_recommendations' not in st.session_state:
        st.session_state['paper_recommendations'] = []
    
    
    # Include a form to receive a topic of interest from the user
    form = st.form(key='my_form')
    topic_of_interest = form.text_input(label='Enter your topic of interest here 👇',
                                        placeholder="I'm interested in papers about Transformers in Computer Vision")
    number_of_paper = form.number_input(label='Number of papers to recommend', min_value=1, max_value=10, value=5)
    submit = form.form_submit_button(label='Recommend papers')
    
    # If user submits a topic of interest
    if submit:
        # Check if input is not empty
        if topic_of_interest.strip() == '':
            st.error("Please enter a topic of interest.")
        else:
            # Calculate cosine similarity between the user's input embedding and the pre-calculated embeddings of paper titles 
            cosine_scores = util.cos_sim(
                embeddings, model.encode(topic_of_interest)
            )
            
            # Get the indices of the top 4 papers with highest consine scores
            topic_of_interest = torch.topk(cosine_scores, dim=0, k=number_of_paper, sorted=True)
            
            # Store the top 4 paper titles into session state to keep them persistent
            st.session_state['paper_recommendations'] = [
                sentences[i.item()] for i in topic_of_interest.indices
            ]
            
            # Reset feedback submission status in session state to False when new recommendations are made
            st.session_state['feedback_submitted'] = False
            
    
    # If there are paper reccommendations in session state
    if st.session_state['paper_recommendations']:
        # Display recommended papers
        st.subheader('Here are some papers you might be interested in reading:')
        for paper_title in st.session_state['paper_recommendations']:
            with st.expander(paper_title):
                abstracts = paper_abstracts.get(paper_title, "Abstract not found")
                
                # Generate and display summary
                summary = generate_summary(abstracts)
                st.markdown(f"**Summary:** {summary}")
                
                # st.markdown(paper_abstracts) # Optionally display the full abstract
                paper_url = paper_urls.get(paper_title, "URL not found")
                st.markdown(f"Read the full paper here: {paper_url}")
                
                

# Main function to route pages
def main():
    st.set_page_config(page_title="AI Research Assistant", page_icon=":brain:", layout='wide')
    params = st.query_params.get("page", "main")
    print(params)
    
    if params == "chat":
        chat_page()
    else:
        main_page()



                
if __name__=='__main__':
    main()      