import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_chat import message
import docx

load_dotenv()
os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))



def get_files_text(uploaded_files):
    """
    Take a list of uploaded files and return a single string with the text content of all files.

    The function checks the file extension of each uploaded file and calls the appropriate helper function to extract the text content. The text content is then appended to a master string.

    Args:
        uploaded_files (list): List of uploaded files

    Returns:
        str: The text content of all files in the list
    """
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            st.info(f"Unsupported file format: {file_extension}")
    return text


def get_pdf_text(pdf):
    """
    Extract text from a PDF file.

    This function reads a PDF file using the PyPDF2 library and extracts the text content from each page. It concatenates all the text and returns it as a single string.

    Args:
        pdf: A file-like object representing the PDF file.

    Returns:
        str: The extracted text content from the PDF.
    """

    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    """
    Extract text from a DOCX file.

    This function reads a DOCX file using the docx library and extracts the text content from each paragraph. It concatenates all the text and returns it as a single string.

    Args:
        file: A file-like object representing the DOCX file.

    Returns:
        str: The extracted text content from the DOCX.
    """
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text



def get_text_chunks(text):
    """
    Split a large text into smaller chunks for processing.

    This function takes a large text and splits it into smaller chunks of 10000 characters each. The chunks are chosen such that there is a 1000 character overlap between each chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks



def get_embeddings(chunks):
    """
    Compute dense vector embeddings for a list of text chunks.

    This function takes a list of text chunks and computes dense vector embeddings for each chunk using the Google Generative AI Embeddings model. The embeddings are then saved to a local FAISS index file.

    Args:
        chunks (list): A list of text chunks.

    Returns:
        None
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_texts(chunks, embedding=embeddings)
    vectors.save_local("faiss_index")



def get_conversational_chain():
    """
    Construct a conversational chain for answering questions from a given context.

    The constructed chain uses the Gemini-1.5-Flash model and the provided prompt template.

    The prompt template is a string that contains placeholders for the context and question. The placeholders are replaced by the actual values during runtime.

    The chain is then loaded using the load_qa_chain function from the langchain library.

    Args:
        None

    Returns:
        A conversational chain that can be used to answer questions from a given context.
    """

    prompt_temp = '''
    Answer the question from the provided context as best you can in English or Bahasa Indonesia. Try to answer in as detailed manner as possible from the provided context.
    If the answer to the question is not known from the provided context, then dont provide wrong answers, in that case just say,
    'Answer to the question is not available in the provided document. Feel free to ask question from the provided context.'
    Context:\n{context}?\n
    Question:\n{question}\n
    '''
    prompt = PromptTemplate(
        template=prompt_temp,
        input_variables=['context', 'question']
    )

    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.5)

    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain



def get_response(user_input):
    """
    This function takes a user's input and returns a response based on the context from the documents that are most similar to the user's input.

    The function takes the user's input and uses the GoogleGenerativeAIEmbeddings model to convert it into a dense vector embedding. The function then uses the FAISS library to search for documents in the local FAISS index that are most similar to the user's input.

    The function then constructs a conversational chain using the Gemini-1.5-Flash model and the prompt template from the get_conversational_chain function. The chain is then loaded using the load_qa_chain function from the langchain library.

    The function then uses the loaded chain to generate a response based on the most similar documents and the user's input. The response is then printed to the console and displayed to the user using the message function.

    Args:
        user_input (str): The user's input.

    Returns:
        None
    """
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embedding, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_input)  # conversion of 'user_input' into embeddings happens implicitly within the 'similarity_search' method
    chain = get_conversational_chain()
    response = chain(
        {'input_documents': docs, 'question': user_input},
        return_only_outputs=True
    )
    print(response)
    message(response['output_text'], is_user=False)




def main():
    """
    This is the main function of the Streamlit app. It sets the page title, 
    icon, and layout. It then creates a chat input field for the user to
    ask a question. If the user enters a question, the function calls the
    get_response function to get a response based on the most similar
    documents to the user's input. The response is then printed to the
    console and displayed to the user using the message function.

    The function also creates a file uploader in the sidebar to upload
    PDFs. If the user uploads PDFs and clicks the 'Submit and Process'
    button, the function calls the get_files_text function to extract
    text from the uploaded PDFs, and the get_text_chunks function to
    chunk the text into sentences. The function then calls the
    get_embeddings function to compute dense vector embeddings for the
    text chunks. Finally, the function displays a success message to
    the user when the process is complete.

    """
    
    # st.set_page_config(page_title='Chat With PDF Using Gemini', page_icon='🤖', layout='wide')
    st.title('Chat With PDF Using Gemini 🤖')
    user_question = st.chat_input(
        'Ask a question related to the uploaded PDFs.'
    )
    if user_question and len(user_question) > 0:
        get_response(user_question)
    with st.sidebar:
        st.title('Upload PDF 📄')
        documents = st.file_uploader(
            label='Upload PDFs',
            label_visibility='hidden',
            type=['pdf', 'docx'],
            accept_multiple_files=True
        )
        if st.button('Submit and Process'):
            with st.spinner('In Process...'):
                text = get_files_text(documents)
                chunks = get_text_chunks(text)
                get_embeddings(chunks)
                st.success('DONE!')
        
        if st.button("Back"):
            st.query_params.clear()
            st.stop()


if __name__ == '__main__':
    main()
