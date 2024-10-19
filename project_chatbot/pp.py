import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import re
import dateparser
from datetime import datetime


# Load environment variables
load_dotenv()

# Ensure the API key is available
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Function to validate phone number
def validate_phone(phone):
    return re.match(r"^\+?[0-9]{7,15}$", phone)

# Function to validate email
def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# Function to extract a valid date from text
def extract_date(input_text):
    # Parse the date from the input text with settings

    date_str = re.search(r'\b(\d{4}-\d{2}-\d{2}|next\w+|tomorrow|today|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|in \d+ days)\b',input_text,re.IGNORECASE)
    
    parsed_date = dateparser.parse(date_str.group(), settings={'PREFER_DATES_FROM': 'future'})
    
    # Check if the parsed date is in the future
    if parsed_date and parsed_date > datetime.now():
        return parsed_date.strftime("%Y-%m-%d")  # Return as string in the desired format
    else:
        return None

# Function to detect if the input is an appointment request
def is_booking_request(input_text):
    booking_keywords = ["book", "appointment", "schedule"]
    return any(keyword in input_text.lower() for keyword in booking_keywords)

# Function to detect if the input is a callback request
def is_callback_request(input_text):
    callback_keywords = ["call me", "contact me", "reach me", "call back"]
    return any(keyword in input_text.lower() for keyword in callback_keywords)

# Initialize Streamlit app
st.title("Document-Based Chatbot with Appointment Booking and Callbacks")

# Initialize session state for form details and document data if not already initialized
if "user_details" not in st.session_state:
    st.session_state.user_details = {}

# User input for general queries, appointment requests, or callback requests
input_text = st.text_input("Ask something about the document, book an appointment, or request a callback (e.g., 'Book me for Sunday' or 'Call me'):")

# 1. Check if the user is requesting an appointment
if input_text and is_booking_request(input_text):
    # Attempt to extract a date from the input text
    suggested_date = extract_date(input_text)
    print(suggested_date)

    if suggested_date:
        st.write(f"It seems you would like to book an appointment on {suggested_date}. Please fill in the following details:")
    else:
        st.write("It seems you would like to book an appointment, but we couldn't identify a valid date. Please provide additional details:")

    # Create a conversational form to collect user information
    with st.form("appointment_form"):
        user_name = st.text_input("Name", value=st.session_state.user_details.get("user_name", ""))
        user_phone = st.text_input("Phone Number", value=st.session_state.user_details.get("user_phone", ""))
        user_email = st.text_input("Email", value=st.session_state.user_details.get("user_email", ""))
        user_date_input = st.text_input("Preferred Date (e.g., 'next Monday')", value=suggested_date or "")

        submit_button = st.form_submit_button(label="Submit")

    # Form submission logic for appointment booking
    if submit_button:
        # Check if all fields are filled
        if not user_name:
            st.error("Name is required.")
        elif not user_phone:
            st.error("Phone number is required.")
        elif not user_email:
            st.error("Email is required.")
        elif not validate_phone(user_phone):
            st.error("Please enter a valid phone number.")
        elif not validate_email(user_email):
            st.error("Please enter a valid email address.")
        else:
            # Try to extract a valid date from the user input
            booking_date = extract_date(user_date_input)
            if not booking_date:
                st.error("Please enter a valid future date (e.g., 'next Monday').")
            else:
                # Store user details and appointment date in session state
                st.session_state.user_details["user_name"] = user_name
                st.session_state.user_details["user_phone"] = user_phone
                st.session_state.user_details["user_email"] = user_email
                st.session_state.user_details["booking_date"] = booking_date

                st.success(f"Thank you, {user_name}! We will contact you at {user_phone} or via email at {user_email}. Your appointment is scheduled for {booking_date}.")

                # Here you can integrate a tool-agent to actually book the appointment in the backend

# 2. Check if the user is requesting a callback
elif input_text and is_callback_request(input_text):
    st.write("It seems you would like us to contact you. Please fill in the following details:")

    # Create a conversational form to collect user information for a callback
    with st.form("callback_form"):
        user_name = st.text_input("Name", value=st.session_state.user_details.get("user_name", ""))
        user_phone = st.text_input("Phone Number", value=st.session_state.user_details.get("user_phone", ""))
        user_email = st.text_input("Email", value=st.session_state.user_details.get("user_email", ""))

        submit_button = st.form_submit_button(label="Submit")

    # Form submission logic for callback request
    if submit_button:
        # Check if all fields are filled
        if not user_name:
            st.error("Name is required.")
        elif not user_phone:
            st.error("Phone number is required.")
        elif not user_email:
            st.error("Email is required.")
        elif not validate_phone(user_phone):
            st.error("Please enter a valid phone number.")
        elif not validate_email(user_email):
            st.error("Please enter a valid email address.")
        else:
            # Store user details in session state
            st.session_state.user_details["user_name"] = user_name
            st.session_state.user_details["user_phone"] = user_phone
            st.session_state.user_details["user_email"] = user_email

            st.success(f"Thank you, {user_name}! We will contact you at {user_phone} or via email at {user_email}.")

# If the user isn't requesting a callback, continue with PDF processing
else:
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Load the uploaded PDF
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully")

        # Load document using PyPDFLoader
        loader = PyPDFLoader(uploaded_file.name)
        documents = loader.load()

        # Split document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)

        # Store document data in session state
        st.session_state.document_data = final_documents

        # Embedding using Huggingface
        huggingface_embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize FAISS VectorStore for efficient search
        vectorstore = FAISS.from_documents(final_documents, huggingface_embeddings)

        # If the user inputs a query and document is already uploaded
        if input_text and st.session_state.document_data:
            ## Prompt Template
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                    "You are a knowledgeable assistant. Your task is to help the user by providing accurate and concise answers based solely on the information in the provided document. "
                    "Be clear and helpful. If the answer to the user's question is not available in the document, respond politely by saying, 'I'm sorry, the answer you're looking for isn't available in the document.' "
                    "{context}"),
                    
                    ("user", "Question: {question}")
                ]
            )

            # ollama LLAma2 LLm 
            llm = Ollama(model="llama3.2")

            # Create the LLM chain
            llm_chain = LLMChain(prompt=prompt_template, llm=llm, output_parser=StrOutputParser())

            # Use FAISS to find relevant document chunks based on the user's query
            relevant_docs = vectorstore.similarity_search(input_text, k=3)

            # Combine the relevant document chunks into a single context
            combined_docs = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Prepare the context and user query for the LLM
            context = f"Context from the document:\n{combined_docs}\n\n"
            query = f"Question: {input_text}"

            # Pass the user input and context to the chain and get the response
            response = llm_chain.run({"question": query, "context": context})

            # Display the response
            st.write(response)

if st.session_state.user_details:
    st.write(f"User Details: {st.session_state.user_details}")




