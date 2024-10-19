📢 **Document-Based Chatbot with Appointment Booking and Callbacks**

🚀 This project is a **Streamlit-based chatbot** application that allows users to upload documents (PDF files) and interact with them. The chatbot can:
- Answer questions based on the uploaded document using **language models**.
- Handle **appointment booking** by extracting date information from text.
- Handle **callback requests** by collecting user details.

🔍 Additionally, the project integrates **Natural Language Processing (NLP)** functionalities for text-based document analysis and user interaction using libraries like **LangChain** and **Hugging Face Embeddings**.

---

💡 **Features:**
- **PDF Document Interaction**: Upload a PDF file, and ask questions based on its content.
- **Appointment Booking**: Automatically detect appointment requests and validate dates.
- **Callback Requests**: Collect and store user contact details for callback purposes.
- **Natural Language Processing**: Uses language models like **LLama2** and **FAISS** for document search and Q&A.

---

🔧 **Technologies Used:**
- **Python 3.x**
- **Streamlit**: Frontend UI for chatbot interaction.
- **LangChain**: Framework for developing language model-driven applications.
- **Hugging Face**: Embeddings for document vectorization.
- **FAISS**: Efficient similarity search for document-based queries.
- **Dateparser**: Natural language date extraction and parsing.
- **Ollama**: LLM chain integration for answering document-based queries.

---

⚙️ **Setup Instructions:**

1. **Prerequisites:**
   - Python 3.8 or higher
   - Streamlit
   - Install dependencies from `requirements.txt`

2. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/document-chatbot.git
   cd document-chatbot
