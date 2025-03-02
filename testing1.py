import os
import fitz  # PyMuPDF for text extraction
from fastapi import FastAPI, UploadFile, File, Form
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

app = FastAPI()
from llama_index.core import Settings
Settings.llm = None  # Ensure OpenAI is not used

UPLOAD_DIR = "uploaded_docs"
INDEX_STORAGE_DIR = "index_storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)

# Load Embedding Model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

index = None  # Global index variable


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Extract text from PDF
    text_content = extract_text_from_pdf(file_path)

    if not text_content.strip():
        return {"error": "No extractable text found in the document."}

    # Create a document index
    global index
    from llama_index.core import Document

# Convert extracted text into a Document object
    document = Document(text=text_content)

    # Pass the document into the index
    index = VectorStoreIndex.from_documents([document], embed_model=embed_model)

    # index = VectorStoreIndex.from_documents([text_content], embed_model=embed_model)
    index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)

    return {"message": f"File '{file.filename}' uploaded and indexed successfully!"}


@app.post("/query/")
async def query_index(query: str = Form(...)):
    global index
    if index is None:
        try:
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
            index = load_index_from_storage(storage_context)
        except Exception:
            return {"error": "No documents indexed yet. Please upload a file first."}

    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    return {"response": str(response)}






# ########################       using mistralai/Mistral-7B-v0.1 models : https://huggingface.co/mistralai/Mistral-7B-v0.1 
# # import os
# # from fastapi import FastAPI, UploadFile, File, Form
# # from fastapi.staticfiles import StaticFiles
# # from fastapi.responses import FileResponse, HTMLResponse
# # from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
# # from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# # from transformers import pipeline


# # hf_token = os.getenv("fdsdPFe")  # Set this in your environment variables

# # Load Mistral-7B model
# # llm_pipeline = pipeline(
# #     "text-generation", 
# #     model="mistralai/Mistral-7B-v0.1",  
# #     token=hf_token  # Use the authentication token
# # )

# # 
# # Settings.llm = None  # Ensures LlamaIndex does not use OpenAI API

# #Initialize FastAPI app
# # app = FastAPI()

# # Define directories for uploaded files and indexing
# # UPLOAD_DIR = "uploaded_docs"
# # INDEX_STORAGE_DIR = "index_storage"
# # os.makedirs(UPLOAD_DIR, exist_ok=True)
# # os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)

# #Use SentenceTransformer for embedding
# # embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Initialize index (set after file upload)
# # index = None

# #File Upload & Indexing API**
# # @app.post("/upload/")
# # async def upload_file(file: UploadFile = File(...)):
# #     file_path = os.path.join(UPLOAD_DIR, file.filename)
# #     with open(file_path, "wb") as buffer:
# #         buffer.write(file.file.read())

# #     documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()
    
# #     global index
# #     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
# #     index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
    
# #     return {"message": f"File '{file.filename}' uploaded and indexed successfully!"}

# # #  **Query Index API**
# # @app.post("/query_index/")
# # async def query_index(query: str = Form(...)):
# #     global index
# #     if index is None:
# #         try:
# #             storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
# #             index = load_index_from_storage(storage_context)
# #         except Exception:
# #             return {"error": "No documents indexed yet. Please upload a file first."}

# #     query_engine = index.as_query_engine()
# #     response = query_engine.query(query)
    
# #     return {"response": str(response)}

# # # **Chatbot API (Mistral-7B)**
# # @app.post("/chatbot/")
# # async def chatbot_query(query: str = Form(...)):
# #     response = llm_pipeline(query, max_length=200)
# #     return {"response": response[0]["generated_text"]}

# # #**Serve HTML Frontend**
# # @app.get("/", response_class=HTMLResponse)
# # async def serve_homepage():
# #     with open("static/index.html", "r") as f:
# #         return HTMLResponse(content=f.read())

# # # 🗂 **Serve Static Files (CSS, JS, Images)**
# # app.mount("/static", StaticFiles(directory="static"), name="static")



# #####################

# ###########using deepseek model   : https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat 
# # import torch
# # from fastapi import FastAPI, Form
# # from transformers import pipeline

# # app = FastAPI()


# # llm_pipeline = pipeline(
# #     "text-generation", 
# #     model="deepseek-ai/deepseek-llm-7b-chat",  
# #     model_kwargs={"torch_dtype": "auto"}  # Remove `device_map`
# # )

# # @app.post("/query/")
# # async def query_index(query: str = Form(...)):
# #     response = llm_pipeline(query, max_length=500, do_sample=True)
# #     return {"response": response[0]["generated_text"]}


