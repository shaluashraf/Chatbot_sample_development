import os
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.embeddings.huggingface import HuggingFaceEmbedding

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# **Force CPU usage**
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
device = torch.device("cpu")  # Use CPU
Settings.llm = None 
# **Directories for storing files**
UPLOAD_DIR = "uploaded_docs"
INDEX_STORAGE_DIR = "index_storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)

# **Load DeepSeek-7B model on CPU**
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch.float32, device_map=None).to(device)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=None).to(device)

model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# **Embedding Model**
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# **Initialize FastAPI app**
app = FastAPI()

# **Index storage**
index = None

# **Upload & Index File**
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()
    global index
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
    
    return JSONResponse({"message": f"File '{file.filename}' uploaded and indexed successfully!"})

# **Retrieve & Generate Response**
@app.post("/query/")
async def query_index(query: str = Form(...)):
    global index
    if index is None:
        try:
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
            index = load_index_from_storage(storage_context)
        except Exception:
            return JSONResponse({"error": "No documents indexed yet. Please upload a file first."})

    retrieved_text = str(index.as_query_engine().query(query))
    messages = [
        {"role": "system", "content": "Use the retrieved text to answer the query accurately."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": retrieved_text}
    ]
    
    # Convert input to tensor
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # Create an attention mask
    attention_mask = input_tensor.ne(tokenizer.pad_token_id).long()  # Ensure correct masking

    # Generate response
    outputs = model.generate(
        input_tensor,
        attention_mask=attention_mask,  # Explicitly pass attention mask
        max_new_tokens=150
    )
    
    response = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    
    return JSONResponse({"response": response})





####################
##using GPU

# import os
# import torch
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# # Directories
# UPLOAD_DIR = "uploaded_docs"
# INDEX_STORAGE_DIR = "index_storage"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)

# # Load DeepSeek-7B model
# model_name = "deepseek-ai/deepseek-llm-7b-chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
# model.generation_config = GenerationConfig.from_pretrained(model_name)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id

# # Embedding Model
# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Initialize FastAPI app
# app = FastAPI()

# # Index storage
# index = None

# # Upload & Index File
# @app.post("/upload/")
# async def upload_file(file: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(file_path, "wb") as buffer:
#         buffer.write(await file.read())

#     documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()
#     global index
#     index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
#     index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
    
#     return JSONResponse({"message": f"File '{file.filename}' uploaded and indexed successfully!"})

# # Retrieve & Generate Response
# @app.post("/query/")
# async def query_index(query: str = Form(...)):
#     global index
#     if index is None:
#         try:
#             storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
#             index = load_index_from_storage(storage_context)
#         except Exception:
#             return JSONResponse({"error": "No documents indexed yet. Please upload a file first."})

#     retrieved_text = str(index.as_query_engine().query(query))
#     messages = [
#         {"role": "system", "content": "Use the retrieved text to answer the query accurately."},
#         {"role": "user", "content": query},
#         {"role": "assistant", "content": retrieved_text}
#     ]
    
#     input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
#     outputs = model.generate(input_tensor.to(model.device), max_new_tokens=150)
#     response = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    
#     return JSONResponse({"response": response})
