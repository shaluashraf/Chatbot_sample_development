# Chatbot_sample_development


This project is a chatbot application built using **FastAPI** for the backend and **Streamlit** for the frontend.  
It allows users to upload PDFs, extract text, and ask questions about the document.

## 🚀 Features  
✅ Upload PDFs and extract text  
✅ Ask questions based on document content  
✅ Uses **DeepSeek-7B** for AI-powered answers  
✅ Deployable with Docker  


## 🛠️ Setup & Installation  

### 1️⃣ Clone the Repository  
git@github.com:shaluashraf/Chatbot_sample_development.git
cd Chatbot_sample_development

## Installation
1. Clone the repository:
   git clone https://github.com/shaluashraf/Chatbot_sample_development.git
   cd Chatbot_sample_development

2.Install dependencies:
pip install -r requirements.txt

3. Run the application: Locally
   uvicorn sample:app --host 0.0.0.0 --port 8000 --reload
   streamlit run app.py

5. Docker:
docker-compose up --build 
