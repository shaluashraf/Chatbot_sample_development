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
git clone https://github.com/yourusername/chatbot-fastapi-streamlit.git
cd chatbot-fastapi-streamlit

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/chatbot-project.git
   cd chatbot-project

2.Install dependencies:
pip install -r requirements.txt

3. Run the application:
 uvicorn testing1:app --host 0.0.0.0 --port 8000 --reload
   streamlit run app.py

5. Deployment with Docker
Build the Docker image:
docker build -t chatbot .

6. Run the Docker container:
docker run -p 8501:8501 chatbot
