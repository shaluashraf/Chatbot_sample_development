# Chatbot_sample_development


This is an AI-powered chatbot that allows users to upload PDF documents and ask questions based on the extracted content.

## Features
✅ Upload PDF documents  
✅ Extract text from uploaded documents  
✅ Ask questions and receive context-aware responses  
✅ Deployed using Docker  

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/chatbot-project.git
   cd chatbot-project

2.Install dependencies:
pip install -r requirements.txt

3. Run the application:
streamlit run app.py

4. Deployment with Docker
Build the Docker image:
docker build -t chatbot .

5. Run the Docker container:
docker run -p 8501:8501 chatbot
