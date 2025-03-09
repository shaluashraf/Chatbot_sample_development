


import streamlit as st
import requests

st.title("Chatbot Development")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    files = {"file": uploaded_file}
    response = requests.post("http://fastapi:8000/upload/", files=files)




    if response.status_code == 200:
        st.success(response.json().get("message"))
    else:
        st.error(f"Upload failed: {response.text}")  # Show error message from API

# Chat Section
query = st.text_input("Ask something:")
if st.button("Send"):
    response = requests.post("http://127.0.0.1:8000/query/", data={"query": query})

    if response.status_code == 200:
        st.write("Bot:", response.json().get("response"))
    else:
        st.error(f"Query failed: {response.text}")  # Show error message
        
