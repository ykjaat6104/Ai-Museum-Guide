# 🖼️ AI Museum Guide 🎨

An intelligent museum tour guide system powered by cutting-edge AI — enabling document-driven answers, voice responses, and AI-generated imagery. Built with **React** (frontend) and **Flask** (backend), this app brings a futuristic art museum experience to your fingertips.

---

## 🚀 Features

- 🧠 **Text-to-Text (RAG)**: Ask questions and get context-aware answers from uploaded documents
- 🔊 **Text-to-Speech**: Converts answers into audio for an interactive voice guide
- 🎨 **Text-to-Image**: Generate AI-powered images from descriptive prompts

---

## 🛠️ Tech Stack

| Layer         | Technology                        |
|---------------|------------------------------------|
| Frontend      | React.js, Streamlit (for demo)     |
| Backend       | Flask (Python)                     |
| Embeddings    | Sentence Transformers              |
| Vector Store  | FAISS                              |
| Text-to-Speech| gTTS, Google Speech Recognition    |
| Image Gen     | Stability AI (Stable Diffusion)    |
| LLM           | Mistral 7b + Ollama                |

---

## 🧠 System Architecture
![Presentation1](https://github.com/user-attachments/assets/2cb64074-7325-4197-8a53-fc8f0aeda702)



### 🔹 Text-to-Text: Retrieval-Augmented Generation

![image](https://github.com/user-attachments/assets/ef82bd55-5b54-46d8-9927-823d2712196d)


🔍 This part powers intelligent Q&A by:
- Splitting documents into chunks
- Generating embeddings
- Retrieving the most relevant chunks
- Feeding context to LLMs for accurate answers

---

### 🔹 Text-to-Speech: Voice Guide


![Presentation1](https://github.com/user-attachments/assets/faa64a68-4a9e-4095-a529-a43f9e28e22c)



🎤 Converts user voice to text → RAG query → answer to audio:
- Uses `distil-whisper-large-v3` for speech-to-text
- `gTTS` for generating spoken answers

---

### 🔹 Text-to-Image: Artistic Interpretation
![image](https://github.com/user-attachments/assets/556bdccb-ed0d-4d1e-8d6c-6339dfaddd32)


🎨 Converts descriptive prompts (via RAG) to high-quality images using **Stability AI**

---

## 📸 Results
![image](https://github.com/user-attachments/assets/24264fd1-8553-46cf-b8b0-406a3edf7faf)
![image](https://github.com/user-attachments/assets/4eed577c-ced6-407c-a267-3f322032bb84)

![image](https://github.com/user-attachments/assets/31935d01-43cc-44b9-b1fb-4f69e4f1ea10)
![image](https://github.com/user-attachments/assets/75ebd59a-9cf9-4325-904e-3313eab7b13e)



---

## 💻 Running Locally

### Setup

```bash
# Frontend
cd frontend
npm install
npm start

# Backend
cd backend
pip install -r requirements.txt
python app.py
