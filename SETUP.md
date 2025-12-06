# AI Museum Guide - Setup Guide

## 📋 Prerequisites

Before you begin, make sure you have:
- **Python 3.8+** - [Download here](https://www.python.org/downloads/)
- **Node.js 16+** - [Download here](https://nodejs.org/)
- **Git** (optional) - For cloning the repository

Check if installed:
```bash
python --version
node --version
npm --version
```

---

## ⚙️ First Time Setup

### Step 1: Clone/Download the Project
```bash
git clone https://github.com/ykjaat6104/Ai-Museum-Guide.git
cd Ai-Museum-Guide
```

### Step 2: Configure API Key

Create a `.env` file in the root directory:
```bash
# Copy the example file
copy .env.example .env
```

Edit `.env` and add your Gemini API key:
```env
GEMINI_API_KEY=your-actual-gemini-api-key-here
SMITHSONIAN_API_KEY=optional-museum-api-key
```

**Important:** Never commit the `.env` file to git. It's already in `.gitignore`.

### Step 3: Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
cd ..
```

### Step 4: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 🚀 Running the Application

You need **two terminal windows** - one for backend, one for frontend.

### Terminal 1 - Start Backend
```bash
cd backend
python app.py
```
✅ Backend runs on: **http://localhost:5000**

### Terminal 2 - Start Frontend
```bash
cd frontend
npm start
```
✅ Frontend opens automatically at: **http://localhost:3000**

---

## 🛑 Stopping the Application

In each terminal window, press:
```
Ctrl + C
```

---

## 🔧 Project Structure

```
Ai-Museum-Guide/
├── backend/              # Flask API server
│   ├── app.py           # Main backend application
│   ├── requirements.txt # Python dependencies
│   ├── routes/          # API endpoints
│   └── utils/           # Helper functions
├── frontend/            # React application
│   ├── src/            # Source code
│   ├── public/         # Static files
│   └── package.json    # Node dependencies
├── .env                # Backend config (API keys) - NOT IN GIT
└── .env.example        # Template for .env file
```

---

## 🎯 Tech Stack

### Backend
- Flask (Python web framework)
- Google Gemini AI
- LangChain + RAG (Retrieval-Augmented Generation)
- FAISS (Vector database)
- Wikipedia API

### Frontend
- React 18
- React Router
- Axios
- Framer Motion

---

## ⚠️ Troubleshooting

### Port Already in Use

**Backend (Port 5000):**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
```

**Frontend (Port 3000):**
```bash
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID_NUMBER> /F
```

### Backend Dependencies Issue
```bash
cd backend
pip install -r requirements.txt --force-reinstall
```

### Frontend Dependencies Issue
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### API Key Not Working
- Check `.env` file exists in root directory
- Verify `GEMINI_API_KEY` is set correctly
- Restart the backend server after changing `.env`

---

## 🔐 Security Notes

- ✅ API keys are stored in `.env` (backend only)
- ✅ `.env` files are in `.gitignore` (not tracked by git)
- ✅ Frontend calls backend API (API key never exposed to browser)
- ❌ Never commit `.env` files to version control

---

## 📝 Quick Commands Reference

```bash
# First time setup
pip install -r backend/requirements.txt
npm install --prefix frontend

# Run backend
cd backend && python app.py

# Run frontend (new terminal)
cd frontend && npm start

# Stop
Ctrl + C (in each terminal)
```

---

## 🌐 API Endpoints

Once backend is running, available at `http://localhost:5000`:

- `GET /api/health` - Health check
- `POST /api/ask` - Ask historical questions
- `POST /api/translate` - Translate text
- `POST /api/summarize` - Summarize content
- `GET /api/museums/search` - Search museums

---

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review the `README.md` file
- Check server logs in the terminal windows

---

**That's it! You're ready to explore world history with AI.** 🎉
