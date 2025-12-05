# 🌍 PastPortals - Your Gateway to World History

**PastPortals** is an intelligent historical exploration platform that brings world history to life through AI-powered conversations, real historical images, and interactive museum guides. Built with React and powered by Google Gemini AI, it provides an immersive journey through time and civilizations.

---

<img width="1920" height="1080" alt="Screenshot (71)" src="https://github.com/user-attachments/assets/69c94a9e-a70c-4bef-bc8a-b66dbd33cb70" />


## ✨ Key Features

### 🤖 **AI-Powered Historical Guide**
- Intelligent conversations about any historical topic, event, or civilization
- Powered by Google Gemini 2.5 Flash for accurate, comprehensive responses
- Context-aware answers with minimum 500+ word detailed explanations
- Real-time fact-checking using Wikipedia integration

### 🖼️ **Real Historical Images**
- Automatic image sourcing from Wikimedia Commons
- Wikipedia-style text wrapping layout with authentic historical photographs
- 3-4 curated images per topic with source attribution
- Professional grid layout optimized for reading

### 🌐 **Multi-Language Translation**
- Real-time translation to 18+ languages
- Support for: English, Hindi, French, Spanish, Portuguese, Arabic, Chinese, Japanese, German, Italian, Russian, Korean, and more
- Seamless language switching without page reload

### 🎤 **Voice Search Integration**
- Hands-free search using Web Speech API
- Visual feedback with color-coded status (Red: listening, Blue: ready)
- Works in Chrome, Edge, and Brave browsers
- Green search button for manual queries

### 🏛️ **Interactive Museum Explorer**
- Virtual tours of world-famous museums
- Featured institutions: Louvre, British Museum, National Museum India, Egyptian Museum, Smithsonian, Palace Museum
- Direct links to official museum websites
- Detailed highlights and establishment history

### 📜 **Historical Timeline Navigation**
- Explore major historical periods and events
- Interactive timeline with categorized eras
- Quick-access topic tags for popular searches

### 🔍 **Smart Search System**
- Wikipedia-powered search with AI enhancement
- Auto-search from topic tags and suggestions
- Search available on all 4 pages (Home, Search, Timeline, Museums)
- Instant results with comprehensive explanations

### 📊 **Professional UI/UX**
- Clean, modern design with light theme
- Elegant sidebar navigation
- Responsive layout for mobile, tablet, and desktop
- Smooth animations and hover effects
- Professional typography and spacing

<img width="1920" height="1080" alt="Screenshot (72)" src="https://github.com/user-attachments/assets/92c0d878-a898-4ed6-a4a0-7c4b77fecc38" />


<img width="1920" height="1080" alt="Screenshot (73)" src="https://github.com/user-attachments/assets/03514f7f-3242-40a6-9021-93511e2b9467" />


## 🎯 Use Cases

- **Students & Educators**: Research historical topics with verified sources
- **History Enthusiasts**: Deep dive into civilizations, wars, and cultural movements
- **Museum Visitors**: Pre-visit research and virtual museum exploration
- **Language Learners**: Study history in multiple languages
- **Content Creators**: Gather accurate historical information with citations

---

## 🏗️ Technical Architecture

### **Frontend Stack**
- **Framework**: React 18.2.0
- **Routing**: React Router DOM 6.20.1
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Styling**: Custom CSS with CSS Variables
- **API Integration**: Fetch API with async/await

### **Backend Stack**
- **Framework**: Flask (Python)
- **AI Model**: Google Gemini 2.5 Flash
- **APIs**: Wikipedia API, Wikimedia Commons API
- **CORS**: Flask-CORS for cross-origin requests

### **Project Structure**
```
PastPortals/
├── frontend/                    # React Application
│   ├── public/
│   │   └── index.html          # HTML template
│   ├── src/
│   │   ├── components/         # React Components
│   │   │   ├── Home.jsx        # Landing page
│   │   │   ├── SearchPageNew.jsx      # Main search interface
│   │   │   ├── TimelinePageNew.jsx    # Timeline explorer
│   │   │   ├── MuseumsPageNew.jsx     # Museum directory
│   │   │   ├── VoiceSearchBar.jsx     # Voice search component
│   │   │   ├── Header.jsx      # Navigation header
│   │   │   └── Sidebar.jsx     # Navigation sidebar
│   │   ├── contexts/           # React Context
│   │   │   ├── APIContext.jsx  # API state management
│   │   │   └── NotificationContext.jsx
│   │   ├── styles/             # CSS Styling
│   │   │   ├── globals.css     # Global styles
│   │   │   ├── components.css  # Component styles
│   │   │   └── sidebar.css     # Sidebar styles
│   │   ├── utils/              # Utilities
│   │   │   ├── api.js          # API functions
│   │   │   └── imageSearch.js  # Wikimedia image fetcher
│   │   ├── App.jsx             # Main app component
│   │   └── index.js            # Entry point
│   └── package.json            # Dependencies
│
├── backend/                    # Flask API Server
│   ├── app.py                  # Main Flask app
│   ├── config.py               # Configuration
│   ├── routes/                 # API Routes
│   │   ├── qa_routes.py        # Q&A endpoints
│   │   ├── translate_routes.py # Translation
│   │   ├── summarize_routes.py # Summarization
│   │   ├── museum_routes.py    # Museum data
│   │   └── config_routes.py    # Config endpoints
│   ├── utils/                  # Backend utilities
│   │   ├── ai_utils.py         # Gemini AI integration
│   │   ├── wikipedia_utils.py  # Wikipedia API
│   │   └── museum_utils.py     # Museum data
│   ├── requirements.txt        # Python dependencies
│   └── venv/                   # Virtual environment
│
├── .env                        # Environment variables (API keys)
├── START_APP.bat               # Launch server (background mode)
├── STOP_APP.bat               # Stop server
├── CHECK_STATUS.bat           # Check server status
├── RESTART_APP.bat            # Restart server
└── README.md                  # This file
```

---

## 🚀 Quick Start Guide

### **Prerequisites**

Before running PastPortals, ensure you have:

1. **Node.js** (v16 or higher)
   - Download: https://nodejs.org
   - Verify: `node --version`

2. **Google Gemini API Key**
   - Get free key: https://aistudio.google.com/app/apikey
   - Required for AI features

### **Installation Steps**

#### **Option 1: Easy Launch (Recommended)**

1. **Double-click** `START_APP.bat`
2. Wait 10-15 seconds for server to start
3. Browser opens automatically at `http://localhost:3000`
4. **Close the terminal** - server keeps running in background!

#### **Option 2: Manual Setup**

```bash
# 1. Navigate to project directory
cd "C:\Users\DELL\Desktop\Code\Projects\Ai Musem Guide"

# 2. Install frontend dependencies
cd frontend
npm install

# 3. Configure API Key
# Create .env file in root directory:
echo REACT_APP_GEMINI_API_KEY=your_api_key_here > .env

# 4. Start the server
npm start

# Server runs at http://localhost:3000
```

### **Environment Configuration**

Create a `.env` file in the root directory:

```env
REACT_APP_GEMINI_API_KEY=your_actual_api_key_here
```

*(Replace `your_actual_api_key_here` with your API key from [Google AI Studio](https://aistudio.google.com/app/apikey))*

---

## 📖 How to Use

### **1. Home Page**
- Overview of PastPortals features
- Quick access to all sections
- Featured topics and civilizations

### **2. Search Historical Topics**
- Click "Search" in sidebar
- Type your question or use voice search 🎤
- Click green search button or press Enter
- View AI-generated response with real historical images
- Images appear on the right (Wikipedia-style layout)
- Text wraps naturally around images

### **3. Explore Timeline**
- Navigate through historical periods
- Click on era cards to learn more
- Filter by civilization, war, empire, religion, art

### **4. Visit Museums**
- Browse 6 world-famous museums
- Click museum cards to expand details
- View highlights and artifacts
- Click "Learn More" to visit official museum website

### **5. Voice Search**
- Click the microphone button (turns RED when listening)
- Speak your question clearly
- Button turns BLUE when ready
- Works best with Chrome/Edge
- *Note: Brave browser requires internet connection*

### **6. Language Translation**
- Click language selector (top of search page)
- Choose from 18+ languages
- Content translates instantly
- UI language changes automatically

---

## 🛠️ Server Management

### **Start Server**
```bash
# Windows Batch File
START_APP.bat

# Or PowerShell
START_APP.ps1
```

### **Stop Server**
```bash
STOP_APP.bat
```

### **Check Status**
```bash
CHECK_STATUS.bat
```

### **Restart Server**
```bash
RESTART_APP.bat
```

### **Manual Commands (PowerShell)**
```powershell
# Start
cd frontend
npm start

# Stop all Node processes
Get-Process -Name node | Stop-Process -Force

# Check if running
Get-Process -Name node -ErrorAction SilentlyContinue
```

---

## 🎨 Features in Detail

### **Wikipedia-Style Layout**
- Images float on the right side (320px max-width)
- Text wraps around images naturally
- Professional borders and captions
- Source attribution for all images
- Responsive design: stacks on mobile

### **Image Sources**
- **Wikimedia Commons API**: Free historical photographs
- CORS-friendly, no API key required
- Filters: JPEG/PNG only, minimum 400x300px
- Maximum 3-4 images per search
- Captions with title and source

### **AI Response Quality**
- Minimum 500 words per response
- Structured sections:
  1. Historical background
  2. Key events and timeline
  3. Cultural significance
  4. Impact on civilization
  5. Interesting facts
  6. Modern relevance

---

## 🔧 Troubleshooting

### **Server won't start?**
```bash
cd frontend
npm install
npm start
```

### **Images not loading?**
- Check internet connection
- Wikimedia Commons requires online access
- Images load from external CDN

### **Voice search not working?**
- Use Chrome or Edge browser (best support)
- Enable microphone permissions
- Brave browser requires internet connection
- Check browser console for errors

### **API errors?**
- Verify `.env` file exists in root
- Check API key is valid
- Ensure `REACT_APP_GEMINI_API_KEY` is set
- Restart server after changing .env

### **Port 3000 already in use?**
```powershell
# Find process on port 3000
netstat -ano | findstr :3000

# Kill the process (replace PID)
taskkill /F /PID <PID_NUMBER>
```

---

## 📚 API Endpoints

### **Frontend → Backend Communication**

```javascript
// Search historical topics
POST /api/qa
Body: { question: "Tell me about Roman Empire" }

// Translate content
POST /api/translate
Body: { text: "content", targetLanguage: "hi" }

// Summarize text
POST /api/summarize
Body: { text: "long content" }

// Get museum data
GET /api/museums

// Update configuration
POST /api/config
Body: { apiKey: "new_key" }
```

---

## 🌟 Key Technologies

- **React 18** - Modern UI framework
- **Google Gemini AI** - Advanced language model
- **Wikipedia API** - Historical data source
- **Wikimedia Commons** - Historical image repository
- **Web Speech API** - Voice recognition
- **React Router** - Client-side routing
- **Framer Motion** - Smooth animations
- **CSS Variables** - Theming system

---

## 📝 License

This project is for educational and research purposes. All historical content sourced from Wikipedia and Wikimedia Commons under Creative Commons licenses.

---

## 👥 Credits

- **Historical Data**: Wikipedia Foundation
- **Images**: Wikimedia Commons
- **AI Model**: Google Gemini
- **Museums**: Official museum databases
- **UI Icons**: Lucide React

---

## 🚀 Future Enhancements

- [ ] Dark theme toggle
- [ ] Bookmark favorite topics
- [ ] Export to PDF functionality
- [ ] Advanced filtering and search
- [ ] User accounts and history
- [ ] More museum integrations
- [ ] Interactive 3D artifacts
- [ ] Offline mode support

---

## 📞 Support

For issues, questions, or contributions:
- GitHub: [PastPortals - AI](https://github.com/ykjaat6104/PastPortals-AI)
- Report bugs via GitHub Issues
- Check `SERVER_GUIDE.md` for detailed server documentation

---

**PastPortals** - *Bringing History to Life Through Technology* 🌍✨
