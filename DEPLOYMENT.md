# Deployment Guide - PastPortals

## 🌐 Quick Deployment Options

### **Option 1: Render + Vercel (FREE - Recommended)**

Perfect for presentations and demos. Both services offer free tiers.

---

## 📦 Backend Deployment (Render)

### Step 1: Prepare Repository
Your repo is already prepared with:
- ✅ `Procfile` - Tells Render how to run the app
- ✅ `requirements.txt` - Python dependencies
- ✅ `runtime.txt` - Python version

### Step 2: Deploy to Render

1. **Go to** [render.com](https://render.com)
2. **Sign up** with your GitHub account
3. **Click** "New +" → "Web Service"
4. **Connect** your GitHub repository: `PastPortals-AI`
5. **Configure:**
   ```
   Name: PastPortals-backend
   Environment: Python 3
   Build Command: pip install -r backend/requirements.txt
   Start Command: cd backend && gunicorn app:app
   Instance Type: Free
   ```

6. **Add Environment Variables:**
   ```
   GEMINI_API_KEY=your-gemini-api-key-here
   FLASK_ENV=production
   SECRET_KEY=your-production-secret-key-here
   CORS_ORIGINS=https://your-frontend-url.vercel.app
   ```

7. **Click** "Create Web Service"

8. **Copy the URL** (something like: `https://pastportals-backend.onrender.com`)

---

## 🎨 Frontend Deployment (Vercel)

### Step 1: Update Frontend API URL

Before deploying, update the frontend to use your Render backend URL:

1. Edit `frontend/.env.example`:
   ```env
   REACT_APP_API_URL=https://pastportals-backend.onrender.com
   ```

### Step 2: Deploy to Vercel

1. **Go to** [vercel.com](https://vercel.com)
2. **Sign up** with your GitHub account
3. **Click** "Add New..." → "Project"
4. **Import** your repository: `PastPortals-AI`
5. **Configure:**
   ```
   Framework Preset: Create React App
   Root Directory: frontend
   Build Command: npm run build
   Output Directory: build
   ```

6. **Add Environment Variable:**
   ```
   REACT_APP_API_URL=https://pastportals-backend.onrender.com
   ```

7. **Click** "Deploy"

8. **Your site is live!** (URL: `https://pastportals.vercel.app`)

---

## 🔄 Update Backend CORS

After getting your Vercel URL, update the backend:

1. Go back to **Render Dashboard**
2. Select your backend service
3. Go to **Environment**
4. Update `CORS_ORIGINS`:
   ```
   CORS_ORIGINS=https://your-app-name.vercel.app
   ```
5. **Save Changes** (will auto-redeploy)

---

## ✅ Test Your Deployment

1. **Visit your Vercel URL**: `https://your-app-name.vercel.app`
2. **Test search**: Ask a historical question
3. **Check backend**: Visit `https://your-backend.onrender.com/api/health`

---

## 🚨 Important Notes

### Free Tier Limitations:
- **Render**: Server sleeps after 15 min inactivity (cold start ~30 seconds)
- **Vercel**: 100GB bandwidth/month (plenty for demos)

### Security Reminders:
- ✅ API key stored in Render environment variables (not in code)
- ✅ CORS configured to only allow your frontend domain
- ✅ `.env` file never committed to GitHub

---

## 🔧 Alternative: Deploy Both on Railway

**Railway** can host both backend and frontend together:

1. **Go to** [railway.app](https://railway.app)
2. **Click** "Start a New Project"
3. **Select** "Deploy from GitHub repo"
4. **Add two services:**
   - Backend (Python)
   - Frontend (Node.js)
5. **Set environment variables** for backend
6. Railway auto-detects and builds both!

**Cost**: Free $5 credit/month (enough for testing)

---

## 📊 Option 2: Netlify (Frontend) + Render (Backend)

Same process as Vercel, but use [netlify.com](https://netlify.com) for frontend.

**Netlify Deploy Settings:**
```
Base Directory: frontend
Build Command: npm run build
Publish Directory: frontend/build
```

---

## 🐳 Option 3: Docker + Any Cloud Platform

If you want more control:

1. **Create Dockerfile** for backend
2. **Build & Push** to Docker Hub
3. **Deploy** to AWS, Google Cloud, or DigitalOcean

(Advanced - ask me if you need this)

---

## 💡 Recommended for Presentation

**Best combo:**
- ✅ **Render** (Backend) - Reliable, easy setup
- ✅ **Vercel** (Frontend) - Fast, auto-deploys on git push
- ✅ **Total cost**: FREE

**Deployment time**: ~15 minutes

---

## 🆘 Troubleshooting

**Backend won't start:**
- Check Render logs for errors
- Verify all environment variables are set
- Make sure gunicorn is in requirements.txt

**Frontend can't connect:**
- Check CORS_ORIGINS matches your Vercel URL
- Verify REACT_APP_API_URL is correct
- Check browser console for errors

**Cold starts slow:**
- Render free tier sleeps after inactivity
- First request takes 20-30 seconds
- Keep backend warm with UptimeRobot (free ping service)

---

## 📞 Need Help?

- Render Docs: https://render.com/docs
- Vercel Docs: https://vercel.com/docs
- Railway Docs: https://docs.railway.app

---

**Ready to deploy?** Follow the Render + Vercel guide above! 🚀
