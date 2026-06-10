# Streamlit Cloud Deployment Guide

This project is ready for deployment on **Streamlit Cloud**. Follow these steps:

## 🚀 Deploying to Streamlit Cloud

### Step 1: Create a Streamlit Cloud Account
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **Sign up** 
3. Sign in with your **GitHub account**

### Step 2: Deploy Your App
1. In Streamlit Cloud, click **"New app"**
2. Select your repository: `okothjosh/matatu_ml`
3. Select the branch: `main`
4. Set the main file path: `app.py`
5. Click **"Deploy"**

### Step 3: Configure Secrets (if needed)
- If your app uses API keys or passwords, add them in:
  - Streamlit Cloud dashboard → **Secrets**
  - Or create `.streamlit/secrets.toml` locally

### Step 4: Share Your App
- Your app will be live at: `https://share.streamlit.io/okothjosh/matatu_ml/main/app.py`
- Share this link with anyone!

## ✅ What the Deployment Does

### `.streamlit/config.toml`
This file configures Streamlit's appearance and behavior:
- **Theme colors** - Purple and light theme matching your brand
- **Font** - Sans serif for modern look
- **Error handling** - Don't show error details to users (cleaner UX)

### `.gitignore`
Prevents large/sensitive files from being uploaded:
- Python cache files (`__pycache__/`)
- Virtual environments (`venv/`)
- Model files (`*.pkl`, `*.h5`) - These can be large!
- Environment variables (`.env`)
- Cache files

## 📊 Benefits of Streamlit Cloud

✅ **Free hosting** - No credit card required  
✅ **Auto-deploy** - App updates on every GitHub push  
✅ **Always running** - No servers to manage  
✅ **Secure** - HTTPS by default  
✅ **Easy sharing** - Public URL anyone can access  

## 🔄 How Deployment Works

1. **You push code to GitHub** (this repo)
2. **Streamlit Cloud detects changes**
3. **Automatically rebuilds your app** in ~1-2 minutes
4. **App goes live** with zero downtime

## 📝 Next Steps

1. ✅ You're already set up (config.toml + .gitignore are ready)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your repo
3. Click deploy and wait ~2 minutes
4. Share the link!

---

## Troubleshooting

**App won't deploy?**
- Check the **Logs** in Streamlit Cloud dashboard
- Verify `requirements.txt` has all dependencies
- Make sure `app.py` is in the root directory

**App runs locally but not in cloud?**
- Add missing dependencies to `requirements.txt`
- Check file paths (use relative paths, not absolute)
- Ensure CSV files are in the repo (not in .gitignore)

**Want to add secrets/API keys?**
- Don't commit sensitive info to GitHub
- Use Streamlit Cloud's **Secrets** feature instead
