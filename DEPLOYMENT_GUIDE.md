# 🚀 Deployment Checklist for Streamlit Community Cloud

## ✅ Pre-Deployment Checklist

- [ ] All required files are present:
  - [ ] `app.py` (main application)
  - [ ] `model.pkl` (trained model)
  - [ ] `vectorizer.pkl` (TF-IDF vectorizer)
  - [ ] `requirements.txt` (dependencies)
  - [ ] `setup.sh` (NLTK data setup)
  - [ ] `.streamlit/config.toml` (optional config)
  - [ ] `README.md` (documentation)

- [ ] Code is working locally:
  ```bash
  streamlit run app.py
  ```

- [ ] All dependencies are listed in `requirements.txt`

- [ ] Model files are under 1GB (for free tier)

## 📦 GitHub Setup

1. **Initialize Git** (if not already done)
   ```bash
   git init
   ```

2. **Add all files**
   ```bash
   git add .
   ```

3. **Commit**
   ```bash
   git commit -m "Initial commit - Spam Detection App by Shrish Mishra"
   ```

4. **Create repository on GitHub**
   - Go to https://github.com/new
   - Name it: `spam-detection-model`
   - Don't initialize with README (you already have one!)
   - Click "Create repository"

5. **Push to GitHub**
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/spam-detection-model.git
   git push -u origin main
   ```

## 🌐 Streamlit Cloud Deployment

1. **Visit Streamlit Cloud**
   - Go to: https://share.streamlit.io
   - Sign in with GitHub

2. **Create New App**
   - Click "New app" button
   - Repository: `YOUR-USERNAME/spam-detection-model`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL (optional): Choose a custom name

3. **Advanced Settings** (optional)
   - Python version: 3.9
   - Secrets: None needed for this app

4. **Deploy!**
   - Click "Deploy" button
   - Wait 2-3 minutes for first deployment
   - Your app will be live at: `https://your-app-name.streamlit.app`

## 🎉 Post-Deployment

- [ ] Test the deployed app
- [ ] Share the URL with others
- [ ] Add the deployed URL to your GitHub README
- [ ] Monitor app analytics on Streamlit dashboard

## 🔄 Making Updates

After deployment, any changes pushed to GitHub will automatically redeploy:

```bash
# Make changes to your code
git add .
git commit -m "Update: description of changes"
git push
```

Streamlit will detect the changes and redeploy automatically! 🚀

## 📊 Resource Limits (Free Tier)

- **RAM**: 1 GB
- **CPU**: Shared
- **Apps**: Unlimited public apps
- **Storage**: Model files should be < 100MB (yours are likely much smaller)
- **Uptime**: May sleep after inactivity, wakes up on visit

## 🆘 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'nltk'"
**Solution**: Make sure `requirements.txt` includes nltk

### Issue: "Resource punkt not found"
**Solution**: The `setup.sh` file should handle this. Make sure it's in your repo.

### Issue: "FileNotFoundError: model.pkl not found"
**Solution**: 
- Make sure `model.pkl` and `vectorizer.pkl` are committed to Git
- Check `.gitignore` doesn't exclude `.pkl` files
- Verify file names match exactly in `app.py`

### Issue: App is slow or crashes
**Solution**: 
- Check model file size (should be < 100MB)
- Optimize your code
- Consider upgrading to paid tier if needed

## 🎯 Success Indicators

✅ App loads without errors
✅ Can input text and get predictions
✅ Predictions are correct
✅ App is accessible via public URL
✅ No timeout errors

---

**Author**: Shrish Mishra
**Project**: Email/SMS Spam Classifier
**Deployment Date**: October 2025
