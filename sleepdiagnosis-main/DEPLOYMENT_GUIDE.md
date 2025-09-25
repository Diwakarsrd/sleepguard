# 🚀 Vercel Deployment Guide for SleepDiagnosis

## 📋 **Quick Deploy**

### **Option 1: One-Click Deploy**
Click the button below to deploy instantly:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/akarsh-2005/sleepdiagnosis)

### **Option 2: Manual Deployment**

1. **Fork/Import Repository**
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import `https://github.com/akarsh-2005/sleepdiagnosis`

2. **Configuration**
   - **Framework Preset**: Other
   - **Root Directory**: `./` (leave empty)
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty
   - **Install Command**: `pip install -r requirements.txt`

3. **Environment Variables** (Optional)
   ```
   PYTHONPATH=./backend:./api
   ```

4. **Deploy**
   - Click "Deploy"
   - Wait 2-3 minutes for deployment
   - Your app will be available at `https://your-project.vercel.app`

## 🏗️ **Project Structure for Vercel**

```
sleepdiagnosis/
├── api/                    # Vercel Serverless Functions
│   └── main.py            # FastAPI app (auto-detected)
├── backend/               # Original backend code
│   ├── main.py           # Development server
│   ├── model.py          # ML models and audio processing  
│   └── requirements.txt   # Backend dependencies
├── index.html            # Main frontend (static)
├── vercel.json           # Vercel configuration
├── requirements.txt      # Root Python dependencies
└── package.json          # Project metadata
```

## ⚙️ **Configuration Files**

### **vercel.json**
```json
{
  "version": 2,
  "name": "sleepdiagnosis",
  "builds": [
    {
      "src": "index.html",
      "use": "@vercel/static"
    },
    {
      "src": "api/main.py", 
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/main.py"
    },
    {
      "src": "/(.*\\.(css|js|png|jpg|jpeg|gif|svg|ico))",
      "dest": "/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ],
  "functions": {
    "api/main.py": {
      "runtime": "python3.9",
      "maxDuration": 30
    }
  }
}
```

### **requirements.txt**
```
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
librosa==0.10.1
matplotlib==3.8.2
scikit-learn==1.3.2
numpy==1.26.2
joblib==1.3.2
soundfile==0.12.1
```

## 🔧 **API Routes**

### **Development (localhost:8000)**
- Health: `http://localhost:8000/health`
- Analyze: `http://localhost:8000/analyze`

### **Production (Vercel)**
- Health: `https://your-app.vercel.app/api/health`
- Analyze: `https://your-app.vercel.app/api/analyze`

## 🌐 **Domain Configuration**

### **Custom Domain**
1. Go to Vercel Dashboard → Your Project → Settings → Domains
2. Add your custom domain (e.g., `sleepdiagnosis.com`)
3. Configure DNS records as instructed
4. SSL certificate is automatically provisioned

### **Environment Variables**
Set in Vercel Dashboard → Settings → Environment Variables:
```
PYTHONPATH=./backend:./api
NODE_ENV=production
```

## 🔍 **Testing Deployment**

### **Health Check**
```bash
curl https://your-app.vercel.app/api/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": false,
  "service": "SleepDiagnosis API", 
  "version": "2.0.0",
  "mp3_support": true
}
```

### **Frontend Access**
Visit `https://your-app.vercel.app` and test:
- ✅ Sleep Space photo upload
- ✅ Auto Record with MP3 conversion
- ✅ Audio file upload and analysis
- ✅ Spectrogram generation

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Build Fails**
```bash
Error: Command "pip install -r requirements.txt" failed
```
**Solution**: Check requirements.txt format and Python version compatibility

#### **API Not Working**
```bash
404 - This page could not be found
```
**Solution**: 
- Verify `api/main.py` exists
- Check vercel.json routing configuration
- Ensure Python runtime is specified

#### **Audio Processing Fails**
```bash
ModuleNotFoundError: No module named 'librosa'
```
**Solution**:
- Verify all dependencies in requirements.txt
- Check function timeout (increase to 30s)
- Ensure sufficient memory allocation

#### **MP3 Conversion Issues**
```bash
MP3 conversion failed in browser
```
**Solution**:
- Browser compatibility check
- Fallback to original format
- Check Web Audio API support

### **Performance Optimization**

#### **Function Limits**
- **Execution Time**: 30 seconds max (configured)
- **Memory**: 1024MB (default)
- **File Size**: 50MB upload limit

#### **Cold Start Optimization**
- First request may be slower (cold start)
- Subsequent requests are faster
- Keep functions warm with scheduled requests

## 📊 **Monitoring**

### **Vercel Analytics**
- Enable in Project Settings → Analytics
- Monitor function executions and errors
- Track response times and success rates

### **Logs**
- Real-time logs in Vercel Dashboard → Functions
- Debug API issues and performance
- Monitor audio processing success rates

## 🔐 **Security**

### **CORS Configuration**
```python
origins = [
    "https://sleepdiagnosis.vercel.app",
    "https://*.vercel.app",
    "http://localhost:9002"  # Development only
]
```

### **File Upload Security**
- 50MB file size limit
- Audio format validation
- Temporary file cleanup
- No persistent storage

## 🚀 **Production Checklist**

- [ ] Deploy successfully to Vercel
- [ ] Health check endpoint responds
- [ ] Audio analysis works end-to-end  
- [ ] MP3 conversion functions properly
- [ ] Spectrogram generation working
- [ ] Custom domain configured (optional)
- [ ] Analytics enabled
- [ ] Error monitoring set up
- [ ] Performance tested under load

## 🔄 **CI/CD**

### **Automatic Deployments**
- **Main Branch**: Auto-deploy to production
- **Pull Requests**: Preview deployments
- **Rollback**: Instant rollback to previous version

### **GitHub Integration**
- Connected to `https://github.com/akarsh-2005/sleepdiagnosis`
- Auto-deploys on push to main branch
- Preview URLs for feature branches

## 📞 **Support**

### **Vercel Resources**
- [Vercel Documentation](https://vercel.com/docs)
- [Python Runtime](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [FastAPI on Vercel](https://vercel.com/templates/python/fastapi)

### **Project Issues**
- GitHub Issues: `https://github.com/akarsh-2005/sleepdiagnosis/issues`
- API Documentation: Available at deployed URL + `/docs`

---

**Your SleepDiagnosis app is now ready for production deployment on Vercel! 🎉**

🌐 **Live Demo**: https://sleepdiagnosis.vercel.app  
📚 **Repository**: https://github.com/akarsh-2005/sleepdiagnosis  
🚀 **Deploy**: [![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/akarsh-2005/sleepdiagnosis)