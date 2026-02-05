# Streamlit App Deployment Guide

## Option 1: Streamlit Community Cloud (Recommended - FREE)

**Best for:** Quick deployment, free hosting, automatic updates from GitHub

### Steps:

1. **Push your code to GitHub:**
   ```bash
   cd /Users/amruthnadimpally/Documents/compound_synthesis_rag/final
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Sign up for Streamlit Community Cloud:**
   - Go to https://share.streamlit.io/
   - Sign in with your GitHub account
   - Click "New app"

3. **Deploy:**
   - Repository: Select your GitHub repo
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click "Deploy"

4. **Important Notes:**
   - The app will be public (anyone with the URL can access it)
   - Free tier limits: 1GB RAM, 1 CPU core
   - May need to reduce model to 4-bit quantization for memory limits
   - Database (Qdrant) will reset on each deploy unless you use persistent storage

### Limitations:
- **Memory:** 1GB RAM might not be enough for Qwen2.5-7B model
- **Solution:** Use a smaller model like Phi-3-mini or disable LLM features

---

## Option 2: Google Cloud Run (More Resources)

**Best for:** Apps needing more memory/compute, pay-per-use

### Prerequisites:
- Google Cloud account
- gcloud CLI installed

### Steps:

1. **Create a Dockerfile:**
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Deploy to Cloud Run:**
   ```bash
   gcloud init
   gcloud run deploy materials-rag \
     --source . \
     --platform managed \
     --region us-central1 \
     --memory 8Gi \
     --cpu 4 \
     --timeout 3600
   ```

3. **Access your app:**
   - Cloud Run will provide a URL like: `https://materials-rag-xxxxx-uc.a.run.app`

### Cost Estimate:
- ~$0.05 per hour when running
- Auto-scales to zero when not in use
- Pay only for actual usage time

---

## Option 3: Hugging Face Spaces (AI-Focused)

**Best for:** ML/AI apps, free GPU options

### Steps:

1. **Create a new Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Select "Streamlit" as SDK
   - Choose hardware (CPU free, GPU upgrade available)

2. **Push your code:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME
   cp -r /path/to/your/app/* .
   git add .
   git commit -m "Add Streamlit app"
   git push
   ```

3. **Configure:**
   - Create `README.md` with Space metadata:
     ```yaml
     ---
     title: Materials RAG
     emoji: 🧱
     colorFrom: blue
     colorTo: green
     sdk: streamlit
     sdk_version: 1.32.0
     app_file: streamlit_app.py
     pinned: false
     ---
     ```

### Benefits:
- Free CPU tier
- GPU upgrade available ($0.60/hour for T4)
- Persistent storage
- Better for ML models

---

## Option 4: AWS EC2 (Full Control)

**Best for:** Production apps, custom configurations

### Steps:

1. **Launch EC2 instance:**
   - Instance type: t3.large or larger (8GB+ RAM)
   - OS: Ubuntu 22.04
   - Open port 8501 in security group

2. **Connect and setup:**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Install dependencies
   sudo apt update
   sudo apt install python3-pip
   
   # Clone your repo
   git clone YOUR_REPO_URL
   cd YOUR_REPO_NAME
   
   # Install requirements
   pip install -r requirements.txt
   
   # Run Streamlit
   streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
   ```

3. **Keep it running (use screen or tmux):**
   ```bash
   screen -S streamlit
   streamlit run streamlit_app.py
   # Press Ctrl+A, then D to detach
   ```

4. **Access:**
   - http://YOUR_EC2_IP:8501

### Cost:
- t3.large: ~$0.08/hour (~$60/month)
- t3.xlarge: ~$0.16/hour (~$120/month)

---

## Option 5: Replit (Quick & Simple)

**Best for:** Quick testing, simple deployments

### Steps:

1. **Create a new Repl:**
   - Go to https://replit.com
   - Click "+ Create Repl"
   - Select "Python"

2. **Upload your files:**
   - Upload all your project files
   - Create `.replit` file:
     ```toml
     run = "streamlit run streamlit_app.py"
     
     [nix]
     channel = "stable-22_11"
     ```

3. **Install dependencies:**
   - Replit will auto-detect `requirements.txt`

4. **Run:**
   - Click "Run" button
   - Replit provides a public URL

### Limitations:
- Limited resources on free tier
- May struggle with large models

---

## Recommended Approach Based on Needs:

### For Testing/Demo:
→ **Streamlit Community Cloud** or **Replit**

### For Production (with ML models):
→ **Hugging Face Spaces** (free GPU option) or **Google Cloud Run**

### For Full Control:
→ **AWS EC2** or **Google Compute Engine**

---

## Model Optimization for Cloud Deployment:

If you encounter memory issues, modify `streamlit_app.py`:

```python
# Use smaller model or disable LLM features
pipeline = MaterialsPipeline(
    llama_model_name="microsoft/Phi-3-mini-4k-instruct",  # Smaller model
    use_4bit=True,  # Enable 4-bit quantization
    # Or disable LLM entirely for property prediction only
)
```

---

## Security Considerations:

1. **Environment Variables:**
   - Store API keys in environment variables, not in code
   - Use `.env` file locally, cloud provider secrets for deployment

2. **Authentication:**
   - Add password protection if needed:
     ```python
     import streamlit as st
     
     def check_password():
         password = st.text_input("Password", type="password")
         if password == "your_password":
             return True
         return False
     
     if not check_password():
         st.stop()
     ```

3. **Rate Limiting:**
   - Implement rate limiting to prevent abuse
   - Use cloud provider's built-in rate limiting

---

## Need Help?

Let me know which deployment option you'd like to use, and I can provide more detailed instructions!
