# 📤 How to Upload Project Files to Google Colab

## ✅ Quick Answer: 3 Methods

---

## Method 1: Upload ZIP File (EASIEST - Recommended)

### Step 1: Create the ZIP file

Run this command in your terminal:

```bash
cd /Users/amruthnadimpally/Documents/compound_synthesis_rag/final
./create_colab_zip.sh
```

**Output**: `colab_project.zip` (108KB)

### Step 2: Upload to Colab

1. Open `colab_setup.ipynb` in Google Colab
2. Find the "📁 Clone/Setup Project Structure" section
3. **Uncomment these lines** in the upload cell:

```python
uploaded = files.upload()

# Extract the zip
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✓ Extracted to current directory")
```

4. Run the cell
5. Click "Choose Files" and select `colab_project.zip`
6. Wait for extraction (~5 seconds)

### Step 3: Verify

```python
!ls -la
```

You should see:
- pipeline/
- ingestion/
- rag/
- crystal/
- prediction/
- synthesis/
- reaction.csv
- requirements.txt

---

## Method 2: Manual Drag-and-Drop (VISUAL)

### Step 1: Open Colab File Browser

1. In Google Colab, click the **folder icon** (📁) in the left sidebar
2. You'll see the file browser

### Step 2: Upload Folders

**Drag and drop each folder** from your computer to the Colab file browser:

```
From your computer                  To Colab
-----------------                   --------
final/pipeline/          →         /content/pipeline/
final/ingestion/         →         /content/ingestion/
final/rag/               →         /content/rag/
final/crystal/           →         /content/crystal/
final/prediction/        →         /content/prediction/
final/synthesis/         →         /content/synthesis/
final/utils/             →         /content/utils/
```

### Step 3: Upload Files

Drag and drop these individual files:
- `reaction.csv`
- `requirements.txt`
- `README.md`

### Step 4: Verify Structure

In a Colab cell:
```python
!tree -L 2 -d
```

Expected output:
```
.
├── pipeline
├── ingestion
├── rag
├── crystal
├── prediction
└── synthesis
```

---

## Method 3: Google Drive Mount (PERSISTENT)

### Step 1: Upload to Google Drive

1. Upload the entire `final/` folder to your Google Drive
2. Path should be: `My Drive/final/`

### Step 2: Mount Drive in Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Navigate to Project

```python
import os
os.chdir('/content/drive/MyDrive/final')

# Verify
!pwd
!ls -la
```

### Step 4: Continue with Notebook

All files are now accessible from Drive!

**Benefit**: Files persist across Colab sessions

---

## 🔍 Verification Commands

After uploading, verify everything is in place:

```python
import os

print("Checking project structure...")
print()

required = [
    'pipeline/run_pipeline.py',
    'ingestion/parse_reactions.py',
    'ingestion/precursor_extraction.py',
    'rag/retriever.py',
    'rag/llama_agent.py',
    'crystal/composition_editing.py',
    'crystal/cif_generation.py',
    'prediction/alignff_predict.py',
    'prediction/matgl_predict.py',
    'synthesis/hazard_detection.py',
    'synthesis/synthesis_generator.py',
    'reaction.csv'
]

missing = []
for filepath in required:
    if os.path.exists(filepath):
        print(f"✓ {filepath}")
    else:
        print(f"✗ {filepath} - MISSING")
        missing.append(filepath)

print()
if missing:
    print(f"⚠ {len(missing)} files missing!")
    print("Please upload them manually.")
else:
    print("✓ All required files present!")
    print("✓ Ready to run the notebook!")
```

---

## 📦 What's in colab_project.zip?

```
colab_project.zip (108KB)
├── pipeline/
│   ├── __init__.py
│   └── run_pipeline.py
├── ingestion/
│   ├── __init__.py
│   ├── parse_reactions.py
│   ├── precursor_extraction.py
│   ├── paper_scraper.py
│   └── embed_and_store.py
├── rag/
│   ├── __init__.py
│   ├── retriever.py
│   └── llama_agent.py
├── crystal/
│   ├── __init__.py
│   ├── composition_editing.py
│   └── cif_generation.py
├── prediction/
│   ├── __init__.py
│   ├── alignff_predict.py
│   └── matgl_predict.py
├── synthesis/
│   ├── __init__.py
│   ├── hazard_detection.py
│   └── synthesis_generator.py
├── utils/
│   ├── __init__.py
│   └── sample_data.py
├── reaction.csv
├── requirements.txt
└── README.md
```

---

## 🚨 Common Issues & Solutions

### Issue: "No such file or directory: pipeline/"

**Cause**: Files uploaded to wrong location

**Solution**:
```python
# Check current directory
!pwd

# If not in /content, upload files there
# Or change to /content
import os
os.chdir('/content')
```

### Issue: "Module not found: pipeline"

**Cause**: Python can't find the modules

**Solution**:
```python
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

# Verify
print(sys.path)
```

### Issue: Upload keeps failing

**Cause**: Large files or slow connection

**Solution**: Use Google Drive method (Method 3)

### Issue: Files disappear after session ends

**Cause**: Colab doesn't persist uploaded files

**Solutions**:
- Use Google Drive mount (Method 3)
- Or re-upload each session
- Or keep the zip file and re-extract

---

## ⏱️ Upload Time Estimates

| Method | Size | Time |
|--------|------|------|
| ZIP file | 108KB | 5-10 seconds |
| Manual folders | ~100KB | 1-2 minutes |
| Google Drive | ~100KB | One-time setup |

---

## 🎯 Recommended Workflow

**For Single Use**:
1. ✅ Create zip: `./create_colab_zip.sh`
2. ✅ Upload zip to Colab
3. ✅ Extract and run

**For Multiple Sessions**:
1. ✅ Upload `final/` to Google Drive
2. ✅ Mount Drive in Colab
3. ✅ Navigate to folder
4. ✅ Run notebook (files persist!)

**For Development**:
1. ✅ Keep files in Google Drive
2. ✅ Edit locally
3. ✅ Sync to Drive (auto or manual)
4. ✅ Refresh in Colab

---

## ✅ Final Checklist

Before running the notebook:

- [ ] All 6 folders uploaded (pipeline, ingestion, rag, crystal, prediction, synthesis)
- [ ] reaction.csv uploaded
- [ ] Current directory contains the folders (`!ls` shows them)
- [ ] Python can import modules (`import sys; sys.path.insert(0, '.')`)
- [ ] verification script shows all files present

---

## 🚀 After Upload

Once files are uploaded, proceed with the notebook:

1. ✅ Initialize MaterialsPipeline
2. ✅ Run Example 1 (Ba2Cl8Ni1Pb1)
3. ✅ Run Example 2 (K2Cu4F10 substitution)
4. ✅ Run Example 3 (Li1Ni1F6 high-hazard)
5. ✅ Download results

**Total time**: ~5-10 minutes on A100 GPU

---

**Need help?** Run the verification script above to check what's missing!
