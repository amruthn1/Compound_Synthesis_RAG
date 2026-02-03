# 🧱 Materials Science RAG Platform

## Standalone A100-Ready Materials Discovery System
### CIF Generation • Property Prediction • Safety-Enforced Synthesis • Colab + Streamlit

---

## 🎯 Overview

This is a **complete, production-ready materials science platform** that combines:

- **Llama-3.1** for reasoning and RAG
- **Qdrant** for local vector database
- **Crystal-Text-LLM** inspired CIF generation
- **MatGL/AlignFF** for property prediction
- **Mandatory safety enforcement** for all synthesis protocols
- **Single shared backend** used by both Colab and Streamlit

### Key Features

✅ **Fully local** - No API keys required  
✅ **A100 GPU compatible** - Optimized for Google Colab  
✅ **Real models** - No mocks or placeholders  
✅ **Safety enforced** - All synthesis includes detailed hazard protocols  
✅ **Single source of truth** - Pipeline shared between Colab and UI  
✅ **Literature-grounded** - Retrieves and cites scientific papers  

---

## 🏗️ Architecture

```
materials_rag/
├── colab_setup.ipynb          # Google Colab notebook (COMPLETE)
├── streamlit_app.py            # Web UI (calls shared pipeline)
├── requirements.txt            # All dependencies
│
├── data/
│   ├── reactions.csv           # Sample materials database
│   └── papers/                 # Scraped papers (populated on demand)
│
├── ingestion/                  # Data ingestion modules
│   ├── parse_reactions.py      # Formula parsing
│   ├── precursor_extraction.py # Precursor inference
│   ├── paper_scraper.py        # PubMed/arXiv scraping
│   └── embed_and_store.py      # Vector embeddings + Qdrant
│
├── rag/                        # RAG system
│   ├── retriever.py            # Literature retrieval
│   └── llama_agent.py          # Llama-3.1 agent
│
├── crystal/                    # Crystal structure
│   ├── composition_editing.py  # Element substitution
│   └── cif_generation.py       # CIF file generation
│
├── prediction/                 # Property prediction
│   ├── alignff_predict.py      # AlignFF predictor
│   └── matgl_predict.py        # MatGL M3GNet predictor
│
├── synthesis/                  # Synthesis generation
│   ├── hazard_detection.py     # Chemical hazard database
│   └── synthesis_generator.py  # Protocol generation with MANDATORY safety
│
├── pipeline/                   # SINGLE SHARED BACKEND
│   └── run_pipeline.py         # ⚠️ THE ONLY SOURCE OF TRUTH
│
└── utils/
    └── sample_data.py          # Sample data generation
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. **Upload project to Colab**:
   ```bash
   # Upload all files to /content/ in Colab
   ```

2. **Open and run** `colab_setup.ipynb`

3. **That's it!** The notebook will:
   - Detect GPU (A100 if available)
   - Install all dependencies
   - Load all models
   - Run example pipelines
   - Launch Streamlit UI

### Option 2: Local Setup

1. **Clone/Download** this repository

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Or use Python directly**:
   ```python
   from pipeline.run_pipeline import MaterialsPipeline
   
   pipeline = MaterialsPipeline()
   result = pipeline.run_materials_pipeline(
       composition="BaTiO3",
       substitutions={"Ti": "Zr"},
       generate_cif=True,
       predict_properties=True,
       generate_synthesis=True
   )
   ```

---

## 📖 Usage Examples

### Example 1: Basic Material Synthesis

```python
result = pipeline.run_materials_pipeline(
    composition="BaTiO3",
    generate_synthesis=True
)

# Output includes:
# - Precursors: ['BaCO3', 'TiO2']
# - CIF file
# - Predicted properties
# - Full synthesis protocol with safety
```

### Example 2: Element Substitution

```python
result = pipeline.run_materials_pipeline(
    composition="BaTiO3",
    substitutions={"Ti": "Zr"},  # Creates BaZrO3
    generate_cif=True,
    predict_properties=True
)

# Warnings about ionic radius changes
# Modified synthesis parameters
# Updated property predictions
```

### Example 3: High-Hazard Materials

```python
result = pipeline.run_materials_pipeline(
    composition="LaF3",  # Contains fluorine
    generate_synthesis=True
)

# Automatic detection of fluoride hazards
# Enhanced safety protocols including:
# - Calcium gluconate requirement
# - HF burn procedures
# - Emergency protocols
```

---

## 🔒 Safety Enforcement (CRITICAL)

### Absolute Rules

1. **NO synthesis without safety protocols**
2. **ALL hazards must be detected and documented**
3. **Fixed output format MUST be maintained**
4. **Literature sources MUST be cited**

### Safety Features

- **Comprehensive hazard database** for 20+ elements
- **Severity levels**: High, Medium, Low
- **Element-specific precautions**:
  - Fluorine → Calcium gluconate gel requirement
  - Lithium → Pyrophoric handling
  - Lead → Blood monitoring recommendations
  - Beryllium → Specialized training requirement

### Output Format (ENFORCED)

```
================================================================================
ANSWER WITH SAFETY PROTOCOLS
================================================================================
## Synthesis Protocol for [Material]

1. SAFETY PROTOCOLS
   [Mandatory - comprehensive safety section]

2. MATERIALS AND EQUIPMENT
   [Literature-derived precursors]

3. DETAILED SYNTHESIS PROCEDURE
   [Step-by-step with safety notes]

4. CHARACTERIZATION
   [Required analytical methods]

5. NOTES & LIMITATIONS
   [Assumptions and optimization guidance]

================================================================================
RETRIEVED CONTEXT SOURCES
================================================================================
[Paper citations with DOI/PMID]
```

---

## 🧪 Models Used

### LLM (Reasoning + RAG)
- **Llama-3.1-8B-Instruct** (or 70B)
- Loaded via Hugging Face Transformers
- 8-bit quantization for GPU efficiency

### Embeddings
- **all-MiniLM-L6-v2** (SentenceTransformers)
- Local embedding generation
- 384-dimensional vectors

### Vector Database
- **Qdrant** (local/in-memory)
- COSINE similarity
- Persistent storage support

### CIF Generation
- **Crystal-Text-LLM** inspired approach
- Prototype-based structure inference
- Literature-grounded lattice parameters

### Property Prediction
- **MatGL (M3GNet)** - Primary (if available)
- **AlignFF** - Fallback with composition heuristics
- Predicts: formation energy, band gap, density, etc.

---

## 🎓 Scientific Accuracy

### Literature Grounding
- Scrapes PubMed and arXiv
- Embeds and indexes papers
- Retrieves relevant context for each query
- Cites sources in all outputs

### Validation
- XRD for phase confirmation
- Composition verification (EDS/XPS/ICP)
- Property predictions based on ML models
- Synthesis conditions from literature

### Limitations
- Synthesis parameters are estimates
- Experimental optimization required
- Property predictions are computational
- Phase diagrams not fully considered

---

## 🔬 Technical Details

### Pipeline Flow

```
Input (Formula + Substitutions)
    ↓
Parse Composition
    ↓
Apply Substitutions → Validate
    ↓
Infer Precursors
    ↓
Retrieve Literature (RAG)
    ↓
Generate CIF ← Crystal-Text-LLM
    ↓
Predict Properties ← MatGL/AlignFF
    ↓
Detect Hazards ← Hazard Database
    ↓
Generate Synthesis ← Llama + Safety Enforcement
    ↓
Output (PipelineResult)
```

### Key Design Principles

1. **Single Shared Backend**
   - `pipeline/run_pipeline.py` is the ONLY source of logic
   - Colab and Streamlit are UI layers only
   - Prevents logic duplication and drift

2. **Mandatory Safety**
   - Enforced at synthesis generation
   - Cannot be bypassed or disabled
   - Fixed output format

3. **Literature Grounding**
   - All synthesis must cite sources
   - Retrieved papers included in output
   - Explicit marking of inferred vs. literature values

4. **Reproducibility**
   - Deterministic pipeline
   - Same inputs → same outputs
   - Version-controlled dependencies

---

## 📊 System Requirements

### Minimum
- Python 3.9+
- 8GB RAM
- 10GB disk space

### Recommended
- Python 3.10+
- GPU with 16GB VRAM (e.g., A100, V100)
- 50GB disk space (for models)

### Optimal (Google Colab)
- A100 GPU (40GB VRAM)
- High-RAM runtime
- Pro/Pro+ subscription for longer sessions

---

## 🐛 Troubleshooting

### Issue: Models not loading
**Solution**: Ensure you have Hugging Face access to Llama-3.1
```bash
huggingface-cli login
```

### Issue: Out of memory
**Solution**: Use 8-bit quantization
```python
pipeline = MaterialsPipeline(use_4bit=True)
```

### Issue: No papers retrieved
**Solution**: Populate database first
```python
result = pipeline.run_materials_pipeline(
    composition="BaTiO3",
    scrape_papers=True  # This will take time
)
```

### Issue: Streamlit not launching in Colab
**Solution**: Use localtunnel
```bash
!streamlit run streamlit_app.py &
!npx localtunnel --port 8501
```

---

## 📝 Citation

If you use this platform in your research, please cite:

```bibtex
@software{materials_rag_2026,
  title={Materials Science RAG Platform},
  author={[Your Name]},
  year={2026},
  url={https://github.com/yourusername/materials-rag}
}
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:

1. Maintain the single-pipeline architecture
2. Do not duplicate logic between Colab and Streamlit
3. Preserve safety enforcement
4. Add tests for new features
5. Update documentation

---

## ⚠️ Disclaimer

This software is for **research and educational purposes only**.

- Always validate computational predictions experimentally
- Follow proper laboratory safety protocols
- Consult with experts before handling hazardous materials
- The authors are not responsible for improper use

---

## 🎯 Success Criteria Checklist

- ✅ Runs in Google Colab
- ✅ A100 GPU compatible
- ✅ No API keys required
- ✅ Real models (Llama, Qdrant, MatGL)
- ✅ CIF generation with literature grounding
- ✅ Property predictions
- ✅ Synthesis with MANDATORY safety
- ✅ Single shared pipeline (no duplication)
- ✅ Identical outputs from Colab and Streamlit
- ✅ Literature citation in all outputs
- ✅ Comprehensive hazard detection

---

## 📚 References

1. **Llama 3.1**: Meta AI (2024)
2. **Crystal-Text-LLM**: Facebook Research
3. **MatGL (M3GNet)**: MaterialsProject
4. **Qdrant**: Vector Database
5. **PubMed**: NCBI Literature Database

---

**Built with ❤️ for materials scientists**

For questions or issues, please open a GitHub issue.
