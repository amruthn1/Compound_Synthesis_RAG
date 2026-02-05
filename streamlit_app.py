"""
Streamlit UI for Materials Science RAG Platform

CRITICAL: This is a UI LAYER ONLY. All logic is in pipeline/run_pipeline.py
This app MUST call the shared pipeline. NO logic duplication allowed.
"""

import streamlit as st
import sys
import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.run_pipeline import MaterialsPipeline, PipelineResult
from rag.openrouter_agent import OpenRouterAgent
import pandas as pd
import csv


# Load sample materials from reaction.csv
@st.cache_data
def load_sample_materials():
    """Load sample materials from reaction.csv."""
    csv_path = os.path.join(os.path.dirname(__file__), "reaction.csv")
    samples = []
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    composition = row.get('composition', '').strip()
                    if composition:
                        samples.append(composition)
        except Exception as e:
            print(f"Warning: Could not load reaction.csv: {e}")
    
    # Fallback samples if CSV not available
    if not samples:
        samples = ["K2Cu4F10", "Li1Ni1F6", "Ba2Cl8Ni1Pb1"]
    
    return samples


def get_suggested_substitutions(composition):
    """Get suggested element substitutions based on composition."""
    from pymatgen.core import Composition
    
    try:
        comp = Composition(composition)
        elements = [str(el) for el in comp.elements]
        
        # Common substitution patterns
        substitution_map = {
            'Cu': ['Ni', 'Ag', 'Zn', 'Co'],
            'Ni': ['Cu', 'Co', 'Fe', 'Zn'],
            'Fe': ['Co', 'Ni', 'Mn'],
            'Co': ['Ni', 'Fe', 'Cu'],
            'K': ['Na', 'Rb', 'Cs'],
            'Na': ['K', 'Li', 'Rb'],
            'Li': ['Na', 'K'],
            'Rb': ['K', 'Cs'],
            'Cs': ['Rb', 'K'],
            'Ca': ['Sr', 'Ba', 'Mg'],
            'Sr': ['Ca', 'Ba'],
            'Ba': ['Sr', 'Ca'],
            'Mg': ['Ca', 'Zn'],
            'Ti': ['Zr', 'Hf', 'V'],
            'Zr': ['Ti', 'Hf'],
            'Hf': ['Zr', 'Ti'],
            'O': ['S', 'Se'],
            'S': ['O', 'Se'],
            'F': ['Cl', 'Br'],
            'Cl': ['F', 'Br'],
        }
        
        suggestions = []
        for el in elements:
            if el in substitution_map:
                for target in substitution_map[el][:2]:  # Max 2 suggestions per element
                    suggestions.append(f"{el}:{target}")
        
        return suggestions[:3]  # Return top 3 suggestions
    except:
        return []


# Page configuration
st.set_page_config(
    page_title="Materials Science RAG",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size: 20px !important;
    font-weight: bold;
}
.hazard-high {
    background-color: #ffcccc;
    padding: 10px;
    border-left: 5px solid red;
    margin: 10px 0;
}
.hazard-medium {
    background-color: #fff4cc;
    padding: 10px;
    border-left: 5px solid orange;
    margin: 10px 0;
}
.cif-container {
    background-color: #f0f0f0;
    padding: 15px;
    border-radius: 5px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    white-space: pre;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline(_cache_version="v2"):  # Add version parameter to bust cache
    """Load the single shared pipeline. Cached to avoid reloading."""
    import torch
    
    # Check if we should use OpenRouter or local model
    # Try Streamlit secrets first, then fall back to environment variables
    try:
        use_openrouter = st.secrets.get("USE_OPENROUTER", False)
        if isinstance(use_openrouter, str):
            use_openrouter = use_openrouter.lower() == "true"
    except:
        use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    
    # Show loading status
    with st.spinner("Initializing pipeline... This may take a minute on first run."):
        if use_openrouter:
            # Use OpenRouter API
            st.info("Using OpenRouter API (Qwen2.5-7B-Instruct)")
            
            # Get API key from secrets or env
            try:
                api_key = st.secrets.get("OPENROUTER_API_KEY")
            except:
                api_key = os.getenv("OPENROUTER_API_KEY")
            
            openrouter_agent = OpenRouterAgent(
                api_key=api_key,
                model="qwen/qwen-2.5-7b-instruct"
            )
            
            # Use in-memory Qdrant for Streamlit Cloud (faster, no disk issues)
            pipeline = MaterialsPipeline(
                llama_agent=openrouter_agent,
                qdrant_path=":memory:",  # In-memory mode for cloud
                embedding_model="all-MiniLM-L6-v2",
                use_4bit=False
            )
        else:
            # Use local model (NOT recommended for Streamlit Cloud - 8GB download)
            st.info("Using local model (Qwen2.5-7B-Instruct)")
            st.warning("⚠️ Local model requires 8GB+ download. Use OpenRouter for cloud deployment.")
            use_4bit = torch.cuda.is_available()
            
            pipeline = MaterialsPipeline(
                llama_model_name="Qwen/Qwen2.5-7B-Instruct",
                qdrant_path=":memory:",  # In-memory for cloud compatibility
                embedding_model="all-MiniLM-L6-v2",
                use_4bit=use_4bit
            )
    
    st.success("Pipeline loaded successfully!")
    return pipeline


def display_header():
    """Display app header."""
    st.title("🧱 Materials Science RAG Platform")
    st.markdown("""
    ### CIF Generation • Property Prediction • Safety-Enforced Synthesis
    
    **Powered by:** Qwen2.5-7B • Qdrant • MatGL/AlignFF
    
    ---
    """)


def display_sidebar():
    """Display sidebar with options."""
    st.sidebar.header("⚙️ Configuration")
    
    # Dataset info
    samples = load_sample_materials()
    st.sidebar.metric("📊 Dataset Size", f"{len(samples)} materials", help="Materials available in reaction.csv")
    
    st.sidebar.markdown("---")
    
    # Pipeline options
    st.sidebar.subheader("Pipeline Options")
    
    generate_cif = st.sidebar.checkbox("Generate CIF", value=True)
    predict_properties = st.sidebar.checkbox("Predict Properties", value=True)
    generate_synthesis = st.sidebar.checkbox("Generate Synthesis", value=True)
    scrape_papers = st.sidebar.checkbox("Scrape New Papers (slow)", value=False)
    
    retrieve_top_k = st.sidebar.slider(
        "Papers to Retrieve",
        min_value=1,
        max_value=10,
        value=5
    )
    
    st.sidebar.markdown("---")
    
    # Database management
    st.sidebar.subheader("📚 Vector Database")
    
    # Check for cache file
    import os
    cache_file = 'scraped_papers_cache.json'
    cache_exists = os.path.exists(cache_file)
    
    if cache_exists:
        import json
        try:
            with open(cache_file, 'r') as f:
                cached_papers = json.load(f)
            st.sidebar.info(f"💾 Cache: {len(cached_papers)} papers stored")
        except:
            st.sidebar.warning("⚠ Cache file corrupted")
    else:
        st.sidebar.warning("📦 No cache file - papers need scraping")
    
    st.sidebar.markdown("---")
    
    # Get pipeline from cache if available
    if 'pipeline' in st.session_state:
        pipeline = st.session_state['pipeline']
        try:
            stats = pipeline.embedder.get_collection_stats()
            paper_count = stats.get('points_count', 0)
            
            if paper_count > 0:
                st.sidebar.success(f"✓ {paper_count} papers indexed")
            else:
                st.sidebar.warning("⚠ Database empty")
                if cache_exists:
                    st.sidebar.info("Papers loaded from cache on init")
                else:
                    st.sidebar.info("Enable 'Scrape New Papers' to populate")
        except Exception as e:
            st.sidebar.error(f"Could not check database: {str(e)[:50]}")
    
    if st.sidebar.button("🔄 Populate Database"):
        st.session_state['populate_db'] = True
    
    # Add button to save current papers to cache
    if st.sidebar.button("💾 Save Papers to Cache"):
        if 'pipeline' in st.session_state:
            pipeline = st.session_state['pipeline']
            try:
                # Check if method exists (for backwards compatibility)
                if hasattr(pipeline, 'save_papers_to_cache'):
                    count = pipeline.save_papers_to_cache()
                    if count and count > 0:
                        st.sidebar.success(f"✓ Saved {count} papers to cache")
                        st.sidebar.info("Commit scraped_papers_cache.json to Git!")
                    else:
                        st.sidebar.warning("No papers to save")
                else:
                    st.sidebar.error("Please refresh the page to update the pipeline")
                    if st.sidebar.button("🔄 Clear Cache & Refresh"):
                        st.cache_resource.clear()
                        st.rerun()
            except Exception as e:
                st.sidebar.error(f"Save failed: {str(e)[:50]}")
                st.sidebar.info("Try: Clear cache and refresh")
    
    st.sidebar.markdown("---")
    
    # System info
    st.sidebar.subheader("📊 System Info")
    import torch
    
    if torch.cuda.is_available():
        st.sidebar.success(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        st.sidebar.info(f"4-bit quantization: Enabled")
    else:
        st.sidebar.warning("⚠ No GPU available")
        st.sidebar.info("Using CPU (slower)")
    
    return {
        'generate_cif': generate_cif,
        'predict_properties': predict_properties,
        'generate_synthesis': generate_synthesis,
        'scrape_papers': scrape_papers,
        'retrieve_top_k': retrieve_top_k
    }


def display_input_section():
    """Display input section for material composition."""
    st.header("📝 Input Material")
    
    # Show dataset info
    samples = load_sample_materials()
    st.info(f"📊 Dataset: {len(samples)} materials available in reaction.csv")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Load samples to set default
        default_value = "K2Cu4F10"
        
        composition = st.text_input(
            "Chemical Formula",
            value=default_value,
            help="Enter chemical formula from reaction.csv or custom"
        )
    
    with col2:
        # Load sample materials from reaction.csv
        selected_sample = st.selectbox(
            "Or select from reaction.csv:",
            ["Custom"] + samples,
            help="Materials from reaction.csv dataset"
        )
        
        if selected_sample != "Custom":
            composition = selected_sample
    
    # Substitutions
    st.subheader("🔄 Element Substitutions (Optional)")
    
    # Get suggestions based on composition
    suggestions = get_suggested_substitutions(composition)
    if suggestions:
        st.caption(f"💡 Suggested for {composition}: {', '.join(suggestions)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_sub = st.checkbox("Enable substitutions")
    
    substitutions = None
    if enable_sub:
        with col2:
            st.markdown("Enter substitutions as `old:new` (e.g., Ti:Zr)")
        
        # Pre-fill with first suggestion if available
        default_sub = suggestions[0] if suggestions else ""
        
        sub_text = st.text_input(
            "Substitution",
            value=default_sub,
            help="Format: Ti:Zr to replace Ti with Zr"
        )
        
        if sub_text and ":" in sub_text:
            old, new = sub_text.split(":")
            substitutions = {old.strip(): new.strip()}
    
    return composition, substitutions


def display_results(result: PipelineResult):
    """Display pipeline results."""
    
    # Status
    if result.success:
        st.success("✓ Pipeline completed successfully")
    else:
        st.error("✗ Pipeline failed")
        for error in result.errors:
            st.error(f"Error: {error}")
        return
    
    # Warnings
    if result.warnings:
        with st.expander("⚠️ Warnings", expanded=False):
            for warning in result.warnings:
                st.warning(warning)
    
    # Summary
    st.header("📊 Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original Formula", result.original_formula)
    with col2:
        st.metric("Final Formula", result.final_formula)
    with col3:
        st.metric("Papers Retrieved", len(result.retrieved_papers))
    
    # Tabs for different outputs
    tabs = st.tabs(["🔬 CIF", "📈 Properties", "⚖️ Stoichiometry", "🔄 Precursor Alternatives", "⚗️ Synthesis", "📚 Literature"])
    
    # CIF Tab
    with tabs[0]:
        display_cif_section(result)
    
    # Properties Tab
    with tabs[1]:
        display_properties_section(result)
    
    # Stoichiometry Tab
    with tabs[2]:
        display_stoichiometry_section(result)
    
    # Precursor Alternatives Tab
    with tabs[3]:
        display_precursor_combinations_section(result)
    
    # Synthesis Tab
    with tabs[4]:
        display_synthesis_section(result)
    
    # Literature Tab
    with tabs[5]:
        display_literature_section(result)


def display_cif_section(result: PipelineResult):
    """Display CIF generation results."""
    if not result.cif_content:
        st.warning("No CIF generated")
        return
    
    st.subheader("Generated CIF File")
    
    # Display metadata
    if result.cif_metadata:
        cols = st.columns(2)
        with cols[0]:
            st.metric("Formula", result.cif_metadata.get('formula', 'N/A'))
        with cols[1]:
            st.metric("Reference", result.cif_metadata.get('reference', 'N/A'))
    
    # Display CIF content
    st.markdown(f'<div class="cif-container">{result.cif_content}</div>', unsafe_allow_html=True)
    
    # Download button
    st.download_button(
        label="📥 Download CIF",
        data=result.cif_content,
        file_name=f"{result.final_formula}.cif",
        mime="text/plain"
    )


def generate_element_substitutions(formula: str, max_substitutions: int = 10):
    """Generate chemically reasonable element substitutions."""
    from pymatgen.core import Composition
    import random
    
    substitution_map = {
        'Cu': ['Ni', 'Co', 'Zn', 'Fe'],
        'Ni': ['Cu', 'Co', 'Fe', 'Zn'],
        'Fe': ['Co', 'Ni', 'Mn', 'Cr'],
        'Co': ['Ni', 'Fe', 'Cu', 'Mn'],
        'K': ['Na', 'Rb', 'Cs', 'Li'],
        'Na': ['K', 'Li', 'Rb'],
        'Li': ['Na', 'K'],
        'Rb': ['K', 'Cs', 'Na'],
        'Cs': ['Rb', 'K'],
        'Ca': ['Sr', 'Ba', 'Mg'],
        'Sr': ['Ca', 'Ba'],
        'Ba': ['Sr', 'Ca'],
        'Mg': ['Ca', 'Zn', 'Ni'],
        'Ti': ['Zr', 'Hf', 'V'],
        'Zr': ['Ti', 'Hf'],
        'O': ['S', 'Se'],
        'F': ['Cl', 'Br', 'I'],
        'Cl': ['Br', 'F', 'I'],
    }
    
    comp = Composition(formula)
    elements = [str(el) for el in comp.elements]
    
    # Generate substitutions
    substituted_formulas = []
    
    for elem in elements:
        if elem in substitution_map:
            for substitute in substitution_map[elem][:3]:  # Max 3 per element
                # Create new formula with substitution
                new_formula = formula.replace(elem, substitute)
                try:
                    # Validate it's a valid composition
                    Composition(new_formula)
                    substituted_formulas.append({
                        'formula': new_formula,
                        'substitution': f"{elem} → {substitute}",
                        'original_element': elem,
                        'new_element': substitute
                    })
                except:
                    pass
    
    # Limit to max_substitutions
    if len(substituted_formulas) > max_substitutions:
        substituted_formulas = random.sample(substituted_formulas, max_substitutions)
    
    return substituted_formulas


def generate_precursor_combinations(formula: str, max_combinations: int = 20):
    """Generate all feasible precursor combinations for a given formula."""
    from ingestion.parse_reactions import parse_chemical_formula
    from ingestion.precursor_extraction import get_precursor_alternatives
    from pymatgen.core import Composition
    from itertools import product
    import random
    
    comp_dict = parse_chemical_formula(formula)
    
    # Get all alternative precursors for each element
    element_alternatives = {}
    for element in comp_dict.keys():
        alternatives = get_precursor_alternatives(element)
        element_alternatives[element] = alternatives
    
    # Generate all combinations (limit to prevent explosion)
    elements = list(comp_dict.keys())
    all_alternatives = [element_alternatives[elem] for elem in elements]
    
    # Calculate total combinations
    total_combinations = 1
    for alts in all_alternatives:
        total_combinations *= len(alts)
    
    # If too many, sample randomly
    if total_combinations > max_combinations:
        combinations = []
        for _ in range(max_combinations):
            combo = [random.choice(alts) for alts in all_alternatives]
            combo_dict = dict(zip(elements, combo))
            if combo_dict not in combinations:
                combinations.append(combo_dict)
    else:
        combinations = []
        for combo in product(*all_alternatives):
            combo_dict = dict(zip(elements, combo))
            combinations.append(combo_dict)
    
    return combinations, element_alternatives


def evaluate_precursor_feasibility(precursor_combo: Dict[str, str], formula: str):
    """Evaluate feasibility of a precursor combination."""
    from pymatgen.core import Composition
    from prediction.alignff_predict import AlignFFPredictor
    
    # Initialize predictor (uses fallback if ALIGNN not available)
    predictor = AlignFFPredictor()
    
    # Calculate formation energy estimate
    from ingestion.parse_reactions import parse_chemical_formula
    comp_dict = parse_chemical_formula(formula)
    predictions = predictor.predict(formula, comp_dict, None)
    
    formation_energy = predictions.get('formation_energy_eV_atom', 0.0)
    
    # Calculate precursor cost estimate (relative)
    cost_map = {
        'CO3': 1.0,  # Carbonates (cheap)
        'O': 1.2,    # Oxides (moderate)
        'OH': 1.5,   # Hydroxides (moderate-expensive)
        'NO3': 2.0,  # Nitrates (expensive)
        'Cl': 1.8,   # Chlorides (moderate-expensive)
    }
    
    total_cost = 0
    for precursor in precursor_combo.values():
        # Estimate cost based on anion
        cost = 1.5  # default
        for anion, anion_cost in cost_map.items():
            if anion in precursor:
                cost = anion_cost
                break
        total_cost += cost
    
    avg_cost = total_cost / len(precursor_combo)
    
    # Feasibility score (lower formation energy + lower cost = better)
    # Normalize formation energy (typically -5 to 0 eV/atom)
    normalized_energy = (formation_energy + 5) / 5  # 0 to 1 scale
    normalized_cost = (avg_cost - 1) / 1  # 0 to 1 scale
    
    # Weighted score (60% stability, 40% cost)
    feasibility_score = (1 - normalized_energy) * 0.6 + (1 - normalized_cost) * 0.4
    feasibility_score = max(0, min(1, feasibility_score))  # Clamp to 0-1
    
    # Determine if feasible (formation energy < -1.5 eV/atom typically stable)
    is_feasible = formation_energy < -1.0
    
    return {
        'formation_energy': formation_energy,
        'avg_cost': avg_cost,
        'feasibility_score': feasibility_score,
        'is_feasible': is_feasible,
        'stability_rating': 'High' if formation_energy < -3.0 else 'Medium' if formation_energy < -2.0 else 'Low'
    }


def display_precursor_combinations_section(result: PipelineResult):
    """Display feasible precursor combinations validated by ALIGNN."""
    st.subheader("🔄 Alternative Precursor Combinations")
    
    # Add tabs for different types of alternatives
    alt_tabs = st.tabs(["🔧 Precursor Swaps", "⚗️ Element Substitutions"])
    
    # Tab 1: Precursor alternatives for same formula
    with alt_tabs[0]:
        st.info("""
        Different precursor combinations that could be used to synthesize 
        the **same** target material, with feasibility assessed based on formation energy predictions.
        """)
        display_precursor_swaps(result)
    
    # Tab 2: Element substitutions (different formulas)
    with alt_tabs[1]:
        st.info("""
        Alternative materials with **element substitutions** (e.g., K→Na, Cu→Ni). 
        These create different compounds with potentially similar properties.
        """)
        display_element_substitutions(result)


def display_element_substitutions(result: PipelineResult):
    """Display alternative formulas with element substitutions."""
    st.markdown("### 🧪 Element Substitution Alternatives")
    
    with st.spinner("Generating element substitutions..."):
        try:
            substitutions = generate_element_substitutions(result.final_formula)
            
            if not substitutions:
                st.warning("No common element substitutions found for this composition.")
                return
            
            st.markdown(f"**Original Formula:** `{result.final_formula}`")
            st.markdown("---")
            
            # Evaluate each substitution
            sub_results = []
            progress_bar = st.progress(0)
            
            for idx, sub in enumerate(substitutions):
                new_formula = sub['formula']
                substitution = sub['substitution']
                
                # Evaluate feasibility
                from ingestion.parse_reactions import parse_chemical_formula
                comp_dict = parse_chemical_formula(new_formula)
                evaluation = evaluate_precursor_feasibility({}, new_formula)
                
                # Get precursors for new formula
                from ingestion.precursor_extraction import infer_precursors
                precursors = infer_precursors(comp_dict)
                
                sub_results.append({
                    'Formula': new_formula,
                    'Substitution': substitution,
                    'Original Element': sub['original_element'],
                    'New Element': sub['new_element'],
                    'Precursors': precursors,
                    'Formation Energy (eV/atom)': f"{evaluation['formation_energy']:.3f}",
                    'Stability': evaluation['stability_rating'],
                    'Feasible': '✅' if evaluation['is_feasible'] else '⚠️',
                    '_raw_energy': evaluation['formation_energy'],
                    '_raw_score': evaluation['feasibility_score']
                })
                progress_bar.progress((idx + 1) / len(substitutions))
            
            progress_bar.empty()
            
            # Sort by feasibility
            sub_results.sort(key=lambda x: x['_raw_score'], reverse=True)
            
            # Display summary
            feasible_count = sum(1 for r in sub_results if r['Feasible'] == '✅')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Substitutions", len(sub_results))
            with col2:
                st.metric("Feasible Alternatives", feasible_count)
            with col3:
                st.metric("Original Stability", result.predicted_properties.get('formation_energy_eV_atom', 'N/A'))
            
            st.markdown("---")
            
            # Display each substitution
            for idx, res in enumerate(sub_results, 1):
                emoji = res['Feasible']
                
                with st.expander(
                    f"{emoji} Alternative #{idx}: {res['Formula']} - {res['Substitution']} - {res['Stability']} Stability",
                    expanded=(idx <= 3)
                ):
                    # Highlight the swap
                    st.markdown(f"### {result.final_formula} → {res['Formula']}")
                    st.markdown(f"**Element Swap:** {res['Original Element']} → **{res['New Element']}**")
                    st.markdown("")
                    
                    # Display precursors for new formula
                    st.markdown("**Required Precursors:**")
                    precursor_display = ' + '.join([f"**{p}**" for p in res['Precursors']])
                    st.markdown(precursor_display)
                    
                    # Display metrics
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric("Formation Energy", res['Formation Energy (eV/atom)'])
                    with metric_cols[1]:
                        st.metric("Stability Rating", res['Stability'])
                    with metric_cols[2]:
                        st.metric("Status", "Feasible" if res['Feasible'] == '✅' else "Questionable")
                    
                    # Comparison with original
                    original_energy = result.predicted_properties.get('formation_energy_eV_atom', 0.0)
                    energy_diff = res['_raw_energy'] - original_energy
                    
                    if abs(energy_diff) < 0.5:
                        st.success(f"✓ Similar stability to original (Δ = {energy_diff:.3f} eV/atom)")
                    elif energy_diff < 0:
                        st.success(f"✓ More stable than original (Δ = {energy_diff:.3f} eV/atom)")
                    else:
                        st.warning(f"⚠ Less stable than original (Δ = {energy_diff:.3f} eV/atom)")
                    
                    # Button to view full synthesis for this alternative
                    if st.button(f"🔬 View Full Synthesis Protocol", key=f"synth_{idx}"):
                        st.info(f"Run the pipeline with '{res['Formula']}' to get full synthesis details!")
        
        except Exception as e:
            st.error(f"Error generating substitutions: {e}")
            import traceback
            st.code(traceback.format_exc())


def display_precursor_swaps(result: PipelineResult):
    """Display precursor swaps for the same formula."""
    st.markdown("### 🔧 Precursor Alternatives for Same Formula")
    
    with st.spinner("Generating precursor combinations..."):
        try:
            combinations, element_alternatives = generate_precursor_combinations(
                result.final_formula,
                max_combinations=20
            )
            
            # Display available alternatives per element
            with st.expander("📋 Available Precursor Options per Element", expanded=False):
                for element, alternatives in element_alternatives.items():
                    st.markdown(f"**{element}**: {', '.join(alternatives)}")
            
            st.markdown("---")
            st.markdown(f"#### Evaluated Combinations ({len(combinations)} total)")
            
            # Evaluate each combination
            results = []
            progress_bar = st.progress(0)
            for idx, combo in enumerate(combinations):
                evaluation = evaluate_precursor_feasibility(combo, result.final_formula)
                # Create display name with bold precursors
                combo_display = ' + '.join([f"**{prec}**" for prec in combo.values()])
                combo_simple = ' + '.join([f"{elem}: {prec}" for elem, prec in combo.items()])
                results.append({
                    'Combination': combo_simple,
                    'CombinationDisplay': combo_display,
                    'Precursors': combo,
                    'Formation Energy (eV/atom)': f"{evaluation['formation_energy']:.3f}",
                    'Stability': evaluation['stability_rating'],
                    'Relative Cost': f"{evaluation['avg_cost']:.2f}",
                    'Feasibility Score': f"{evaluation['feasibility_score']:.3f}",
                    'Feasible': '✅' if evaluation['is_feasible'] else '⚠️',
                    '_raw_score': evaluation['feasibility_score'],
                    '_raw_energy': evaluation['formation_energy']
                })
                progress_bar.progress((idx + 1) / len(combinations))
            
            progress_bar.empty()
            
            # Sort by feasibility score
            results.sort(key=lambda x: x['_raw_score'], reverse=True)
            
            # Display summary metrics
            feasible_count = sum(1 for r in results if r['Feasible'] == '✅')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Combinations", len(results))
            with col2:
                st.metric("Feasible Combinations", feasible_count)
            with col3:
                feasibility_rate = (feasible_count / len(results)) * 100
                st.metric("Feasibility Rate", f"{feasibility_rate:.1f}%")
            
            st.markdown("---")
            
            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                show_only_feasible = st.checkbox("Show only feasible combinations", value=False)
            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Feasibility Score", "Formation Energy", "Cost"],
                    index=0
                )
            
            # Filter and sort
            display_results = results.copy()
            if show_only_feasible:
                display_results = [r for r in display_results if r['Feasible'] == '✅']
            
            if sort_by == "Formation Energy":
                display_results.sort(key=lambda x: x['_raw_energy'])
            elif sort_by == "Cost":
                display_results.sort(key=lambda x: float(x['Relative Cost']))
            
            # Display results in expandable cards
            st.markdown(f"### Top {len(display_results)} Combinations")
            
            for idx, res in enumerate(display_results, 1):
                # Color code by feasibility
                if res['Feasible'] == '✅':
                    border_color = "green"
                    emoji = "✅"
                else:
                    border_color = "orange"
                    emoji = "⚠️"
                
                # Create title with bold precursor names
                precursor_list = ' + '.join([f"**{prec}**" for prec in res['Precursors'].values()])
                
                with st.expander(
                    f"{emoji} Combination #{idx} - Score: {res['Feasibility Score']} - {res['Stability']} Stability",
                    expanded=(idx <= 3)
                ):
                    # Display precursors with element mapping in bold
                    st.markdown(f"**Precursors:** {precursor_list}")
                    st.markdown("")
                    st.markdown("*Element Mapping:*")
                    for elem, prec in res['Precursors'].items():
                        st.markdown(f"- {elem} → **{prec}**")
                    
                    # Display metrics
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Formation Energy", res['Formation Energy (eV/atom)'])
                    with metric_cols[1]:
                        st.metric("Stability", res['Stability'])
                    with metric_cols[2]:
                        st.metric("Relative Cost", res['Relative Cost'])
                    with metric_cols[3]:
                        st.metric("Feasibility", res['Feasibility Score'])
                    
                    # Calculate and show stoichiometry for this combination
                    if st.button(f"📊 Calculate Stoichiometry", key=f"stoich_{idx}"):
                        precursor_list = list(res['Precursors'].values())
                        try:
                            prec_data, target_mw, target_moles = calculate_detailed_stoichiometry(
                                result.final_formula,
                                target_mass=10.0
                            )
                            st.markdown("**Stoichiometry for 10g batch:**")
                            import pandas as pd
                            df = pd.DataFrame(prec_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        except Exception as e:
                            st.error(f"Could not calculate: {e}")
            
            # Export button
            st.markdown("---")
            import pandas as pd
            export_df = pd.DataFrame([{
                'Rank': i+1,
                'Precursors': r['Combination'],
                'Formation Energy (eV/atom)': r['Formation Energy (eV/atom)'],
                'Stability': r['Stability'],
                'Relative Cost': r['Relative Cost'],
                'Feasibility Score': r['Feasibility Score'],
                'Feasible': r['Feasible']
            } for i, r in enumerate(display_results)])
            
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Combinations (CSV)",
                data=csv_data,
                file_name=f"{result.final_formula}_precursor_combinations.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error generating combinations: {e}")
            import traceback
            st.code(traceback.format_exc())


def calculate_detailed_stoichiometry(formula: str, target_mass: float = 10.0):
    """Calculate detailed stoichiometry with molecular weights."""
    from ingestion.parse_reactions import parse_chemical_formula
    from ingestion.precursor_extraction import infer_precursors
    from pymatgen.core import Composition
    
    comp_dict = parse_chemical_formula(formula)
    precursors = infer_precursors(comp_dict)
    
    # Calculate molecular weight of target compound
    target_comp = Composition(formula)
    target_mw = target_comp.weight
    target_moles = target_mass / target_mw
    
    # Calculate precursor amounts
    precursor_data = []
    
    for precursor in precursors:
        try:
            prec_comp = Composition(precursor)
            prec_mw = prec_comp.weight
            
            # Find the element this precursor contributes
            contributing_element = None
            for elem in prec_comp.elements:
                if str(elem) in comp_dict:
                    contributing_element = str(elem)
                    break
            
            if contributing_element:
                # Calculate moles needed based on target composition
                target_elem_moles = target_moles * comp_dict[contributing_element]
                prec_elem_count = prec_comp[contributing_element]
                
                # Moles of precursor needed
                prec_moles_needed = target_elem_moles / prec_elem_count
                prec_mass_needed = prec_moles_needed * prec_mw
                
                precursor_data.append({
                    'Precursor': precursor,
                    'Element': contributing_element,
                    'Molar Mass (g/mol)': f"{prec_mw:.3f}",
                    'Moles Needed': f"{prec_moles_needed:.6f}",
                    'Mass Needed (g)': f"{prec_mass_needed:.4f}"
                })
        except Exception as e:
            precursor_data.append({
                'Precursor': precursor,
                'Element': 'Error',
                'Molar Mass (g/mol)': 'N/A',
                'Moles Needed': 'N/A',
                'Mass Needed (g)': 'Error'
            })
    
    return precursor_data, target_mw, target_moles


def display_stoichiometry_section(result: PipelineResult):
    """Display detailed stoichiometry calculations."""
    st.subheader("⚖️ Stoichiometric Calculations")
    
    # User input for target mass
    col1, col2 = st.columns([2, 1])
    with col1:
        target_mass = st.number_input(
            "Target product mass (grams)",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            step=1.0,
            help="Enter the desired amount of final product to synthesize"
        )
    with col2:
        st.metric("Target Formula", result.final_formula)
    
    try:
        precursor_data, target_mw, target_moles = calculate_detailed_stoichiometry(
            result.final_formula,
            target_mass
        )
        
        # Display target compound info
        st.markdown("#### 🎯 Target Compound")
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Formula", result.final_formula)
        with info_cols[1]:
            st.metric("Molar Mass", f"{target_mw:.3f} g/mol")
        with info_cols[2]:
            st.metric("Moles", f"{target_moles:.6f} mol")
        
        st.markdown("---")
        
        # Display precursor requirements
        st.markdown("#### 📦 Required Precursors")
        st.markdown(f"*To synthesize **{target_mass:.2f} g** of {result.final_formula}*")
        
        # Create DataFrame for better display
        import pandas as pd
        df = pd.DataFrame(precursor_data)
        
        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Calculate total mass
        try:
            total_precursor_mass = sum(
                float(row['Mass Needed (g)']) 
                for row in precursor_data 
                if row['Mass Needed (g)'] != 'Error'
            )
            
            st.markdown("---")
            st.markdown("#### 📊 Summary")
            summary_cols = st.columns(3)
            
            with summary_cols[0]:
                st.metric("Total Precursor Mass", f"{total_precursor_mass:.4f} g")
            with summary_cols[1]:
                mass_loss_percent = ((total_precursor_mass - target_mass) / total_precursor_mass) * 100
                st.metric("Expected Mass Loss", f"{mass_loss_percent:.1f}%",
                         help="Due to decomposition (e.g., CO2 from carbonates)")
            with summary_cols[2]:
                theoretical_yield = (target_mass / total_precursor_mass) * 100
                st.metric("Theoretical Yield", f"{theoretical_yield:.1f}%")
        except:
            pass
        
        # Preparation instructions
        st.markdown("---")
        st.markdown("#### 📝 Weighing Instructions")
        st.info("""
        **Procedure:**
        1. Pre-dry all precursors at 110°C for 2-4 hours
        2. Cool in desiccator for 30 minutes
        3. Weigh each precursor according to the table above (±0.1 mg accuracy)
        4. Record actual masses for yield calculations
        5. Mix thoroughly according to synthesis procedure
        """)
        
        # Download button
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Stoichiometry Table (CSV)",
            data=csv_data,
            file_name=f"{result.final_formula}_stoichiometry.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error calculating stoichiometry: {e}")
        st.info("Unable to calculate detailed stoichiometry. Check that the formula is valid.")


def display_properties_section(result: PipelineResult):
    """Display predicted properties."""
    if not result.predicted_properties:
        st.warning("No properties predicted")
        return
    
    st.subheader(f"Predicted Properties ({result.property_method})")
    
    # Create DataFrame
    props_df = pd.DataFrame([
        {"Property": k, "Value": v}
        for k, v in result.predicted_properties.items()
    ])
    
    st.dataframe(props_df, use_container_width=True)
    
    # Highlight key properties
    key_props = {}
    if 'band_gap_eV' in result.predicted_properties:
        key_props['Band Gap (eV)'] = result.predicted_properties['band_gap_eV']
    if 'formation_energy_eV_atom' in result.predicted_properties:
        key_props['Formation Energy (eV/atom)'] = result.predicted_properties['formation_energy_eV_atom']
    if 'density_g_cm3' in result.predicted_properties:
        key_props['Density (g/cm³)'] = result.predicted_properties['density_g_cm3']
    
    if key_props:
        st.subheader("Key Properties")
        cols = st.columns(len(key_props))
        for i, (name, value) in enumerate(key_props.items()):
            with cols[i]:
                st.metric(name, f"{value:.3f}" if isinstance(value, float) else value)


def extract_synthesis_steps(protocol_text: str) -> dict:
    """Extract structured synthesis parameters from protocol text."""
    import re
    
    steps = {
        'temperatures': {},
        'heating_rates': [],
        'holding_times': [],
        'atmosphere': [],
        'steps': []
    }
    
    if not protocol_text:
        return steps
    
    lines = protocol_text.split('\n')
    
    # Extract temperatures (look for patterns like "800°C", "900 C", etc.)
    temp_pattern = r'(\d+)[-–]?(\d+)?\s*[°]?[CcKk]'
    for line in lines:
        temps = re.findall(temp_pattern, line)
        line_lower = line.lower()
        
        if temps:
            for temp in temps:
                temp_val = f"{temp[0]}{'-' + temp[1] if temp[1] else ''}°C"
                
                if 'calcin' in line_lower:
                    steps['temperatures']['Calcination'] = temp_val
                elif 'sinter' in line_lower:
                    steps['temperatures']['Sintering'] = temp_val
                elif 'anneal' in line_lower:
                    steps['temperatures']['Annealing'] = temp_val
                elif 'heat' in line_lower or 'temperature' in line_lower:
                    if 'Target Temperature' not in steps['temperatures']:
                        steps['temperatures']['Target Temperature'] = temp_val
    
    # Extract heating/cooling rates (e.g., "5°C/min", "10 K/min")
    rate_pattern = r'(\d+\.?\d*)\s*[°]?[CcKk]/min'
    for line in lines:
        rates = re.findall(rate_pattern, line)
        for rate in rates:
            if rate not in steps['heating_rates']:
                steps['heating_rates'].append(f"{rate}°C/min")
    
    # Extract holding times (e.g., "2 hours", "4h", "6 hrs")
    time_pattern = r'(\d+\.?\d*)\s*(hours?|hrs?|h\b)'
    for line in lines:
        times = re.findall(time_pattern, line, re.IGNORECASE)
        for time in times:
            time_str = f"{time[0]} {time[1]}"
            if time_str not in steps['holding_times']:
                steps['holding_times'].append(time_str)
    
    # Extract atmosphere mentions
    atmosphere_keywords = ['air', 'argon', 'nitrogen', 'vacuum', 'oxygen', 'inert', 'atmosphere']
    for line in lines:
        line_lower = line.lower()
        for keyword in atmosphere_keywords:
            if keyword in line_lower and line.strip():
                steps['atmosphere'].append(line.strip())
                break
    
    # Extract numbered steps
    step_pattern = r'^\s*(\d+[\.\)]|\([a-z]\)|\*|\-)\s+(.+)'
    for line in lines:
        match = re.match(step_pattern, line)
        if match and len(match.group(2).strip()) > 10:  # Meaningful step
            steps['steps'].append(match.group(2).strip())
    
    return steps


def display_synthesis_section(result: PipelineResult):
    """Display synthesis protocol with safety."""
    if not result.synthesis_protocol:
        st.warning("No synthesis protocol generated")
        return
    
    # Display hazards first
    if result.hazards_detected:
        st.subheader("⚠️ Detected Hazards")
        
        for hazard in result.hazards_detected:
            severity = hazard['severity']
            
            if severity == 'high':
                st.markdown(
                    f'<div class="hazard-high">'
                    f'<b>{hazard["element"].upper()} - HIGH SEVERITY</b><br>'
                    f'Type: {hazard["type"]}<br>'
                    f'{hazard["description"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            elif severity == 'medium':
                st.markdown(
                    f'<div class="hazard-medium">'
                    f'<b>{hazard["element"].upper()} - MEDIUM SEVERITY</b><br>'
                    f'Type: {hazard["type"]}<br>'
                    f'{hazard["description"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
    # Create sub-tabs for different views
    synthesis_tabs = st.tabs(["📋 Structured Steps", "📄 Full Protocol"])
    
    # Structured Steps Tab
    with synthesis_tabs[0]:
        st.subheader("🔬 Synthesis Parameters")
        
        # Extract structured information
        extracted = extract_synthesis_steps(result.synthesis_protocol)
        
        # Display temperatures
        if extracted['temperatures']:
            st.markdown("#### 🌡️ Temperatures")
            temp_cols = st.columns(len(extracted['temperatures']))
            for i, (temp_type, temp_val) in enumerate(extracted['temperatures'].items()):
                with temp_cols[i]:
                    st.metric(temp_type, temp_val)
        
        # Display rates and times
        col1, col2 = st.columns(2)
        
        with col1:
            if extracted['heating_rates']:
                st.markdown("#### 📈 Heating/Cooling Rates")
                for rate in extracted['heating_rates']:
                    st.markdown(f"- {rate}")
            else:
                st.markdown("#### 📈 Heating/Cooling Rates")
                st.info("Typical range: 3-5°C/min (recommended)")
        
        with col2:
            if extracted['holding_times']:
                st.markdown("#### ⏱️ Holding Times")
                for time in extracted['holding_times']:
                    st.markdown(f"- {time}")
            else:
                st.markdown("#### ⏱️ Holding Times")
                st.info("Typical range: 4-6 hours (recommended)")
        
        # Display atmosphere
        if extracted['atmosphere']:
            st.markdown("#### 🌬️ Atmosphere")
            for atm in extracted['atmosphere']:
                st.markdown(f"- {atm}")
        
        # Display clean step-by-step
        if extracted['steps']:
            st.markdown("#### 📝 Synthesis Steps")
            for i, step in enumerate(extracted['steps'], 1):
                st.markdown(f"**Step {i}:** {step}")
        else:
            # Fallback: show protocol with better formatting
            st.markdown("#### 📝 Synthesis Steps")
            protocol_lines = [line.strip() for line in result.synthesis_protocol.split('\n') if line.strip()]
            for i, line in enumerate(protocol_lines[:20], 1):  # Show first 20 meaningful lines
                if len(line) > 10:
                    st.markdown(f"{i}. {line}")
    
    # Full Protocol Tab
    with synthesis_tabs[1]:
        st.subheader("Complete Synthesis Protocol")
        
        # Use text area for better formatting
        st.text_area(
            "Protocol",
            value=result.synthesis_protocol,
            height=600,
            label_visibility="collapsed"
        )
    
    # Download button (outside tabs)
    st.download_button(
        label="📥 Download Protocol",
        data=result.synthesis_protocol,
        file_name=f"{result.final_formula}_synthesis.txt",
        mime="text/plain"
    )


def display_literature_section(result: PipelineResult):
    """Display retrieved literature."""
    if not result.retrieved_papers or len(result.retrieved_papers) == 0:
        st.warning("No literature retrieved")
        st.info("Consider enabling 'Scrape New Papers' to populate the database")
        return
    
    st.subheader(f"Retrieved Papers ({len(result.retrieved_papers)})")
    
    for i, paper in enumerate(result.retrieved_papers, 1):
        with st.expander(f"[{i}] {paper.get('title', 'Unknown')} (Score: {paper.get('score', 0):.3f})"):
            
            # Metadata
            cols = st.columns(3)
            with cols[0]:
                if paper.get('doi'):
                    st.markdown(f"**DOI:** {paper['doi']}")
            with cols[1]:
                if paper.get('pmid'):
                    st.markdown(f"**PMID:** {paper['pmid']}")
            with cols[2]:
                if paper.get('year'):
                    st.markdown(f"**Year:** {paper['year']}")
            
            # Abstract
            st.markdown("**Abstract:**")
            st.write(paper.get('abstract', 'No abstract available'))
            
            # URL
            if paper.get('url'):
                st.markdown(f"[View Paper]({paper['url']})")


def display_explorer_page(pipeline):
    """Interactive material variation explorer."""
    st.header("🎨 Material Variation Explorer")
    
    # Show dataset info
    samples = load_sample_materials()
    st.info(f"📊 Dataset: {len(samples)} materials available in reaction.csv")
    
    st.markdown("""
    Generate and compare material variations with AlignFF-relaxed structures and predicted properties.
    All structures are automatically relaxed before property prediction.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        base_material = st.text_input(
            "Base Material Formula",
            value="K2Cu4F10",
            help="Enter a chemical formula to explore variations"
        )
    
    with col2:
        st.metric("Relaxation Method", "AlignFF", help="All structures are automatically relaxed")
    
    # Get suggested substitutions
    suggestions = get_suggested_substitutions(base_material)
    if suggestions:
        st.info(f"💡 Common substitutions for {base_material}: {', '.join(suggestions)}")
    
    st.subheader("Select Element Substitutions")
    
    # Determine which checkboxes to pre-select based on composition
    from pymatgen.core import Composition
    try:
        comp = Composition(base_material)
        elements_in_material = [str(el) for el in comp.elements]
        
        # Pre-select relevant substitutions
        default_cu_ni = 'Cu' in elements_in_material
        default_cu_ag = 'Cu' in elements_in_material
        default_cu_zn = 'Cu' in elements_in_material
        default_k_na = 'K' in elements_in_material
        default_k_li = 'K' in elements_in_material
        default_f_cl = 'F' in elements_in_material
    except:
        # Fallback defaults for K2Cu4F10
        default_cu_ni = True
        default_cu_ag = True
        default_cu_zn = True
        default_k_na = False
        default_k_li = False
        default_f_cl = False
    
    # Predefined substitution options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Transition Metals**")
        sub_cu_ni = st.checkbox("Cu → Ni", value=default_cu_ni)
        sub_cu_ag = st.checkbox("Cu → Ag", value=default_cu_ag)
        sub_cu_zn = st.checkbox("Cu → Zn", value=default_cu_zn)
        sub_cu_co = st.checkbox("Cu → Co")
        sub_fe_co = st.checkbox("Fe → Co")
        sub_fe_ni = st.checkbox("Fe → Ni")
    
    with col2:
        st.markdown("**Alkali/Alkaline Earth**")
        sub_k_na = st.checkbox("K → Na", value=default_k_na)
        sub_k_li = st.checkbox("K → Li", value=default_k_li)
        sub_k_rb = st.checkbox("K → Rb")
        sub_na_li = st.checkbox("Na → Li")
        sub_ca_sr = st.checkbox("Ca → Sr")
        sub_ba_sr = st.checkbox("Ba → Sr")
    
    with col3:
        st.markdown("**Other Elements**")
        sub_ti_zr = st.checkbox("Ti → Zr")
        sub_ti_hf = st.checkbox("Ti → Hf")
        sub_o_s = st.checkbox("O → S")
        sub_o_se = st.checkbox("O → Se")
        sub_f_cl = st.checkbox("F → Cl", value=default_f_cl)
    
    # Collect selected substitutions
    substitutions_map = {
        'Cu → Ni': {'Cu': 'Ni'} if sub_cu_ni else None,
        'Cu → Ag': {'Cu': 'Ag'} if sub_cu_ag else None,
        'Cu → Zn': {'Cu': 'Zn'} if sub_cu_zn else None,
        'Cu → Co': {'Cu': 'Co'} if sub_cu_co else None,
        'Fe → Co': {'Fe': 'Co'} if sub_fe_co else None,
        'Fe → Ni': {'Fe': 'Ni'} if sub_fe_ni else None,
        'K → Na': {'K': 'Na'} if sub_k_na else None,
        'K → Li': {'K': 'Li'} if sub_k_li else None,
        'K → Rb': {'K': 'Rb'} if sub_k_rb else None,
        'Na → Li': {'Na': 'Li'} if sub_na_li else None,
        'Ca → Sr': {'Ca': 'Sr'} if sub_ca_sr else None,
        'Ba → Sr': {'Ba': 'Sr'} if sub_ba_sr else None,
        'Ti → Zr': {'Ti': 'Zr'} if sub_ti_zr else None,
        'Ti → Hf': {'Ti': 'Hf'} if sub_ti_hf else None,
        'O → S': {'O': 'S'} if sub_o_s else None,
        'O → Se': {'O': 'Se'} if sub_o_se else None,
        'F → Cl': {'F': 'Cl'} if sub_f_cl else None,
    }
    
    selected_subs = {k: v for k, v in substitutions_map.items() if v is not None}
    
    st.markdown("---")
    
    if st.button("🚀 Explore Variations", type="primary"):
        
        if not base_material:
            st.error("Please enter a base material formula")
            st.stop()
        
        if len(selected_subs) == 0:
            st.error("Please select at least one substitution")
            st.stop()
        
        # Processing with detailed progress
        st.markdown("---")
        st.subheader("⚙️ Exploring Material Variations")
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_materials = len(selected_subs) + 1
        
        # Process base material
        status_text.text(f"[1/{total_materials}] Processing base material: {base_material}")
        try:
            result = pipeline.run_materials_pipeline(
                composition=base_material,
                substitutions=None,
                generate_cif=True,
                predict_properties=True,
                generate_synthesis=False,
                retrieve_top_k=0
            )
            
            if result.success and result.predicted_properties:
                results.append({
                    'Formula': base_material,
                    'Variation': 'Original',
                    'Formation Energy (eV/atom)': result.predicted_properties.get('formation_energy_eV_atom', float('nan')),
                    'Band Gap (eV)': result.predicted_properties.get('band_gap_eV', float('nan')),
                    'Density (g/cm³)': result.predicted_properties.get('density_g_cm3', float('nan')),
                    'Status': '✓'
                })
        except Exception as e:
            st.warning(f"Base material failed: {str(e)[:100]}")
        
        progress_bar.progress(1 / total_materials)
        
        # Process variations
        for i, (sub_name, sub_dict) in enumerate(selected_subs.items(), 1):
            status_text.text(f"[{i+1}/{total_materials}] Processing variation: {sub_name}")
            
            try:
                result = pipeline.run_materials_pipeline(
                    composition=base_material,
                    substitutions=sub_dict,
                    generate_cif=True,
                    predict_properties=True,
                    generate_synthesis=False,
                    retrieve_top_k=0
                )
                
                if result.success and result.predicted_properties:
                    results.append({
                        'Formula': result.final_formula,
                        'Variation': sub_name,
                        'Formation Energy (eV/atom)': result.predicted_properties.get('formation_energy_eV_atom', float('nan')),
                        'Band Gap (eV)': result.predicted_properties.get('band_gap_eV', float('nan')),
                        'Density (g/cm³)': result.predicted_properties.get('density_g_cm3', float('nan')),
                        'Status': '✓'
                    })
            except Exception as e:
                st.warning(f"{sub_name} failed: {str(e)[:100]}")
            
            progress_bar.progress((i + 1) / total_materials)
        
        # Completion
        progress_bar.progress(1.0)
        status_text.text("✓ All variations processed!")
        import time
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        # Display results
        if len(results) > 0:
            st.success(f"✓ Explored {len(results)} material variations successfully!")
            
            st.markdown("---")
            st.subheader("📊 Results Comparison")
            
            df = pd.DataFrame(results)
            
            # Display styled dataframe
            st.dataframe(
                df.style.format({
                    'Formation Energy (eV/atom)': '{:.4f}',
                    'Band Gap (eV)': '{:.3f}',
                    'Density (g/cm³)': '{:.3f}'
                }).background_gradient(subset=['Formation Energy (eV/atom)'], cmap='RdYlGn_r'),
                use_container_width=True,
                height=400
            )
            
            # Key insights
            if len(df) > 1:
                st.markdown("---")
                st.subheader("🔍 Key Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    most_stable_idx = df['Formation Energy (eV/atom)'].idxmin()
                    most_stable = df.loc[most_stable_idx]
                    st.metric(
                        "🏆 Most Stable",
                        most_stable['Formula'],
                        f"{most_stable['Formation Energy (eV/atom)']:.4f} eV/atom",
                        help="Lowest formation energy indicates highest thermodynamic stability"
                    )
                
                with col2:
                    largest_gap_idx = df['Band Gap (eV)'].idxmax()
                    largest_gap = df.loc[largest_gap_idx]
                    st.metric(
                        "🔋 Largest Band Gap",
                        largest_gap['Formula'],
                        f"{largest_gap['Band Gap (eV)']:.3f} eV",
                        help="Larger band gap indicates better insulating properties"
                    )
                
                with col3:
                    smallest_gap_idx = df['Band Gap (eV)'].idxmin()
                    smallest_gap = df.loc[smallest_gap_idx]
                    st.metric(
                        "⚡ Smallest Band Gap",
                        smallest_gap['Formula'],
                        f"{smallest_gap['Band Gap (eV)']:.3f} eV",
                        help="Smaller band gap may indicate semiconducting or conducting behavior"
                    )
            
            # Download button
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_data,
                file_name=f"{base_material}_variations.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ No successful results. Please try different substitutions or base material.")


def main():
    """Main Streamlit app."""
    
    # Display header
    display_header()
    
    # Load pipeline (cached)
    with st.spinner("Loading pipeline..."):
        try:
            pipeline = load_pipeline()
            st.session_state['pipeline'] = pipeline  # Store for sidebar access
            st.success("✓ Pipeline loaded")
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            st.stop()
    
    # Handle database population if requested
    if st.session_state.get('populate_db', False):
        st.session_state['populate_db'] = False
        
        st.subheader("📚 Populating Database")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, message):
            progress = (current + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"[{current+1}/{total}] {message}")
        
        try:
            paper_count = pipeline.populate_database_from_reactions(
                force_reload=False,
                progress_callback=update_progress
            )
            progress_bar.empty()
            status_text.empty()
            if paper_count > 0:
                st.success(f"✓ Database populated with {paper_count} papers!")
            else:
                st.info("ℹ️ Database already populated or no papers found")
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Failed to populate database: {e}")
    
    # Page selection
    page = st.sidebar.radio(
        "Navigation",
        ["🚀 Single Material", "🎨 Variation Explorer"],
        help="Choose between single material analysis or exploring multiple variations"
    )
    
    st.sidebar.markdown("---")
    
    # Sidebar (for single material options)
    if page == "🚀 Single Material":
        options = display_sidebar()
    
    # Route to appropriate page
    if page == "🎨 Variation Explorer":
        display_explorer_page(pipeline)
    else:
        # Original single material workflow
        # Input section
        composition, substitutions = display_input_section()
        
        # Run button
        if st.button("🚀 Run Pipeline", type="primary"):
            
            # Validate input
            if not composition:
                st.error("Please enter a chemical formula")
                st.stop()
            
            # Run pipeline with progress indicators
            st.markdown("---")
            st.subheader("⚙️ Processing")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps = []
            if substitutions:
                steps.append("Element substitution")
            if options['generate_cif']:
                steps.append("CIF generation")
            if options['predict_properties']:
                steps.append("Property prediction")
            if options['scrape_papers'] or options['retrieve_top_k'] > 0:
                steps.append("Paper retrieval")
            if options['generate_synthesis']:
                steps.append("Synthesis generation")
            
            total_steps = len(steps)
            
            try:
                for i, step in enumerate(steps):
                    progress = i / total_steps
                    progress_bar.progress(progress)
                    status_text.text(f"[{i+1}/{total_steps}] {step}...")
                
                progress_bar.progress(0.5)
                status_text.text("Running pipeline...")
                
                result = pipeline.run_materials_pipeline(
                    composition=composition,
                    substitutions=substitutions,
                    generate_cif=options['generate_cif'],
                    predict_properties=options['predict_properties'],
                    generate_synthesis=options['generate_synthesis'],
                    scrape_papers=options['scrape_papers'],
                    retrieve_top_k=options['retrieve_top_k']
                )
                
                progress_bar.progress(1.0)
                status_text.text("✓ Complete!")
                
                # Store in session state
                st.session_state['last_result'] = result
                
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Pipeline failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
        
        # Display results
        if 'last_result' in st.session_state:
            st.markdown("---")
            display_results(st.session_state['last_result'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>
    Materials Science RAG Platform | 
    Powered by Qwen2.5-7B, Qdrant, and Materials ML Models |
    <b>UI Layer Only - All logic in shared pipeline</b>
    </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
