"""
Streamlit UI for Materials Science RAG Platform

CRITICAL: This is a UI LAYER ONLY. All logic is in pipeline/run_pipeline.py
This app MUST call the shared pipeline. NO logic duplication allowed.
"""

import streamlit as st
import sys
import os
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.run_pipeline import MaterialsPipeline, PipelineResult
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
def load_pipeline():
    """Load the single shared pipeline. Cached to avoid reloading."""
    import torch
    
    use_8bit = torch.cuda.is_available()
    
    pipeline = MaterialsPipeline(
        llama_model_name="meta-llama/Llama-3.1-8B-Instruct",
        qdrant_path="./qdrant_storage",
        embedding_model="all-MiniLM-L6-v2",
        use_8bit=use_8bit
    )
    
    return pipeline


def display_header():
    """Display app header."""
    st.title("🧱 Materials Science RAG Platform")
    st.markdown("""
    ### CIF Generation • Property Prediction • Safety-Enforced Synthesis
    
    **Powered by:** Llama-3.1 • Qdrant • MatGL/AlignFF
    
    ---
    """)


def display_sidebar():
    """Display sidebar with options."""
    st.sidebar.header("⚙️ Configuration")
    
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
    
    # System info
    st.sidebar.subheader("📊 System Info")
    import torch
    
    if torch.cuda.is_available():
        st.sidebar.success(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.warning("⚠ No GPU available")
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Load samples to set default
        samples = load_sample_materials()
        default_value = samples[0] if samples else "K2Cu4F10"
        
        composition = st.text_input(
            "Chemical Formula",
            value=default_value,
            help="Enter chemical formula from reaction.csv or custom"
        )
    
    with col2:
        # Load sample materials from reaction.csv
        samples = load_sample_materials()
        selected_sample = st.selectbox(
            "Or select from reaction.csv:",
            ["Custom"] + samples,
            help="Materials from reaction.csv dataset"
        )
        
        if selected_sample != "Custom":
            composition = selected_sample
    
    # Substitutions
    st.subheader("🔄 Element Substitutions (Optional)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_sub = st.checkbox("Enable substitutions")
    
    substitutions = None
    if enable_sub:
        with col2:
            st.markdown("Enter substitutions as `old:new` (e.g., Ti:Zr)")
        
        sub_text = st.text_input(
            "Substitution",
            value="",
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
    tabs = st.tabs(["🔬 CIF", "📈 Properties", "⚗️ Synthesis", "📚 Literature"])
    
    # CIF Tab
    with tabs[0]:
        display_cif_section(result)
    
    # Properties Tab
    with tabs[1]:
        display_properties_section(result)
    
    # Synthesis Tab
    with tabs[2]:
        display_synthesis_section(result)
    
    # Literature Tab
    with tabs[3]:
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
    
    # Display full protocol
    st.subheader("Complete Synthesis Protocol")
    
    # Use text area for better formatting
    st.text_area(
        "Protocol",
        value=result.synthesis_protocol,
        height=600,
        label_visibility="collapsed"
    )
    
    # Download button
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


def main():
    """Main Streamlit app."""
    
    # Display header
    display_header()
    
    # Load pipeline (cached)
    with st.spinner("Loading pipeline..."):
        try:
            pipeline = load_pipeline()
            st.success("✓ Pipeline loaded")
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            st.stop()
    
    # Sidebar
    options = display_sidebar()
    
    # Input section
    composition, substitutions = display_input_section()
    
    # Run button
    if st.button("🚀 Run Pipeline", type="primary"):
        
        # Validate input
        if not composition:
            st.error("Please enter a chemical formula")
            st.stop()
        
        # Run pipeline (THE ONLY SOURCE OF LOGIC)
        with st.spinner("Running materials discovery pipeline..."):
            try:
                result = pipeline.run_materials_pipeline(
                    composition=composition,
                    substitutions=substitutions,
                    generate_cif=options['generate_cif'],
                    predict_properties=options['predict_properties'],
                    generate_synthesis=options['generate_synthesis'],
                    scrape_papers=options['scrape_papers'],
                    retrieve_top_k=options['retrieve_top_k']
                )
                
                # Store in session state
                st.session_state['last_result'] = result
                
            except Exception as e:
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
    Powered by Llama-3.1, Qdrant, and Materials ML Models |
    <b>UI Layer Only - All logic in shared pipeline</b>
    </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
