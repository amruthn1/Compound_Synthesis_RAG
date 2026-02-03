#!/usr/bin/env python3
"""
Quick start script for Materials Science RAG Platform
Run this to test the installation and see a demo
"""

import sys
import os

def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    
    required = [
        'torch',
        'transformers',
        'sentence_transformers',
        'qdrant_client',
        'pandas',
        'numpy'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print("\nMissing packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed\n")
    return True


def run_quick_demo():
    """Run a quick demonstration of the pipeline."""
    print("="*80)
    print("MATERIALS SCIENCE RAG PLATFORM - QUICK DEMO")
    print("="*80)
    
    # Import pipeline
    from pipeline.run_pipeline import MaterialsPipeline
    
    print("\n[1/3] Initializing pipeline...")
    print("(This will download models on first run - may take a few minutes)")
    
    # Initialize with minimal settings for demo
    import torch
    use_8bit = torch.cuda.is_available()
    
    try:
        pipeline = MaterialsPipeline(
            llama_model_name="meta-llama/Llama-3.1-8B-Instruct",
            use_8bit=use_8bit
        )
        print("✓ Pipeline initialized")
    except Exception as e:
        print(f"\n✗ Failed to initialize pipeline: {e}")
        print("\nNote: You may need to authenticate with Hugging Face:")
        print("  huggingface-cli login")
        return
    
    print("\n[2/3] Running example: K2Cu4F10 synthesis (from reaction.csv)")
    
    try:
        result = pipeline.run_materials_pipeline(
            composition="K2Cu4F10",
            generate_cif=True,
            predict_properties=True,
            generate_synthesis=True,
            scrape_papers=False,  # Skip scraping for quick demo
            retrieve_top_k=3
        )
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Formula: {result.final_formula}")
        print(f"Precursors: {', '.join(result.precursors)}")
        print(f"Papers retrieved: {len(result.retrieved_papers)}")
        print(f"CIF generated: {'Yes' if result.cif_content else 'No'}")
        print(f"Properties predicted: {'Yes' if result.predicted_properties else 'No'}")
        print(f"Synthesis protocol: {'Yes' if result.synthesis_protocol else 'No'}")
        
        if result.hazards_detected:
            print(f"\nHazards detected: {len(result.hazards_detected)}")
            for h in result.hazards_detected:
                print(f"  • {h['element']}: {h['severity'].upper()}")
        
        # Save outputs
        print("\n[3/3] Saving outputs...")
        
        if result.cif_content:
            with open("demo_K2Cu4F10.cif", 'w') as f:
                f.write(result.cif_content)
            print("  ✓ Saved: demo_K2Cu4F10.cif")
        
        if result.synthesis_protocol:
            with open("demo_K2Cu4F10_synthesis.txt", 'w') as f:
                f.write(result.synthesis_protocol)
            print("  ✓ Saved: demo_K2Cu4F10_synthesis.txt")
        
        # Save complete results
        from pipeline.run_pipeline import save_result_to_json
        save_result_to_json(result, "demo_K2Cu4F10_results.json")
        print("  ✓ Saved: demo_K2Cu4F10_results.json")
        
        print("\n" + "="*80)
        print("DEMO COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review the generated files above")
        print("  2. Launch Streamlit UI: streamlit run streamlit_app.py")
        print("  3. Open colab_setup.ipynb in Google Colab for full examples")
        print("  4. Read README.md for detailed documentation")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    print("\n🧱 Materials Science RAG Platform")
    print("Quick Start Script\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run demo
    try:
        run_quick_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
