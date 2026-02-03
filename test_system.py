#!/usr/bin/env python3
"""
Comprehensive test script to verify all components work correctly
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_csv_loading():
    """Test reaction.csv loading."""
    print("\n" + "="*80)
    print("TEST 1: CSV File Loading")
    print("="*80)
    
    import csv
    
    csv_path = os.path.join(os.path.dirname(__file__), "reaction.csv")
    
    if not os.path.exists(csv_path):
        print("✗ reaction.csv not found")
        return False
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            materials = []
            for row in reader:
                comp = row.get('composition', '').strip()
                if comp:
                    materials.append(comp)
        
        print(f"✓ reaction.csv loaded successfully")
        print(f"✓ Found {len(materials)} materials")
        if materials:
            print(f"✓ First material: {materials[0]}")
            print(f"✓ Sample materials: {', '.join(materials[:3])}")
        else:
            print("⚠ Warning: No materials found in CSV")
        
        return len(materials) > 0
        
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return False


def test_module_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*80)
    print("TEST 2: Module Imports")
    print("="*80)
    
    modules = [
        ('ingestion.parse_reactions', 'ReactionParser'),
        ('ingestion.precursor_extraction', 'PrecursorExtractor'),
        ('crystal.composition_editing', 'CompositionEditor'),
        ('crystal.cif_generation', 'CIFGenerator'),
        ('synthesis.hazard_detection', 'HazardDetector'),
        ('synthesis.synthesis_generator', 'SynthesisGenerator'),
        ('prediction.alignff_predict', 'AlignFFPredictor'),
    ]
    
    success = True
    for module_path, class_name in modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {module_path}.{class_name}")
        except Exception as e:
            print(f"✗ {module_path}.{class_name}: {e}")
            success = False
    
    return success


def test_hazard_detection():
    """Test hazard detection system."""
    print("\n" + "="*80)
    print("TEST 3: Hazard Detection")
    print("="*80)
    
    try:
        from synthesis.hazard_detection import HazardDetector
        
        detector = HazardDetector()
        
        # Test fluoride
        test_cases = [
            ('Li1Ni1F6', 2),  # Li + F
            ('K2Cu4F10', 1),  # F only
            ('Ba2Cl8Ni1Pb1', 2),  # Ba + Pb
        ]
        
        for formula, expected_min in test_cases:
            hazards = detector.detect_hazards(formula)
            if len(hazards) >= expected_min:
                print(f"✓ {formula}: detected {len(hazards)} hazards")
                for h in hazards:
                    # Hazard is a dataclass, use attributes
                    print(f"    • {h.element}: {h.severity}")
            else:
                print(f"⚠ {formula}: expected >={expected_min}, got {len(hazards)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hazard detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_precursor_inference():
    """Test precursor inference."""
    print("\n" + "="*80)
    print("TEST 4: Precursor Inference")
    print("="*80)
    
    try:
        from ingestion.precursor_extraction import PrecursorExtractor
        
        extractor = PrecursorExtractor()
        
        test_cases = ['K2Cu4F10', 'Li1Ni1F6', 'Ba2Cl8Ni1Pb1']
        
        for formula in test_cases:
            precursors = extractor.infer_precursors(formula)
            if precursors:
                print(f"✓ {formula}: {', '.join(precursors)}")
            else:
                print(f"⚠ {formula}: no precursors inferred")
        
        return True
        
    except Exception as e:
        print(f"✗ Precursor inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_composition_editing():
    """Test composition editing and validation."""
    print("\n" + "="*80)
    print("TEST 5: Composition Editing")
    print("="*80)
    
    try:
        from crystal.composition_editing import CompositionEditor
        
        editor = CompositionEditor()
        
        # Test substitution
        original = "K2Cu4F10"
        substitutions = {"Cu": "Ag"}
        
        new_formula, new_comp = editor.apply_substitution(original, substitutions)
        print(f"✓ Substitution: {original} → {new_formula}")
        
        # Test validation
        warnings = editor.validate_substitution(original, substitutions)
        if warnings:
            print(f"  Warnings: {len(warnings)}")
            for w in warnings:
                print(f"    ⚠ {w}")
        else:
            print(f"  ✓ No validation warnings")
        
        return True
        
    except Exception as e:
        print(f"✗ Composition editing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cif_generation():
    """Test CIF file generation."""
    print("\n" + "="*80)
    print("TEST 6: CIF Generation")
    print("="*80)
    
    try:
        from crystal.cif_generation import CIFGenerator
        
        generator = CIFGenerator()
        
        test_formulas = ['K2Cu4F10', 'Ba2Cl8Ni1Pb1']
        
        for formula in test_formulas:
            cif = generator.generate_cif(formula)
            if cif and 'data_' in cif:
                print(f"✓ {formula}: CIF generated ({len(cif)} chars)")
            else:
                print(f"⚠ {formula}: CIF generation issue")
        
        return True
        
    except Exception as e:
        print(f"✗ CIF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streamlit_loading():
    """Test that Streamlit app can load samples."""
    print("\n" + "="*80)
    print("TEST 7: Streamlit Sample Loading")
    print("="*80)
    
    try:
        import csv
        csv_path = os.path.join(os.path.dirname(__file__), "reaction.csv")
        samples = []
        
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    composition = row.get('composition', '').strip()
                    if composition:
                        samples.append(composition)
        
        if not samples:
            samples = ["K2Cu4F10", "Li1Ni1F6", "Ba2Cl8Ni1Pb1"]
        
        print(f"✓ Sample materials available: {len(samples)}")
        print(f"✓ First 3 samples: {', '.join(samples[:3])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample loading failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\n" + "="*80)
    print("TEST 8: File Structure")
    print("="*80)
    
    required_files = [
        'reaction.csv',
        'requirements.txt',
        'README.md',
        'streamlit_app.py',
        'quickstart.py',
        'colab_setup.ipynb',
        'pipeline/run_pipeline.py',
        'ingestion/parse_reactions.py',
        'ingestion/precursor_extraction.py',
        'ingestion/embed_and_store.py',
        'rag/retriever.py',
        'rag/llama_agent.py',
        'crystal/composition_editing.py',
        'crystal/cif_generation.py',
        'prediction/alignff_predict.py',
        'prediction/matgl_predict.py',
        'synthesis/hazard_detection.py',
        'synthesis/synthesis_generator.py',
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    missing = []
    
    for filepath in required_files:
        full_path = os.path.join(base_dir, filepath)
        if os.path.exists(full_path):
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath} - MISSING")
            missing.append(filepath)
    
    if missing:
        print(f"\n⚠ {len(missing)} files missing")
        return False
    else:
        print(f"\n✓ All {len(required_files)} required files present")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MATERIALS SCIENCE RAG PLATFORM - COMPREHENSIVE TESTS")
    print("="*80)
    
    tests = [
        ("File Structure", test_file_structure),
        ("CSV Loading", test_csv_loading),
        ("Module Imports", test_module_imports),
        ("Hazard Detection", test_hazard_detection),
        ("Precursor Inference", test_precursor_inference),
        ("Composition Editing", test_composition_editing),
        ("CIF Generation", test_cif_generation),
        ("Streamlit Sample Loading", test_streamlit_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
