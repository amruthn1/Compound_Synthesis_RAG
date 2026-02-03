#!/usr/bin/env python3
"""
Test Colab notebook structure and compatibility
"""

import json
import sys
import os

def test_notebook_structure():
    """Test that the notebook is properly structured."""
    print("="*80)
    print("COLAB NOTEBOOK VALIDATION")
    print("="*80)
    
    nb_path = 'colab_setup.ipynb'
    
    if not os.path.exists(nb_path):
        print(f"✗ {nb_path} not found")
        return False
    
    try:
        with open(nb_path, 'r') as f:
            nb = json.load(f)
        
        cells = nb.get('cells', [])
        metadata = nb.get('metadata', {})
        
        print(f"✓ Notebook loaded successfully")
        print(f"  Total cells: {len(cells)}")
        
        # Check cell types
        code_cells = [c for c in cells if c['cell_type'] == 'code']
        markdown_cells = [c for c in cells if c['cell_type'] == 'markdown']
        
        print(f"  Code cells: {len(code_cells)}")
        print(f"  Markdown cells: {len(markdown_cells)}")
        
        # Check for required content
        all_content = ' '.join(''.join(c.get('source', [])) for c in cells)
        
        checks = {
            'MaterialsPipeline import': 'from pipeline.run_pipeline import MaterialsPipeline' in all_content,
            'Ba2Cl8Ni1Pb1 example': 'Ba2Cl8Ni1Pb1' in all_content or 'Ba2CI8Ni1Pb1' in all_content,
            'K2Cu4F10 example': 'K2Cu4F10' in all_content,
            'Li1Ni1F6 example': 'Li1Ni1F6' in all_content,
            'GPU detection': 'torch.cuda.is_available()' in all_content or 'GPU' in all_content,
            'Install dependencies': 'pip install' in all_content,
        }
        
        print("\nContent checks:")
        all_passed = True
        for check_name, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Error loading notebook: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_cells_syntax():
    """Test that code cells have valid Python syntax."""
    print("\n" + "="*80)
    print("CODE CELL SYNTAX VALIDATION")
    print("="*80)
    
    try:
        with open('colab_setup.ipynb', 'r') as f:
            nb = json.load(f)
        
        code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        
        errors = []
        for i, cell in enumerate(code_cells):
            source = ''.join(cell.get('source', []))
            
            # Skip empty cells
            if not source.strip():
                continue
            
            # Skip shell commands
            if source.strip().startswith('!') or source.strip().startswith('%'):
                continue
            
            try:
                compile(source, f'<cell {i}>', 'exec')
            except SyntaxError as e:
                errors.append((i, str(e)))
        
        if errors:
            print(f"✗ Found {len(errors)} syntax errors:")
            for cell_num, error in errors:
                print(f"  Cell {cell_num}: {error}")
            return False
        else:
            print(f"✓ All {len(code_cells)} code cells have valid syntax")
            return True
            
    except Exception as e:
        print(f"✗ Error validating syntax: {e}")
        return False


def test_examples_from_csv():
    """Test that examples use materials from reaction.csv."""
    print("\n" + "="*80)
    print("EXAMPLE MATERIALS VALIDATION")
    print("="*80)
    
    try:
        # Load CSV materials
        import csv
        with open('reaction.csv', 'r') as f:
            reader = csv.DictReader(f)
            csv_materials = set(row['composition'].strip() for row in reader if row.get('composition'))
        
        print(f"✓ Loaded {len(csv_materials)} materials from reaction.csv")
        
        # Load notebook
        with open('colab_setup.ipynb', 'r') as f:
            nb = json.load(f)
        
        all_content = ' '.join(''.join(c.get('source', [])) for c in nb['cells'])
        
        # Check for expected materials (accounting for Cl vs CI variations)
        test_materials = [
            ('Ba2Cl8Ni1Pb1', 'Ba2CI8Ni1Pb1'),  # Both spellings
            ('K2Cu4F10', 'K2Cu4F10'),
            ('Li1Ni1F6', 'Li1Ni1F6'),
        ]
        
        found = []
        not_found = []
        
        for expected, alternate in test_materials:
            if expected in all_content or alternate in all_content:
                found.append(expected)
            else:
                not_found.append(expected)
        
        print(f"\nMaterials in notebook:")
        for material in found:
            in_csv = material in csv_materials or material.replace('Cl', 'CI') in csv_materials
            csv_status = "✓ in CSV" if in_csv else "⚠ not in CSV"
            print(f"  ✓ {material} - {csv_status}")
        
        if not_found:
            print(f"\nMissing materials:")
            for material in not_found:
                print(f"  ✗ {material}")
        
        return len(found) >= 2  # At least 2 examples
        
    except Exception as e:
        print(f"✗ Error checking examples: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cell_execution_order():
    """Test that cells are in logical execution order."""
    print("\n" + "="*80)
    print("CELL EXECUTION ORDER VALIDATION")
    print("="*80)
    
    try:
        with open('colab_setup.ipynb', 'r') as f:
            nb = json.load(f)
        
        code_cells = [(i, ''.join(c.get('source', []))) 
                      for i, c in enumerate(nb['cells']) 
                      if c['cell_type'] == 'code' and c.get('source')]
        
        # Check order
        checks = []
        
        # 1. pip install should come before heavy imports (torch, transformers, etc.)
        pip_cell = next((i for i, src in code_cells if 'pip install' in src), None)
        heavy_import_cell = next((i for i, src in code_cells 
                                 if ('import torch' in src or 'from transformers' in src) 
                                 and 'pip' not in src), None)
        
        if pip_cell is not None and heavy_import_cell is not None:
            if pip_cell < heavy_import_cell:
                checks.append(("pip install before heavy imports", True))
            else:
                checks.append(("pip install before heavy imports", False))
        
        # 2. MaterialsPipeline initialization before usage
        init_cell = next((i for i, src in code_cells if 'MaterialsPipeline(' in src), None)
        usage_cell = next((i for i, src in code_cells if 'run_materials_pipeline' in src), None)
        
        if init_cell is not None and usage_cell is not None:
            if init_cell < usage_cell:
                checks.append(("Pipeline init before usage", True))
            else:
                checks.append(("Pipeline init before usage", False))
        
        # 3. Examples come after setup
        example_cells = [i for i, src in code_cells if 'Example' in src or 'result' in src.lower()]
        
        if example_cells and init_cell is not None:
            if all(ex > init_cell for ex in example_cells):
                checks.append(("Examples after initialization", True))
            else:
                checks.append(("Examples after initialization", False))
        
        print("Execution order checks:")
        all_passed = True
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        if not checks:
            print("  ⚠ No order checks could be performed")
            return True
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Error checking execution order: {e}")
        return False


def test_colab_compatibility():
    """Test for Colab-specific features."""
    print("\n" + "="*80)
    print("GOOGLE COLAB COMPATIBILITY")
    print("="*80)
    
    try:
        with open('colab_setup.ipynb', 'r') as f:
            nb = json.load(f)
        
        all_content = ' '.join(''.join(c.get('source', [])) for c in nb['cells'])
        
        checks = {
            'Uses !pip install (Colab style)': '!pip install' in all_content,
            'Has GPU detection': 'cuda' in all_content.lower() or 'gpu' in all_content.lower(),
            'Has dependency installs': 'torch' in all_content and 'transformers' in all_content,
            'No local file paths': '/Users/' not in all_content and 'C:\\' not in all_content,
        }
        
        print("Colab compatibility checks:")
        all_passed = True
        for check_name, passed in checks.items():
            status = "✓" if passed else "⚠"
            print(f"  {status} {check_name}")
            if not passed and 'No local' in check_name:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Error checking Colab compatibility: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("GOOGLE COLAB NOTEBOOK TEST SUITE")
    print("="*80)
    print()
    
    tests = [
        ("Notebook Structure", test_notebook_structure),
        ("Code Cell Syntax", test_code_cells_syntax),
        ("Example Materials", test_examples_from_csv),
        ("Cell Execution Order", test_cell_execution_order),
        ("Colab Compatibility", test_colab_compatibility),
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
        print("\n🎉 Colab notebook is ready to use!")
        print("\nNext steps:")
        print("  1. Upload colab_setup.ipynb to Google Colab")
        print("  2. Upload all project files to Colab session")
        print("  3. Run all cells in order")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
