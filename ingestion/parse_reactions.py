"""Parse reactions.csv and extract material compositions."""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


def parse_chemical_formula(formula: str) -> Dict[str, float]:
    """
    Parse a chemical formula into element:count dictionary.
    
    Examples:
        TiO2 -> {'Ti': 1.0, 'O': 2.0}
        BaTiO3 -> {'Ba': 1.0, 'Ti': 1.0, 'O': 3.0}
        La0.7Sr0.3MnO3 -> {'La': 0.7, 'Sr': 0.3, 'Mn': 1.0, 'O': 3.0}
    """
    # Pattern to match element symbol followed by optional number (including decimals)
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    
    matches = re.findall(pattern, formula)
    composition = {}
    
    for element, count in matches:
        if element:  # Skip empty matches
            count_val = float(count) if count else 1.0
            composition[element] = composition.get(element, 0.0) + count_val
    
    return composition


def normalize_composition(composition: Dict[str, float]) -> Dict[str, float]:
    """Normalize composition to sum to 1.0."""
    total = sum(composition.values())
    if total == 0:
        return composition
    return {elem: count / total for elem, count in composition.items()}


def load_reactions(csv_path: str):
    """
    Load and parse reactions.csv file.
    
    Expected columns:
        - composition/target (material formula)
        - precursors (optional)
        - reaction (optional)
    
    Returns:
        pandas DataFrame if pandas available, otherwise dict
    """
    if not HAS_PANDAS:
        import csv
        # Fallback without pandas
        reactions = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                composition_key = 'composition' if 'composition' in row else 'target'
                formula = row.get(composition_key, '').strip()
                if formula:
                    reactions.append({
                        'formula': formula,
                        'parsed_composition': parse_chemical_formula(formula),
                        'precursors': row.get('precursors', ''),
                        'reaction': row.get('reaction', '')
                    })
        return reactions
    
    df = pd.read_csv(csv_path)
    
    # Handle different column names
    composition_col = 'composition' if 'composition' in df.columns else 'target'
    
    # Parse target compositions
    df['parsed_composition'] = df[composition_col].apply(parse_chemical_formula)
    df['elements'] = df['parsed_composition'].apply(lambda x: list(x.keys()))
    df['normalized_composition'] = df['parsed_composition'].apply(normalize_composition)
    
    return df


def get_unique_elements(reactions) -> List[str]:
    """Get all unique elements across all materials."""
    if HAS_PANDAS and isinstance(reactions, pd.DataFrame):
        all_elements = []
        for elements in reactions['elements']:
            all_elements.extend(elements)
        return sorted(set(all_elements))
    else:
        # List of dicts
        all_elements = []
        for reaction in reactions:
            all_elements.extend(reaction['parsed_composition'].keys())
        return sorted(set(all_elements))


def composition_to_formula(composition: Dict[str, float], decimals: int = 2) -> str:
    """
    Convert composition dictionary back to formula string.
    
    Args:
        composition: Dictionary of element:count
        decimals: Number of decimal places for non-integer counts
    """
    # Sort elements by electronegativity (simplified: alphabetical for cations, then O, then F)
    def sort_key(item):
        elem = item[0]
        if elem == 'O':
            return (2, elem)
        elif elem == 'F':
            return (3, elem)
        else:
            return (1, elem)
    
    sorted_comp = sorted(composition.items(), key=sort_key)
    
    formula_parts = []
    for elem, count in sorted_comp:
        if abs(count - round(count)) < 0.01:  # Essentially integer
            count_str = str(int(round(count))) if count > 1 else ""
        else:
            count_str = f"{count:.{decimals}f}".rstrip('0').rstrip('.')
            if count_str == "1":
                count_str = ""
        
        formula_parts.append(f"{elem}{count_str}")
    
    return "".join(formula_parts)


def substitute_element(
    composition: Dict[str, float],
    old_element: str,
    new_element: str
) -> Dict[str, float]:
    """
    Substitute one element for another in a composition.
    
    Args:
        composition: Original composition
        old_element: Element to replace
        new_element: Replacement element
        
    Returns:
        New composition with substitution
    """
    if old_element not in composition:
        raise ValueError(f"Element {old_element} not found in composition")
    
    new_comp = composition.copy()
    new_comp[new_element] = new_comp.pop(old_element)
    
    return new_comp


class ReactionParser:
    """Wrapper class for reaction parsing functionality."""
    
    def __init__(self):
        """Initialize reaction parser."""
        pass
    
    def parse_formula(self, formula: str) -> Dict[str, float]:
        """Parse a chemical formula."""
        return parse_chemical_formula(formula)
    
    def load_reactions(self, csv_path: str):
        """Load reactions from CSV file."""
        return load_reactions(csv_path)


if __name__ == "__main__":
    # Test parsing
    test_formulas = [
        "TiO2",
        "BaTiO3",
        "La0.7Sr0.3MnO3",
        "YBa2Cu3O7",
        "LiFePO4"
    ]
    
    for formula in test_formulas:
        comp = parse_chemical_formula(formula)
        print(f"{formula}: {comp}")
        back = composition_to_formula(comp)
        print(f"  -> {back}\n")
