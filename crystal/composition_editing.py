"""Composition editing and element substitution."""

from typing import Dict, List, Tuple
from ingestion.parse_reactions import (
    parse_chemical_formula,
    composition_to_formula,
    substitute_element,
    normalize_composition
)


class CompositionEditor:
    """Edit and manipulate material compositions."""
    
    def __init__(self):
        """Initialize composition editor."""
        # Ionic radii data (Å) for common oxidation states
        self.ionic_radii = {
            'Li': 0.76, 'Na': 1.02, 'K': 1.38,
            'Mg': 0.72, 'Ca': 1.00, 'Sr': 1.18, 'Ba': 1.35,
            'Al': 0.54, 'Ga': 0.62, 'In': 0.80,
            'Ti': 0.61, 'Zr': 0.72, 'Hf': 0.71,
            'V': 0.54, 'Nb': 0.64, 'Ta': 0.64,
            'Cr': 0.52, 'Mo': 0.65, 'W': 0.60,
            'Mn': 0.53, 'Fe': 0.55, 'Co': 0.55, 'Ni': 0.55, 'Cu': 0.57, 'Zn': 0.60,
            'Y': 0.90, 'La': 1.03, 'Ce': 1.01, 'Nd': 0.98, 'Sm': 0.96,
            'Gd': 0.94, 'Dy': 0.91, 'Er': 0.89, 'Yb': 0.87
        }
        
        # Electronegativity (Pauling scale)
        self.electronegativities = {
            'Li': 0.98, 'Na': 0.93, 'K': 0.82,
            'Mg': 1.31, 'Ca': 1.00, 'Sr': 0.95, 'Ba': 0.89,
            'Al': 1.61, 'Ga': 1.81, 'In': 1.78,
            'Ti': 1.54, 'Zr': 1.33, 'Hf': 1.3,
            'V': 1.63, 'Nb': 1.6, 'Ta': 1.5,
            'Cr': 1.66, 'Mo': 2.16, 'W': 2.36,
            'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
            'O': 3.44, 'F': 3.98, 'N': 3.04, 'S': 2.58,
            'Y': 1.22, 'La': 1.10, 'Ce': 1.12, 'Nd': 1.14, 'Sm': 1.17
        }
    
    def apply_substitution(
        self,
        formula: str,
        substitutions: Dict[str, str]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Apply element substitutions to a formula.
        
        Args:
            formula: Original chemical formula
            substitutions: Dict mapping old_element -> new_element
            
        Returns:
            Tuple of (new_formula, new_composition)
        """
        # Parse original
        composition = parse_chemical_formula(formula)
        
        # Apply substitutions
        new_composition = composition.copy()
        for old_elem, new_elem in substitutions.items():
            if old_elem in new_composition:
                new_composition = substitute_element(
                    new_composition,
                    old_elem,
                    new_elem
                )
        
        # Generate new formula
        new_formula = composition_to_formula(new_composition)
        
        return new_formula, new_composition
    
    def suggest_substitutions(
        self,
        element: str,
        criterion: str = "ionic_radius"
    ) -> List[Tuple[str, float]]:
        """
        Suggest alternative elements based on similarity.
        
        Args:
            element: Element to find substitutes for
            criterion: Similarity criterion ('ionic_radius' or 'electronegativity')
            
        Returns:
            List of (element, similarity_score) tuples, sorted by similarity
        """
        if criterion == "ionic_radius":
            data = self.ionic_radii
        elif criterion == "electronegativity":
            data = self.electronegativities
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        if element not in data:
            return []
        
        target_value = data[element]
        
        # Calculate similarity scores
        similarities = []
        for elem, value in data.items():
            if elem != element:
                # Similarity based on relative difference
                diff = abs(value - target_value)
                similarity = 1.0 / (1.0 + diff)
                similarities.append((elem, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:10]  # Top 10
    
    def validate_substitution(
        self,
        original_formula: str,
        substitutions: Dict[str, str]
    ) -> Dict[str, any]:
        """
        Validate a substitution and provide analysis.
        
        Args:
            original_formula: Original material formula
            substitutions: Proposed substitutions
            
        Returns:
            Validation results with warnings and recommendations
        """
        results = {
            'valid': True,
            'warnings': [],
            'ionic_radius_changes': {},
            'electronegativity_changes': {}
        }
        
        for old_elem, new_elem in substitutions.items():
            # Check ionic radius change
            if old_elem in self.ionic_radii and new_elem in self.ionic_radii:
                old_radius = self.ionic_radii[old_elem]
                new_radius = self.ionic_radii[new_elem]
                percent_change = abs(new_radius - old_radius) / old_radius * 100
                
                results['ionic_radius_changes'][f"{old_elem}->{new_elem}"] = {
                    'old': old_radius,
                    'new': new_radius,
                    'percent_change': percent_change
                }
                
                if percent_change > 15:
                    results['warnings'].append(
                        f"Large ionic radius change ({percent_change:.1f}%) "
                        f"for {old_elem}->{new_elem} may affect structure stability"
                    )
            
            # Check electronegativity change
            if old_elem in self.electronegativities and new_elem in self.electronegativities:
                old_en = self.electronegativities[old_elem]
                new_en = self.electronegativities[new_elem]
                diff = abs(new_en - old_en)
                
                results['electronegativity_changes'][f"{old_elem}->{new_elem}"] = {
                    'old': old_en,
                    'new': new_en,
                    'difference': diff
                }
                
                if diff > 0.5:
                    results['warnings'].append(
                        f"Significant electronegativity difference ({diff:.2f}) "
                        f"for {old_elem}->{new_elem} may alter bonding character"
                    )
        
        return results
    
    def partial_substitution(
        self,
        formula: str,
        element: str,
        substitute: str,
        fraction: float
    ) -> Tuple[str, Dict[str, float]]:
        """
        Create a partial substitution (solid solution).
        
        Args:
            formula: Original formula
            element: Element to partially substitute
            substitute: Substituting element
            fraction: Fraction of substitute (0-1)
            
        Returns:
            Tuple of (new_formula, new_composition)
        """
        if not 0 < fraction < 1:
            raise ValueError("Fraction must be between 0 and 1")
        
        composition = parse_chemical_formula(formula)
        
        if element not in composition:
            raise ValueError(f"Element {element} not in formula")
        
        original_amount = composition[element]
        
        # Split the element
        composition[element] = original_amount * (1 - fraction)
        composition[substitute] = original_amount * fraction
        
        # Clean up very small amounts
        composition = {
            elem: round(count, 4)
            for elem, count in composition.items()
            if count > 0.0001
        }
        
        new_formula = composition_to_formula(composition)
        
        return new_formula, composition


if __name__ == "__main__":
    # Test composition editing
    editor = CompositionEditor()
    
    # Test substitution
    original = "BaTiO3"
    subs = {"Ba": "Sr", "Ti": "Zr"}
    new_formula, new_comp = editor.apply_substitution(original, subs)
    print(f"Original: {original}")
    print(f"After substitution: {new_formula}")
    print(f"Composition: {new_comp}\n")
    
    # Test partial substitution
    partial, partial_comp = editor.partial_substitution("BaTiO3", "Ba", "Sr", 0.3)
    print(f"Partial substitution (30% Sr): {partial}")
    print(f"Composition: {partial_comp}\n")
    
    # Test validation
    validation = editor.validate_substitution(original, subs)
    print("Validation results:")
    for warning in validation['warnings']:
        print(f"  - {warning}")
