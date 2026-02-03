"""Extract and infer synthesis precursors for materials."""

from typing import Dict, List, Set, Union
import re


# Common precursor mapping for solid-state synthesis
COMMON_PRECURSORS = {
    # Oxides
    'Ti': ['TiO2', 'Ti2O3'],
    'Ba': ['BaCO3', 'BaO'],
    'Sr': ['SrCO3', 'SrO'],
    'La': ['La2O3'],
    'Y': ['Y2O3'],
    'Zr': ['ZrO2'],
    'Al': ['Al2O3'],
    'Mg': ['MgO', 'MgCO3'],
    'Ca': ['CaO', 'CaCO3'],
    'Mn': ['MnO2', 'Mn2O3', 'MnO'],
    'Fe': ['Fe2O3', 'Fe3O4', 'FeO'],
    'Co': ['Co3O4', 'CoO'],
    'Ni': ['NiO'],
    'Cu': ['CuO', 'Cu2O'],
    'Zn': ['ZnO'],
    'Cr': ['Cr2O3'],
    'V': ['V2O5'],
    'Nb': ['Nb2O5'],
    'Ta': ['Ta2O5'],
    'W': ['WO3'],
    'Mo': ['MoO3'],
    
    # Alkali metals
    'Li': ['Li2CO3', 'LiOH'],
    'Na': ['Na2CO3', 'NaOH'],
    'K': ['K2CO3', 'KOH'],
    
    # Phosphates and others
    'P': ['NH4H2PO4', 'P2O5'],
    'S': ['S', 'SO3'],
    'Si': ['SiO2'],
    'B': ['H3BO3', 'B2O3'],
    
    # Rare earths
    'Ce': ['CeO2'],
    'Nd': ['Nd2O3'],
    'Sm': ['Sm2O3'],
    'Gd': ['Gd2O3'],
    'Dy': ['Dy2O3'],
    'Er': ['Er2O3'],
    'Yb': ['Yb2O3'],
}


def infer_precursors(composition: Dict[str, float]) -> List[str]:
    """
    Infer likely solid-state synthesis precursors for a material.
    
    Args:
        composition: Dictionary of element:count
        
    Returns:
        List of likely precursor compounds
    """
    precursors = []
    
    for element in composition.keys():
        if element in COMMON_PRECURSORS:
            # Take the first (most common) precursor
            precursors.append(COMMON_PRECURSORS[element][0])
        else:
            # Default to oxide for unknown elements
            precursors.append(f"{element}2O3")
    
    return precursors


def get_precursor_alternatives(element: str) -> List[str]:
    """Get all alternative precursors for an element."""
    return COMMON_PRECURSORS.get(element, [f"{element}2O3"])


def extract_precursors_from_text(text: str) -> Set[str]:
    """
    Extract chemical formulas that might be precursors from text.
    
    This is used when parsing papers to identify mentioned precursors.
    """
    # Pattern to match chemical formulas
    # Matches sequences of capital letters followed by optional lowercase and numbers
    pattern = r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b'
    
    matches = re.findall(pattern, text)
    
    # Filter to likely chemical formulas (contain at least one capital letter and one number or multiple capitals)
    precursors = set()
    for match in matches:
        # Must have at least one digit or multiple capital letters
        if re.search(r'\d', match) or len(re.findall(r'[A-Z]', match)) > 1:
            # Exclude common non-chemical words
            if match not in ['PDF', 'XRD', 'SEM', 'TEM', 'EDS', 'XPS', 'FTIR', 'UV', 'IR']:
                precursors.add(match)
    
    return precursors


def calculate_stoichiometry(
    target_composition: Dict[str, float],
    precursors: List[str],
    target_mass: float = 10.0  # grams
) -> Dict[str, float]:
    """
    Calculate stoichiometric amounts of precursors needed.
    
    This is a simplified calculation - real synthesis would need molecular weights.
    
    Args:
        target_composition: Target material composition
        precursors: List of precursor formulas
        target_mass: Desired total mass of product
        
    Returns:
        Dictionary of precursor:mass in grams
    """
    from ingestion.parse_reactions import parse_chemical_formula
    
    # This is a simplified version - would need actual molecular weight calculations
    # For now, return relative ratios
    
    precursor_amounts = {}
    for precursor in precursors:
        # Parse precursor
        prec_comp = parse_chemical_formula(precursor)
        
        # Find matching elements
        for elem in prec_comp.keys():
            if elem in target_composition:
                # Simplified ratio calculation
                precursor_amounts[precursor] = target_composition[elem]
                break
    
    # Normalize to target mass
    total = sum(precursor_amounts.values())
    if total > 0:
        precursor_amounts = {
            prec: (amount / total) * target_mass
            for prec, amount in precursor_amounts.items()
        }
    
    return precursor_amounts


class PrecursorExtractor:
    """Wrapper class for precursor extraction functionality."""
    
    def __init__(self):
        """Initialize precursor extractor."""
        pass
    
    def infer_precursors(self, composition: Union[str, Dict[str, float]]) -> List[str]:
        """
        Infer synthesis precursors.
        
        Args:
            composition: Chemical formula string or composition dict
            
        Returns:
            List of precursor formulas
        """
        from ingestion.parse_reactions import parse_chemical_formula
        
        if isinstance(composition, str):
            composition = parse_chemical_formula(composition)
        
        return list(infer_precursors(composition))


if __name__ == "__main__":
    # Test precursor inference
    from ingestion.parse_reactions import parse_chemical_formula
    
    test_materials = [
        "BaTiO3",
        "La0.7Sr0.3MnO3",
        "LiFePO4"
    ]
    
    for formula in test_materials:
        comp = parse_chemical_formula(formula)
        precursors = infer_precursors(comp)
        print(f"{formula}:")
        print(f"  Composition: {comp}")
        print(f"  Precursors: {precursors}\n")
