"""CIF file generation using Crystal-Text-LLM approach."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CrystalStructure:
    """Represents a crystal structure."""
    formula: str
    space_group: str
    lattice_params: Dict[str, float]  # a, b, c, alpha, beta, gamma
    atoms: List[Dict[str, any]]  # List of {element, x, y, z, occupancy}
    
    
class CIFGenerator:
    """
    Generate CIF files for materials.
    
    This uses a knowledge-based approach inspired by Crystal-Text-LLM,
    leveraging structural databases and prototype structures.
    """
    
    def __init__(self):
        """Initialize CIF generator with prototype structures."""
        # Common crystal structure prototypes
        self.prototypes = self._initialize_prototypes()
    
    def _initialize_prototypes(self) -> Dict[str, CrystalStructure]:
        """Initialize common structure prototypes."""
        prototypes = {}
        
        # Perovskite (ABO3) - Pm-3m
        prototypes['perovskite_cubic'] = CrystalStructure(
            formula="ABO3",
            space_group="Pm-3m",
            lattice_params={
                'a': 4.0, 'b': 4.0, 'c': 4.0,
                'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0
            },
            atoms=[
                {'label': 'A', 'element': 'Ba', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'occupancy': 1.0},
                {'label': 'B', 'element': 'Ti', 'x': 0.5, 'y': 0.5, 'z': 0.5, 'occupancy': 1.0},
                {'label': 'O', 'element': 'O', 'x': 0.5, 'y': 0.5, 'z': 0.0, 'occupancy': 1.0},
            ]
        )
        
        # Rutile (AO2) - P42/mnm
        prototypes['rutile'] = CrystalStructure(
            formula="AO2",
            space_group="P4_2/mnm",
            lattice_params={
                'a': 4.6, 'b': 4.6, 'c': 2.96,
                'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0
            },
            atoms=[
                {'label': 'A', 'element': 'Ti', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'occupancy': 1.0},
                {'label': 'O', 'element': 'O', 'x': 0.305, 'y': 0.305, 'z': 0.0, 'occupancy': 1.0},
            ]
        )
        
        # Rock salt (AB) - Fm-3m
        prototypes['rocksalt'] = CrystalStructure(
            formula="AB",
            space_group="Fm-3m",
            lattice_params={
                'a': 4.2, 'b': 4.2, 'c': 4.2,
                'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0
            },
            atoms=[
                {'label': 'A', 'element': 'Na', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'occupancy': 1.0},
                {'label': 'B', 'element': 'Cl', 'x': 0.5, 'y': 0.5, 'z': 0.5, 'occupancy': 1.0},
            ]
        )
        
        # Spinel (AB2O4) - Fd-3m
        prototypes['spinel'] = CrystalStructure(
            formula="AB2O4",
            space_group="Fd-3m",
            lattice_params={
                'a': 8.4, 'b': 8.4, 'c': 8.4,
                'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0
            },
            atoms=[
                {'label': 'A', 'element': 'Mg', 'x': 0.125, 'y': 0.125, 'z': 0.125, 'occupancy': 1.0},
                {'label': 'B', 'element': 'Al', 'x': 0.5, 'y': 0.5, 'z': 0.5, 'occupancy': 1.0},
                {'label': 'O', 'element': 'O', 'x': 0.263, 'y': 0.263, 'z': 0.263, 'occupancy': 1.0},
            ]
        )
        
        return prototypes
    
    def infer_structure_type(self, composition: Dict[str, float]) -> str:
        """
        Infer likely structure type from composition.
        
        Args:
            composition: Element:count dictionary
            
        Returns:
            Structure type key
        """
        # Count elements
        n_elements = len(composition)
        
        # Check for perovskite (ABO3)
        if n_elements == 3 and composition.get('O', 0) == 3.0:
            return 'perovskite_cubic'
        
        # Check for rutile (AO2)
        if n_elements == 2 and composition.get('O', 0) == 2.0:
            return 'rutile'
        
        # Check for spinel (AB2O4)
        if n_elements == 3 and composition.get('O', 0) == 4.0:
            return 'spinel'
        
        # Check for rock salt (AB)
        if n_elements == 2:
            return 'rocksalt'
        
        # Default to perovskite for oxides
        if 'O' in composition:
            return 'perovskite_cubic'
        
        return 'rocksalt'
    
    def estimate_lattice_parameter(
        self,
        composition: Dict[str, float],
        structure_type: str
    ) -> float:
        """
        Estimate lattice parameter based on ionic radii.
        
        Args:
            composition: Element composition
            structure_type: Structure type
            
        Returns:
            Estimated lattice parameter (Å)
        """
        # Simplified ionic radii (Å)
        ionic_radii = {
            'Li': 0.76, 'Na': 1.02, 'K': 1.38,
            'Mg': 0.72, 'Ca': 1.00, 'Sr': 1.18, 'Ba': 1.35,
            'Al': 0.54, 'Ti': 0.61, 'Zr': 0.72,
            'O': 1.40, 'F': 1.33, 'N': 1.71,
            'Fe': 0.55, 'Mn': 0.53, 'Co': 0.55, 'Ni': 0.55
        }
        
        # Get radii for composition
        radii_sum = 0
        count = 0
        for elem, amount in composition.items():
            if elem in ionic_radii:
                radii_sum += ionic_radii[elem] * amount
                count += amount
        
        avg_radius = radii_sum / count if count > 0 else 2.0
        
        # Structure-specific scaling
        if structure_type == 'perovskite_cubic':
            # a ≈ 2(r_A + r_O) or 2√2(r_B + r_O)
            return avg_radius * 2.5
        elif structure_type == 'rutile':
            return avg_radius * 2.3
        elif structure_type == 'rocksalt':
            return avg_radius * 2.0
        elif structure_type == 'spinel':
            return avg_radius * 4.0
        
        return 4.0  # Default
    
    def generate_cif(
        self,
        formula: str,
        composition: Optional[Dict[str, float]] = None,
        structure_type: Optional[str] = None,
        reference_doi: Optional[str] = None
    ) -> str:
        """
        Generate CIF file for a material.
        
        Args:
            formula: Chemical formula
            composition: Element composition (auto-parsed if None)
            structure_type: Structure type (auto-inferred if None)
            reference_doi: DOI reference for structure
            
        Returns:
            CIF file content as string
        """
        # Parse composition if needed
        if composition is None:
            from ingestion.parse_reactions import parse_chemical_formula
            composition = parse_chemical_formula(formula)
        
        # Infer structure if not provided
        if structure_type is None:
            structure_type = self.infer_structure_type(composition)
        
        if structure_type not in self.prototypes:
            structure_type = 'perovskite_cubic'  # Fallback
        
        prototype = self.prototypes[structure_type]
        
        # Estimate lattice parameter
        a = self.estimate_lattice_parameter(composition, structure_type)
        
        # Use prototype lattice ratios
        proto_a = prototype.lattice_params['a']
        scale = a / proto_a
        
        lattice = {
            'a': prototype.lattice_params['a'] * scale,
            'b': prototype.lattice_params['b'] * scale,
            'c': prototype.lattice_params['c'] * scale,
            'alpha': prototype.lattice_params['alpha'],
            'beta': prototype.lattice_params['beta'],
            'gamma': prototype.lattice_params['gamma']
        }
        
        # Map composition to sites
        atoms = self._map_composition_to_sites(composition, prototype)
        
        # Generate CIF content
        cif = self._format_cif(
            formula=formula,
            space_group=prototype.space_group,
            lattice=lattice,
            atoms=atoms,
            reference_doi=reference_doi
        )
        
        return cif
    
    def _map_composition_to_sites(
        self,
        composition: Dict[str, float],
        prototype: CrystalStructure
    ) -> List[Dict]:
        """Map actual composition to prototype sites."""
        atoms = []
        composition_elements = list(composition.keys())
        
        for atom in prototype.atoms:
            new_atom = atom.copy()
            
            # Map element
            if atom['element'] in composition:
                # Keep prototype element if it matches
                pass
            else:
                # Find suitable element from composition
                label = atom['label']
                if label == 'A' and len(composition_elements) > 0:
                    new_atom['element'] = composition_elements[0]
                elif label == 'B' and len(composition_elements) > 1:
                    new_atom['element'] = composition_elements[1]
                elif label == 'O':
                    new_atom['element'] = 'O'
                else:
                    # Use first available element
                    for elem in composition_elements:
                        if elem not in [a['element'] for a in atoms]:
                            new_atom['element'] = elem
                            break
            
            atoms.append(new_atom)
        
        return atoms
    
    def _format_cif(
        self,
        formula: str,
        space_group: str,
        lattice: Dict[str, float],
        atoms: List[Dict],
        reference_doi: Optional[str] = None
    ) -> str:
        """Format CIF file content."""
        lines = []
        
        # Header
        lines.append("data_" + formula.replace(" ", "_"))
        lines.append("")
        
        # References
        if reference_doi:
            lines.append(f"_publ_section_references '{reference_doi}'")
            lines.append("")
        
        # Chemical formula
        lines.append(f"_chemical_formula_structural '{formula}'")
        lines.append(f"_chemical_formula_sum '{formula}'")
        lines.append("")
        
        # Space group
        lines.append(f"_space_group_name_H-M_alt '{space_group}'")
        lines.append("")
        
        # Cell parameters
        lines.append(f"_cell_length_a {lattice['a']:.6f}")
        lines.append(f"_cell_length_b {lattice['b']:.6f}")
        lines.append(f"_cell_length_c {lattice['c']:.6f}")
        lines.append(f"_cell_angle_alpha {lattice['alpha']:.3f}")
        lines.append(f"_cell_angle_beta {lattice['beta']:.3f}")
        lines.append(f"_cell_angle_gamma {lattice['gamma']:.3f}")
        lines.append("")
        
        # Symmetry
        lines.append("loop_")
        lines.append("_space_group_symop_operation_xyz")
        lines.append("'x, y, z'")
        lines.append("")
        
        # Atomic positions
        lines.append("loop_")
        lines.append("_atom_site_label")
        lines.append("_atom_site_type_symbol")
        lines.append("_atom_site_fract_x")
        lines.append("_atom_site_fract_y")
        lines.append("_atom_site_fract_z")
        lines.append("_atom_site_occupancy")
        
        for i, atom in enumerate(atoms, 1):
            label = f"{atom['element']}{i}"
            lines.append(
                f"{label:6s} {atom['element']:3s} "
                f"{atom['x']:8.5f} {atom['y']:8.5f} {atom['z']:8.5f} "
                f"{atom['occupancy']:6.4f}"
            )
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test CIF generation
    from ingestion.parse_reactions import parse_chemical_formula
    
    generator = CIFGenerator()
    
    # Test materials
    materials = ["BaTiO3", "TiO2", "LiFePO4"]
    
    for formula in materials:
        comp = parse_chemical_formula(formula)
        cif = generator.generate_cif(formula, comp)
        
        print(f"\n{'='*80}")
        print(f"CIF for {formula}:")
        print('='*80)
        print(cif)
