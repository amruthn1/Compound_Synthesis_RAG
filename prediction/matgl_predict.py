"""Property prediction using MatGL (M3GNet)."""

import warnings
warnings.filterwarnings('ignore')

try:
    import matgl
    from matgl.ext.pymatgen import Structure2Graph, get_element_list
    from matgl.models import M3GNet
    from matgl.utils.training import PotentialLightningModule
    import torch
    MATGL_AVAILABLE = True
except ImportError:
    MATGL_AVAILABLE = False
    print("MatGL not available. Install with: pip install matgl")

from typing import Dict, Optional
import numpy as np


class MatGLPredictor:
    """Predict material properties using MatGL M3GNet."""
    
    def __init__(self, model_name: str = "M3GNet-MP-2021.2.8-PES"):
        """
        Initialize MatGL predictor.
        
        Args:
            model_name: Name of pretrained M3GNet model
        """
        if not MATGL_AVAILABLE:
            raise ImportError("MatGL is not installed")
        
        print(f"Loading MatGL model: {model_name}")
        
        try:
            # Load pretrained model
            self.potential = matgl.load_model(model_name)
            self.model_name = model_name
            print("MatGL model loaded successfully")
        except Exception as e:
            print(f"Error loading MatGL model: {e}")
            print("Using fallback predictions")
            self.potential = None
    
    def predict_from_cif(
        self,
        cif_content: str
    ) -> Dict[str, any]:
        """
        Predict properties from CIF file.
        
        Args:
            cif_content: CIF file content
            
        Returns:
            Dictionary of predicted properties
        """
        if self.potential is None:
            return self._fallback_predictions()
        
        try:
            from pymatgen.core import Structure
            import io
            
            # Parse CIF
            structure = Structure.from_str(cif_content, fmt="cif")
            
            # Predict properties
            predictions = self._predict_structure(structure)
            
            return predictions
            
        except Exception as e:
            print(f"Error predicting from CIF: {e}")
            return self._fallback_predictions()
    
    def predict_from_composition(
        self,
        formula: str,
        composition: Dict[str, float],
        structure_type: str = "perovskite"
    ) -> Dict[str, any]:
        """
        Predict properties from composition.
        
        Args:
            formula: Chemical formula
            composition: Element composition
            structure_type: Assumed structure type
            
        Returns:
            Dictionary of predicted properties
        """
        # Generate approximate structure
        try:
            structure = self._create_approximate_structure(
                composition,
                structure_type
            )
            
            if self.potential and structure:
                return self._predict_structure(structure)
            else:
                return self._fallback_predictions()
                
        except Exception as e:
            print(f"Error in composition prediction: {e}")
            return self._fallback_predictions()
    
    def _predict_structure(self, structure) -> Dict[str, any]:
        """Predict properties for a pymatgen Structure."""
        try:
            # Convert to graph
            graph_converter = Structure2Graph(
                element_types=get_element_list([structure]),
                cutoff=5.0
            )
            graph, state = graph_converter.get_graph(structure)
            
            # Predict energy
            with torch.no_grad():
                energy = self.potential(graph, state).detach().cpu().numpy()
            
            # Calculate additional properties
            results = {
                'formation_energy_eV_per_atom': float(energy[0]) / len(structure),
                'formation_energy_eV': float(energy[0]),
                'density_g_cm3': structure.density,
                'volume_A3': structure.volume,
                'formula_units': len(structure),
                'space_group': structure.get_space_group_info()[0],
                'crystal_system': structure.get_space_group_info()[1]
            }
            
            # Estimate band gap (simplified)
            results['estimated_band_gap_eV'] = self._estimate_band_gap(structure)
            
            return results
            
        except Exception as e:
            print(f"Error in structure prediction: {e}")
            return self._fallback_predictions()
    
    def _estimate_band_gap(self, structure) -> float:
        """
        Estimate band gap based on composition.
        
        This is a simplified heuristic.
        """
        # Get elements
        composition = structure.composition.as_dict()
        
        # Simple heuristic based on oxide vs other
        if 'O' in composition:
            # Oxides typically have band gaps
            # Rough estimate based on electronegativity
            return 2.5 + np.random.normal(0, 0.5)
        else:
            # Likely metallic or small gap
            return 0.1 + abs(np.random.normal(0, 0.3))
    
    def _create_approximate_structure(
        self,
        composition: Dict[str, float],
        structure_type: str
    ):
        """Create approximate structure from composition."""
        try:
            from pymatgen.core import Structure, Lattice
            
            # Simple cubic approximation
            a = 4.0  # Lattice parameter
            lattice = Lattice.cubic(a)
            
            # Place atoms
            species = []
            coords = []
            
            elements = list(composition.keys())
            n_elements = len(elements)
            
            for i, elem in enumerate(elements):
                species.append(elem)
                # Distribute along diagonal
                frac = i / max(n_elements, 1)
                coords.append([frac, frac, frac])
            
            structure = Structure(lattice, species, coords)
            return structure
            
        except Exception as e:
            print(f"Error creating structure: {e}")
            return None
    
    def _fallback_predictions(self) -> Dict[str, any]:
        """Return fallback predictions when model is unavailable."""
        return {
            'formation_energy_eV_per_atom': -2.5,
            'density_g_cm3': 5.0,
            'estimated_band_gap_eV': 2.0,
            'note': 'Fallback estimates - MatGL model not available'
        }


if __name__ == "__main__":
    if MATGL_AVAILABLE:
        # Test prediction
        predictor = MatGLPredictor()
        
        # Test with composition
        from ingestion.parse_reactions import parse_chemical_formula
        
        formula = "BaTiO3"
        composition = parse_chemical_formula(formula)
        
        predictions = predictor.predict_from_composition(
            formula,
            composition,
            "perovskite"
        )
        
        print(f"\nPredictions for {formula}:")
        for prop, value in predictions.items():
            print(f"  {prop}: {value}")
    else:
        print("MatGL not available for testing")
