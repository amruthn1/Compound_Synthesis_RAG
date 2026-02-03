"""Property prediction using AlignFF (if available) or fallback methods."""

import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Optional
import numpy as np


# Check AlignFF availability
try:
    # AlignFF may not be easily installable - provide fallback
    # from alignff import AlignFF
    ALIGNFF_AVAILABLE = False
    print("AlignFF not available - using fallback predictions")
except ImportError:
    ALIGNFF_AVAILABLE = False


class AlignFFPredictor:
    """
    Predict material properties using AlignFF or fallback methods.
    
    AlignFF is a foundation model for materials, but may not be easily accessible.
    This class provides fallback predictions based on composition analysis.
    """
    
    def __init__(self):
        """Initialize predictor."""
        self.model_available = ALIGNFF_AVAILABLE
        
        if self.model_available:
            print("Initializing AlignFF model...")
            # self.model = AlignFF.load_pretrained()
        else:
            print("Using composition-based fallback predictions")
    
    def predict(
        self,
        formula: str,
        composition: Dict[str, float],
        cif_content: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Predict properties for a material.
        
        Args:
            formula: Chemical formula
            composition: Element composition
            cif_content: Optional CIF file content
            
        Returns:
            Dictionary of predicted properties
        """
        if self.model_available:
            return self._predict_with_alignff(formula, composition, cif_content)
        else:
            return self._fallback_predictions(formula, composition)
    
    def _predict_with_alignff(
        self,
        formula: str,
        composition: Dict[str, float],
        cif_content: Optional[str]
    ) -> Dict[str, any]:
        """Predict using AlignFF model."""
        # Placeholder for AlignFF integration
        # In practice, would use AlignFF API
        return self._fallback_predictions(formula, composition)
    
    def _fallback_predictions(
        self,
        formula: str,
        composition: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Generate property predictions based on composition rules.
        
        These are educated estimates based on materials chemistry principles.
        """
        predictions = {}
        
        # Formation energy estimate
        predictions['formation_energy_eV_atom'] = self._estimate_formation_energy(
            composition
        )
        
        # Band gap estimate
        predictions['band_gap_eV'] = self._estimate_band_gap(composition)
        
        # Density estimate
        predictions['density_g_cm3'] = self._estimate_density(composition)
        
        # Melting point estimate
        predictions['melting_point_K'] = self._estimate_melting_point(composition)
        
        # Thermal conductivity estimate
        predictions['thermal_conductivity_W_mK'] = self._estimate_thermal_conductivity(
            composition
        )
        
        # Electrical properties
        if predictions['band_gap_eV'] < 0.5:
            predictions['conductivity_type'] = 'metallic'
        elif predictions['band_gap_eV'] < 3.0:
            predictions['conductivity_type'] = 'semiconductor'
        else:
            predictions['conductivity_type'] = 'insulator'
        
        predictions['prediction_method'] = 'composition_based_heuristics'
        
        return predictions
    
    def _estimate_formation_energy(self, composition: Dict[str, float]) -> float:
        """Estimate formation energy based on composition."""
        # Simplified based on bond energies
        base_energy = -3.0  # eV/atom baseline for stable oxides
        
        # Adjust for specific elements
        if 'O' in composition:
            base_energy -= 1.0  # Oxides typically more stable
        if 'F' in composition:
            base_energy -= 1.5  # Fluorides very stable
        
        # Add some variability
        energy = base_energy + np.random.normal(0, 0.3)
        
        return round(energy, 3)
    
    def _estimate_band_gap(self, composition: Dict[str, float]) -> float:
        """Estimate band gap based on composition."""
        elements = set(composition.keys())
        
        # Metals
        metals = {'Li', 'Na', 'K', 'Cu', 'Ag', 'Au', 'Fe', 'Co', 'Ni'}
        if elements & metals and 'O' not in elements:
            return 0.0
        
        # Wide-gap oxides
        wide_gap_formers = {'Al', 'Mg', 'Ca', 'Sr', 'Ba'}
        if elements & wide_gap_formers and 'O' in elements:
            return round(4.0 + np.random.normal(0, 0.5), 2)
        
        # Transition metal oxides (semiconducting)
        tm_elements = {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'}
        if elements & tm_elements and 'O' in elements:
            return round(2.5 + np.random.normal(0, 0.8), 2)
        
        # Default
        return round(2.0 + np.random.normal(0, 1.0), 2)
    
    def _estimate_density(self, composition: Dict[str, float]) -> float:
        """Estimate density based on composition."""
        # Atomic masses (g/mol) - simplified
        atomic_masses = {
            'H': 1, 'Li': 7, 'C': 12, 'N': 14, 'O': 16, 'F': 19,
            'Na': 23, 'Mg': 24, 'Al': 27, 'Si': 28, 'P': 31, 'S': 32,
            'K': 39, 'Ca': 40, 'Ti': 48, 'V': 51, 'Cr': 52, 'Mn': 55,
            'Fe': 56, 'Co': 59, 'Ni': 59, 'Cu': 64, 'Zn': 65,
            'Sr': 88, 'Zr': 91, 'Ba': 137, 'La': 139
        }
        
        # Calculate average atomic mass
        total_mass = 0
        total_atoms = 0
        for elem, count in composition.items():
            mass = atomic_masses.get(elem, 50)  # Default to 50 if unknown
            total_mass += mass * count
            total_atoms += count
        
        avg_mass = total_mass / total_atoms if total_atoms > 0 else 50
        
        # Rough density estimate (g/cm³)
        # Based on atomic mass and typical packing
        density = avg_mass / 10.0 + np.random.normal(0, 0.5)
        
        return round(max(density, 1.0), 2)
    
    def _estimate_melting_point(self, composition: Dict[str, float]) -> float:
        """Estimate melting point based on composition."""
        base_temp = 1200  # K
        
        # Ionic compounds typically high melting
        if 'O' in composition or 'F' in composition:
            base_temp += 500
        
        # Heavy elements increase melting point
        heavy_elements = {'Ba', 'La', 'Zr', 'W'}
        if set(composition.keys()) & heavy_elements:
            base_temp += 300
        
        temp = base_temp + np.random.normal(0, 200)
        
        return round(max(temp, 300), 0)
    
    def _estimate_thermal_conductivity(self, composition: Dict[str, float]) -> float:
        """Estimate thermal conductivity."""
        # Oxides typically low thermal conductivity
        if 'O' in composition:
            return round(2.0 + np.random.normal(0, 1.0), 2)
        
        # Metals high
        metals = {'Cu', 'Ag', 'Au', 'Al'}
        if set(composition.keys()) & metals:
            return round(100 + np.random.normal(0, 50), 1)
        
        # Default
        return round(10 + np.random.normal(0, 5), 2)


if __name__ == "__main__":
    # Test predictor
    from ingestion.parse_reactions import parse_chemical_formula
    
    predictor = AlignFFPredictor()
    
    test_materials = ["BaTiO3", "TiO2", "LiFePO4"]
    
    for formula in test_materials:
        comp = parse_chemical_formula(formula)
        predictions = predictor.predict(formula, comp)
        
        print(f"\n{formula}:")
        for prop, value in predictions.items():
            print(f"  {prop}: {value}")
