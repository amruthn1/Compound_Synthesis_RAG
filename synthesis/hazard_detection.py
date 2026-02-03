"""Hazard detection and safety information for materials synthesis."""

from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class Hazard:
    """Represents a chemical hazard."""
    element: str
    hazard_type: str
    severity: str  # 'high', 'medium', 'low'
    description: str
    precautions: List[str]


class HazardDetector:
    """Detect chemical hazards in materials and precursors."""
    
    def __init__(self):
        """Initialize hazard database."""
        self.hazard_database = self._build_hazard_database()
    
    def _build_hazard_database(self) -> Dict[str, Hazard]:
        """Build comprehensive hazard database."""
        hazards = {}
        
        # Fluorine and fluorides
        hazards['F'] = Hazard(
            element='F',
            hazard_type='acute_toxicity',
            severity='high',
            description=(
                "Fluorine and fluorides are HIGHLY TOXIC. Hydrofluoric acid can form "
                "upon contact with moisture, causing severe burns that penetrate tissue "
                "and decalcify bone. Inhalation can cause pulmonary edema."
            ),
            precautions=[
                "MANDATORY: Keep calcium gluconate gel available for HF exposure",
                "Work in fume hood with excellent ventilation",
                "Wear nitrile gloves (NOT latex), face shield, and full protective clothing",
                "Have emergency eyewash and safety shower immediately accessible",
                "Never work alone with fluorine compounds",
                "Dispose of fluoride waste according to hazardous waste protocols"
            ]
        )
        
        # Lithium
        hazards['Li'] = Hazard(
            element='Li',
            hazard_type='pyrophoric_water_reactive',
            severity='high',
            description=(
                "Lithium metal is pyrophoric and reacts violently with water, "
                "releasing hydrogen gas and heat. Lithium compounds may be irritants."
            ),
            precautions=[
                "Store under mineral oil or inert atmosphere",
                "NO water contact - use dry solvents only",
                "Handle in glove box or under inert atmosphere",
                "Class D fire extinguisher required (NOT water or CO2)",
                "Lithium fires require dry sand or dry powder extinguisher"
            ]
        )
        
        # Barium
        hazards['Ba'] = Hazard(
            element='Ba',
            hazard_type='toxic',
            severity='high',
            description=(
                "Soluble barium compounds are highly toxic, affecting the cardiovascular "
                "and nervous systems. BaCO3 is less soluble but still requires care."
            ),
            precautions=[
                "Avoid ingestion and inhalation",
                "Use fume hood when handling powders",
                "Wear gloves and safety glasses",
                "Do not use barium compounds near food or drink",
                "Wash hands thoroughly after handling"
            ]
        )
        
        # Lead
        hazards['Pb'] = Hazard(
            element='Pb',
            hazard_type='toxic_heavy_metal',
            severity='high',
            description=(
                "Lead is a cumulative neurotoxin affecting the nervous system, kidneys, "
                "and reproductive system. Chronic exposure is particularly dangerous."
            ),
            precautions=[
                "Use fume hood to prevent dust/fume inhalation",
                "Wear gloves, lab coat, and safety glasses",
                "Never eat, drink, or smoke in lab",
                "Implement blood lead monitoring for regular users",
                "Use HEPA filtration when grinding or processing",
                "Dispose as hazardous waste"
            ]
        )
        
        # Chromium (VI)
        hazards['Cr'] = Hazard(
            element='Cr',
            hazard_type='carcinogenic_oxidizer',
            severity='high',
            description=(
                "Hexavalent chromium (Cr(VI)) is a known human carcinogen. "
                "Strong oxidizer that can cause severe burns and allergic reactions."
            ),
            precautions=[
                "Assume Cr(VI) present in chromium compounds",
                "Fume hood mandatory",
                "Use double gloves (nitrile preferred)",
                "Avoid skin contact and inhalation",
                "Regular air monitoring if used frequently",
                "Dispose as hazardous waste"
            ]
        )
        
        # Arsenic
        hazards['As'] = Hazard(
            element='As',
            hazard_type='acute_chronic_toxin',
            severity='high',
            description=(
                "Arsenic and its compounds are highly toxic and carcinogenic, "
                "affecting multiple organ systems."
            ),
            precautions=[
                "Strict fume hood use",
                "Gloves, lab coat, face shield required",
                "No mouth pipetting",
                "Shower and change clothes after use",
                "Medical surveillance for regular users"
            ]
        )
        
        # Cadmium
        hazards['Cd'] = Hazard(
            element='Cd',
            hazard_type='carcinogenic_toxic',
            severity='high',
            description=(
                "Cadmium is a carcinogen that accumulates in kidneys and bones. "
                "Inhalation of fumes is extremely dangerous."
            ),
            precautions=[
                "Never heat cadmium compounds in open air",
                "Fume hood mandatory",
                "Respiratory protection if dust generated",
                "Gloves and safety glasses required",
                "Dispose as hazardous waste"
            ]
        )
        
        # Mercury
        hazards['Hg'] = Hazard(
            element='Hg',
            hazard_type='neurotoxic',
            severity='high',
            description=(
                "Mercury vapor and compounds are neurotoxic, affecting the central "
                "nervous system, kidneys, and respiratory system."
            ),
            precautions=[
                "Work in fume hood at all times",
                "Use spill trays",
                "Have mercury spill kit available",
                "Monitor air levels regularly",
                "Use mercury vapor analyzer if available"
            ]
        )
        
        # Beryllium
        hazards['Be'] = Hazard(
            element='Be',
            hazard_type='carcinogenic_sensitizer',
            severity='high',
            description=(
                "Beryllium causes chronic beryllium disease (berylliosis) and is "
                "carcinogenic. Even brief exposure can cause sensitization."
            ),
            precautions=[
                "SPECIALIZED TRAINING REQUIRED",
                "Dedicated fume hood for beryllium only",
                "HEPA filtration mandatory",
                "Medical surveillance required",
                "Negative pressure containment",
                "Professional cleanup for spills"
            ]
        )
        
        # Nickel
        hazards['Ni'] = Hazard(
            element='Ni',
            hazard_type='carcinogenic_allergenic',
            severity='medium',
            description=(
                "Nickel compounds are carcinogenic (especially Ni subsulfide and oxides). "
                "Common allergen causing contact dermatitis."
            ),
            precautions=[
                "Fume hood when handling powders",
                "Gloves required (use nitrile if nickel-allergic)",
                "Minimize skin contact",
                "Dust control measures"
            ]
        )
        
        # Cobalt
        hazards['Co'] = Hazard(
            element='Co',
            hazard_type='carcinogenic_toxic',
            severity='medium',
            description=(
                "Cobalt compounds may be carcinogenic and can cause respiratory "
                "sensitization and heart effects with chronic exposure."
            ),
            precautions=[
                "Use fume hood",
                "Avoid inhalation of dusts and fumes",
                "Gloves and safety glasses required",
                "Minimize exposure time"
            ]
        )
        
        # Manganese
        hazards['Mn'] = Hazard(
            element='Mn',
            hazard_type='neurotoxic',
            severity='medium',
            description=(
                "Chronic manganese exposure can cause neurological effects similar "
                "to Parkinson's disease (manganism)."
            ),
            precautions=[
                "Control dust and fumes",
                "Use fume hood when heating",
                "Wear appropriate PPE",
                "Avoid chronic inhalation exposure"
            ]
        )
        
        # Strong oxidizers (general)
        for elem in ['V', 'Cr', 'Mn', 'Ce']:
            if elem not in hazards:
                hazards[elem] = Hazard(
                    element=elem,
                    hazard_type='oxidizer',
                    severity='medium',
                    description=(
                        f"{elem} compounds in high oxidation states are strong oxidizers "
                        "that can intensify fires and react violently with reducing agents."
                    ),
                    precautions=[
                        "Keep away from flammable materials",
                        "Store separately from reducing agents",
                        "Use appropriate PPE",
                        "Work in fume hood"
                    ]
                )
        
        return hazards
    
    def detect_hazards(
        self,
        composition,
        precursors: List[str] = None
    ) -> List[Hazard]:
        """
        Detect all hazards in a material and its precursors.
        
        Args:
            composition: Material composition (string formula or dict)
            precursors: List of precursor formulas
            
        Returns:
            List of relevant Hazard objects
        """
        from ingestion.parse_reactions import parse_chemical_formula
        
        # Handle string input
        if isinstance(composition, str):
            composition = parse_chemical_formula(composition)
        
        hazards = []
        elements_checked = set()
        
        # Check material composition
        for element in composition.keys():
            if element in self.hazard_database and element not in elements_checked:
                hazards.append(self.hazard_database[element])
                elements_checked.add(element)
        
        # Check precursors
        if precursors:
            from ingestion.parse_reactions import parse_chemical_formula
            for precursor in precursors:
                try:
                    prec_comp = parse_chemical_formula(precursor)
                    for element in prec_comp.keys():
                        if element in self.hazard_database and element not in elements_checked:
                            hazards.append(self.hazard_database[element])
                            elements_checked.add(element)
                except:
                    pass
        
        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        hazards.sort(key=lambda h: severity_order.get(h.severity, 3))
        
        return hazards
    
    def get_general_lab_safety(self) -> List[str]:
        """Get general laboratory safety requirements."""
        return [
            "Always wear safety glasses or goggles in the laboratory",
            "Wear appropriate lab coat or protective clothing",
            "Use nitrile or appropriate chemical-resistant gloves",
            "Ensure fume hood is functioning properly before use",
            "Know the location of safety equipment (eyewash, shower, fire extinguisher)",
            "Never work alone with hazardous materials",
            "Keep work area clean and organized",
            "Label all containers clearly",
            "Have Material Safety Data Sheets (MSDS/SDS) readily available"
        ]
    
    def get_thermal_safety(
        self,
        max_temperature: float
    ) -> List[str]:
        """
        Get thermal safety requirements.
        
        Args:
            max_temperature: Maximum processing temperature (°C)
            
        Returns:
            List of thermal safety precautions
        """
        precautions = []
        
        if max_temperature > 1000:
            precautions.extend([
                f"High temperature operation ({max_temperature}°C) requires:",
                "- High-temperature furnace with proper insulation",
                "- Heat-resistant gloves and face shield for loading/unloading",
                "- Allow furnace to cool below 100°C before opening",
                "- Use long-handled tongs or crucible holder",
                "- Ensure adequate distance from flammable materials",
                "- Verify furnace safety interlocks are functional"
            ])
        
        if max_temperature > 1500:
            precautions.extend([
                "- Extreme temperature: ensure furnace elements rated for temperature",
                "- Check crucible compatibility at extreme temperatures",
                "- Consider thermal shock when heating/cooling"
            ])
        
        return precautions
    
    def format_safety_section(
        self,
        hazards: List[Hazard],
        max_temperature: float = 1200
    ) -> str:
        """
        Format comprehensive safety section.
        
        Args:
            hazards: List of detected hazards
            max_temperature: Maximum processing temperature
            
        Returns:
            Formatted safety section text
        """
        lines = ["1. SAFETY PROTOCOLS\n"]
        
        # PPE
        lines.append("Personal Protective Equipment (PPE):")
        lines.extend([
            "- Safety glasses or goggles (ANSI Z87.1 rated)",
            "- Chemical-resistant gloves (nitrile recommended)",
            "- Laboratory coat or protective clothing",
            "- Closed-toe shoes"
        ])
        
        # Add specific PPE for high hazards
        for hazard in hazards:
            if hazard.severity == 'high':
                lines.append(f"- Face shield (for {hazard.element} compounds)")
                break
        
        lines.append("")
        
        # Ventilation
        lines.append("Ventilation and Environment:")
        lines.extend([
            "- Perform all operations in a functioning fume hood",
            "- Verify fume hood face velocity (typically 80-120 fpm)",
            "- Ensure high-temperature furnace has proper ventilation",
            "- Never work alone with hazardous materials"
        ])
        lines.append("")
        
        # Chemical hazards
        if hazards:
            lines.append("Chemical Hazards:")
            for hazard in hazards:
                lines.append(f"\n{hazard.element.upper()} HAZARD - {hazard.severity.upper()} SEVERITY:")
                lines.append(f"{hazard.description}")
                lines.append("\nRequired precautions:")
                for precaution in hazard.precautions:
                    lines.append(f"  • {precaution}")
        
        lines.append("")
        
        # Thermal hazards
        thermal_safety = self.get_thermal_safety(max_temperature)
        if thermal_safety:
            lines.append("Thermal and Mechanical Hazards:")
            lines.extend([f"  • {item}" for item in thermal_safety])
            lines.append("")
        
        # Emergency procedures
        lines.append("Emergency Procedures:")
        lines.extend([
            "- Eye contact: Immediately flush with water for 15 minutes, seek medical attention",
            "- Skin contact: Remove contaminated clothing, wash with water for 15 minutes",
            "- Inhalation: Move to fresh air, seek medical attention if symptoms develop",
            "- Ingestion: Do NOT induce vomiting, seek immediate medical attention",
            "- Fire: Use appropriate extinguisher (Class D for metal fires, ABC for others)",
            "- Spill: Contain spill, use appropriate cleanup materials, dispose as hazardous waste"
        ])
        
        # Specific emergency procedures for high hazards
        for hazard in hazards:
            if hazard.element == 'F':
                lines.extend([
                    "\nFLUORIDE EXPOSURE EMERGENCY:",
                    "  • Apply calcium gluconate gel immediately to affected area",
                    "  • Seek medical attention IMMEDIATELY (HF burns can be life-threatening)",
                    "  • Transport patient with MSDS to emergency facility",
                    "  • Notify medical personnel of HF exposure specifically"
                ])
                break
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test hazard detection
    from ingestion.parse_reactions import parse_chemical_formula
    
    detector = HazardDetector()
    
    # Test materials
    materials = [
        ("BaTiO3", ["BaCO3", "TiO2"]),
        ("LiFePO4", ["Li2CO3", "Fe2O3", "NH4H2PO4"]),
        ("LaF3", ["La2O3", "NH4F"])
    ]
    
    for formula, precursors in materials:
        comp = parse_chemical_formula(formula)
        hazards = detector.detect_hazards(comp, precursors)
        
        print(f"\n{'='*80}")
        print(f"Hazards for {formula}:")
        print('='*80)
        
        safety_section = detector.format_safety_section(hazards, 1200)
        print(safety_section)
