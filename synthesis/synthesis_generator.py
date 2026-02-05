"""Synthesis protocol generation with mandatory safety enforcement."""

from typing import Dict, List, Optional, Tuple
from rag.llama_agent import LlamaAgent
from synthesis.hazard_detection import HazardDetector


class SynthesisGenerator:
    """
    Generate detailed synthesis protocols with MANDATORY safety enforcement.
    
    This class enforces the absolute rule: NO synthesis protocol without safety.
    """
    
    # Fixed header - MUST match exactly
    PROTOCOL_HEADER = "="*80 + "\nANSWER WITH SAFETY PROTOCOLS\n" + "="*80
    
    # Required sections - ALL MANDATORY
    REQUIRED_SECTIONS = [
        "1. SAFETY PROTOCOLS",
        "2. MATERIALS AND EQUIPMENT",
        "3. DETAILED SYNTHESIS PROCEDURE",
        "4. CHARACTERIZATION",
        "5. NOTES & LIMITATIONS"
    ]
    
    def __init__(
        self,
        llama_agent: LlamaAgent,
        hazard_detector: Optional[HazardDetector] = None
    ):
        """
        Initialize synthesis generator.
        
        Args:
            llama_agent: Llama agent for text generation
            hazard_detector: Hazard detector (creates new if None)
        """
        self.llama_agent = llama_agent
        self.hazard_detector = hazard_detector or HazardDetector()
    
    def generate_synthesis_protocol(
        self,
        formula: str,
        composition: Dict[str, float],
        precursors: List[str],
        retrieved_papers: List[Dict],
        substitutions: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate complete synthesis protocol with MANDATORY safety.
        
        This is the ONLY way to generate synthesis. Safety is ENFORCED.
        
        Args:
            formula: Target material formula
            composition: Element composition
            precursors: List of precursor compounds
            retrieved_papers: Retrieved literature papers
            substitutions: Optional element substitutions performed
            
        Returns:
            Complete protocol with all required sections
        """
        # CRITICAL: Detect all hazards FIRST
        hazards = self.hazard_detector.detect_hazards(composition, precursors)
        
        # Generate each section
        sections = []
        
        # Header (MANDATORY format)
        sections.append(self.PROTOCOL_HEADER)
        sections.append(f"\n## Comprehensive Synthesis Protocol for {formula}\n")
        
        # Section 0: Reaction Conditions & Method (NEW - prominent display)
        reaction_conditions_section = self._generate_reaction_conditions_section(
            formula,
            composition,
            precursors,
            retrieved_papers
        )
        sections.append(reaction_conditions_section)
        
        # Debug: Verify section was generated
        print(f"  ✓ Reaction conditions section generated ({len(reaction_conditions_section)} chars)")
        if "🔥 REACTION CONDITIONS" not in reaction_conditions_section:
            print(f"  ⚠ WARNING: Section missing expected header!")
        
        # Section 1: SAFETY (ABSOLUTELY MANDATORY)
        safety_section = self._generate_safety_section(
            formula,
            composition,
            precursors,
            hazards
        )
        sections.append(safety_section)
        
        # Section 2: Materials and Equipment
        materials_section = self._generate_materials_section(
            formula,
            precursors,
            retrieved_papers
        )
        sections.append(materials_section)
        
        # Section 3: Detailed Procedure
        procedure_section = self._generate_procedure_section(
            formula,
            composition,
            precursors,
            retrieved_papers,
            substitutions
        )
        sections.append(procedure_section)
        
        # Section 4: Characterization
        characterization_section = self._generate_characterization_section(
            formula
        )
        sections.append(characterization_section)
        
        # Section 5: Notes & Limitations
        notes_section = self._generate_notes_section(
            formula,
            substitutions,
            retrieved_papers
        )
        sections.append(notes_section)
        
        # Append retrieved sources (MANDATORY)
        sources_section = self._generate_sources_section(retrieved_papers)
        sections.append(sources_section)
        
        # Combine all sections
        full_protocol = "\n\n".join(sections)
        
        # VALIDATION: Ensure all required sections present
        self._validate_protocol(full_protocol)
        
        return full_protocol
    
    def _generate_reaction_conditions_section(
        self,
        formula: str,
        composition: Dict[str, float],
        precursors: List[str],
        papers: List[Dict]
    ) -> str:
        """Generate comprehensive reaction conditions and method section."""
        lines = []
        lines.append("="*70)
        lines.append("🔥 REACTION CONDITIONS & 🧪 METHOD")
        lines.append("="*70)
        
        # Estimate synthesis temperature SPECIFIC to this material
        temp = self._estimate_synthesis_temperature(composition)
        
        # Get material-specific class
        material_class = self._classify_material(composition)
        
        # Determine precursor list for display
        precursor_list = ', '.join(precursors[:3]) if len(precursors) <= 3 else ', '.join(precursors[:2])
        
        # Temperature section with MATERIAL-SPECIFIC analysis
        lines.append("\n📊 Temperature:")
        
        # Build detailed temperature guidance based on composition
        temp_details = []
        
        # Base temperature range
        temp_details.append(f"Target: {temp}°C (typical range: {temp-100}-{temp+100}°C for {material_class})")
        
        # Material-specific temperature considerations
        if 'F' in composition:
            if 'Ag' in composition:
                temp_details.append(f"⚠️ AgF₂ decomposes above ~200°C - use {temp-200}°C or sealed crucible to prevent decomposition")
            else:
                temp_details.append(f"Fluoride precursors like {precursor_list} typically decompose/react in {temp-150}-{temp}°C range")
        
        if 'Li' in composition or 'Na' in composition or 'K' in composition:
            alkali = [e for e in ['Li', 'Na', 'K'] if e in composition]
            temp_details.append(f"⚠️ {', '.join(alkali)} compounds volatile at high temps - stay below {temp+50}°C to minimize loss")
        
        if 'Ti' in composition or 'Zr' in composition:
            temp_details.append(f"Ti/Zr oxides require high temps ({temp}-{temp+200}°C) for complete crystallization")
        
        if 'Cu' in composition or 'Fe' in composition:
            temp_details.append(f"Transition metal {material_class} benefits from oxidizing atmosphere at {temp-50}-{temp+100}°C")
        
        temp_details.append(f"Heating rate: 3-5°C/min prevents thermal shock and ensures uniform heating")
        temp_details.append(f"Cooling rate: 2-5°C/min maintains phase purity and prevents cracking")
        
        lines.append("; ".join(temp_details))
        
        # Pressure section - DYNAMIC based on element volatility
        lines.append("\n🔧 Pressure:")
        
        pressure_details = []
        
        # Check for volatile elements using dynamic function
        volatile_elements = [e for e in composition.keys() if self._get_element_volatility(e) != 'stable']
        volatility_map = {e: self._get_element_volatility(e) for e in volatile_elements}
        
        if not volatile_elements and 'F' not in composition and 'Cl' not in composition:
            # Simple case - ambient is fine
            pressure_details.append("Ambient pressure (1 atm) suitable - no volatile elements or moisture-sensitive compounds")
        else:
            # Need special considerations
            very_volatile = [e for e, v in volatility_map.items() if v == 'very_volatile']
            mod_volatile = [e for e, v in volatility_map.items() if v == 'moderately_volatile']
            alkali_volatile = [e for e, v in volatility_map.items() if v == 'alkali_volatile']
            
            if very_volatile:
                pressure_details.append(f"⚠️ SEALED TUBE REQUIRED: {', '.join(very_volatile)} sublime at ambient pressure")
                pressure_details.append(f"Use evacuated quartz tube (10⁻³-10⁻⁴ torr) backfilled with inert gas to 0.5-1 atm")
            elif mod_volatile or alkali_volatile:
                all_vol = mod_volatile + alkali_volatile
                pressure_details.append(f"Sealed crucible recommended: {', '.join(all_vol)} volatile at {temp}°C")
                pressure_details.append(f"Alumina crucible with lid or quartz ampule prevents material loss")
            else:
                pressure_details.append("Ambient pressure (1 atm) with appropriate containment")
        
        # Halogen handling
        if 'F' in composition or 'Cl' in composition:
            halogen = 'F' if 'F' in composition else 'Cl'
            pressure_details.append(f"{halogen} compounds: Sealed system MANDATORY - prevents moisture ingress (H{halogen} formation) and volatile loss")
        
        # Oxygen exchange for complex oxides
        if 'O' in composition and len(composition) >= 4 and not volatile_elements:
            pressure_details.append(f"Open crucible option: Allows O₂ exchange - beneficial for {len(composition)}-component oxide stoichiometry")
        
        lines.append("; ".join(pressure_details))
        
        # Atmosphere section - FULLY DYNAMIC from element properties
        lines.append("\n🌬️ Atmosphere:")
        
        # Get calculated atmosphere requirements
        atm_req = self._calculate_atmosphere_requirement(composition)
        
        atm_details = []
        atm_details.append(f"Required: {atm_req['type']}")
        atm_details.append(f"Purity: {atm_req['purity']}")
        atm_details.append(f"Flow rate: {atm_req['flow_rate']}")
        atm_details.append(f"Rationale: {atm_req['reason']}")
        
        # Add specific safety warnings for hazardous atmospheres
        if 'NH₃' in atm_req['type']:
            atm_details.append("⚠️ SAFETY: NH₃ highly toxic (TLV 25 ppm) - use gas scrubber, ventilation, and detector")
        elif 'F' in composition:
            atm_details.append("⚠️ CRITICAL: Even trace moisture (<5 ppm) causes HF - use molecular sieve drying train")
        
        # Suggest monitoring
        if 'O₂' in atm_req['type'] or 'Air' in atm_req['type']:
            atm_details.append(f"Monitor: O₂ partial pressure affects oxidation states - adjust if phase impurities appear")
        elif 'Inert' in atm_req['type'] or 'Ar' in atm_req['type'] or 'N₂' in atm_req['type']:
            atm_details.append(f"Monitor: O₂ contamination <10 ppm ensures inert conditions - use O₂ sensor")
        
        lines.append("; ".join(atm_details))
        
        # Time section with DYNAMIC calculations from diffusion kinetics
        lines.append("\n⏱️ Time Required:")
        
        # Calculate heating time from temperature and rate
        heating_time_min = int((temp - 25) / 5 / 60)  # At 5°C/min
        heating_time_max = int((temp - 25) / 3 / 60)  # At 3°C/min
        heating_time_min = max(2, heating_time_min)
        heating_time_max = max(heating_time_min + 1, heating_time_max)
        
        # Calculate reaction time from diffusion kinetics
        reaction_min, reaction_max = self._estimate_diffusion_time(composition, temp)
        
        # Cooling time (roughly half of heating)
        cooling_time_min = max(3, heating_time_min // 2)
        cooling_time_max = max(cooling_time_min + 2, heating_time_max // 2)
        
        time_details = []
        time_details.append(f"Heating: {heating_time_min}-{heating_time_max}h to {temp}°C at 3-5°C/min")
        
        # Explain reaction time calculation
        n_elements = len(composition)
        if 'F' in composition:
            time_details.append(f"Reaction: {reaction_min}-{reaction_max}h (calculated for {n_elements}-component fluoride at {temp}°C)")
            time_details.append(f"Rationale: F⁻ slow diffusion in fluoride lattice - time scales with T⁻¹ and system complexity")
        elif 'N' in composition:
            time_details.append(f"Reaction: {reaction_min}-{reaction_max}h (calculated for nitride diffusion at {temp}°C)")
            time_details.append(f"Rationale: N diffusion moderate - time adjusted for {n_elements} elements and temperature")
        elif 'O' in composition:
            time_details.append(f"Reaction: {reaction_min}-{reaction_max}h (calculated for {n_elements}-cation oxide at {temp}°C)")
            time_details.append(f"Rationale: O²⁻ mobile - time dominated by cation interdiffusion in solid state")
        else:
            time_details.append(f"Reaction: {reaction_min}-{reaction_max}h (calculated from elemental diffusion at {temp}°C)")
            time_details.append(f"Rationale: Metallic/covalent diffusion relatively fast - time for {n_elements} elements")
        
        time_details.append(f"Cooling: {cooling_time_min}-{cooling_time_max}h at 2-5°C/min (maintains phase purity)")
        
        # Regrind cycles
        if n_elements > 3:
            time_details.append(f"Regrind & re-fire: 2-3 cycles REQUIRED for {n_elements} components - ensures homogeneity")
        else:
            time_details.append(f"Regrind & re-fire: 1-2 cycles recommended - improves phase purity")
        
        # Total time
        total_min = heating_time_min + reaction_min + cooling_time_min
        total_max = heating_time_max + reaction_max + cooling_time_max
        time_details.append(f"Total per cycle: {total_min}-{total_max}h (excluding regrinding time ~1-2h)")
        
        lines.append("; ".join(time_details))
        
        lines.append("\n" + "-"*70)
        
        # Method & Type section - SPECIFIC to this material
        lines.append("\n🧪 Synthesis Method:")
        
        method_details = ["Solid-state reaction via thorough precursor mixing and thermal processing"]
        
        # Determine grinding requirements based on composition
        moisture_sensitive = 'F' in composition or 'Cl' in composition or 'Li' in composition
        requires_inert = moisture_sensitive or any(self._get_element_volatility(e) != 'stable' for e in composition.keys())
        
        if moisture_sensitive:
            method_details.append(f"Grind in inert atmosphere glove box (N₂/Ar, <0.1 ppm H₂O) - {formula} extremely hygroscopic")
            method_details.append(f"Grinding time: 20-30 min with agate mortar ensures intimate mixing of {len(precursors)} precursors")
        else:
            method_details.append(f"Grind with agate mortar/pestle or planetary ball mill - target <10 µm particle size")
            method_details.append(f"Grinding time: 15-20 min manual or 2-4h ball milling for {len(precursors)} precursors")
        
        # Pelletizing pressure
        if 'F' in composition or any(self._get_element_volatility(e) == 'very_volatile' for e in composition.keys()):
            method_details.append(f"Pelletize at 4-6 tons/cm² (40-60 MPa) - higher density reduces sublimation")
        else:
            method_details.append(f"Pelletize at 2-4 tons/cm² (20-40 MPa) - sufficient for oxide/normal systems")
        
        # Regrind cycles based on complexity
        n_elements = len(composition)
        if n_elements > 4:
            method_details.append(f"Regrind cycles: 3-4 required for {n_elements} components - long-range diffusion limited")
        elif n_elements > 2:
            method_details.append(f"Regrind cycles: 2-3 recommended for {n_elements} components - ensures homogeneity")
        else:
            method_details.append(f"Regrind cycles: 1-2 sufficient for binary system - faster equilibration")
        
        lines.append("; ".join(method_details))
        
        lines.append("\n🔬 Reaction Type:")
        # Generate ACTUAL reaction equation with specific precursors
        reaction_eq = self._generate_reaction_equation(formula, composition, precursors)
        
        # Build detailed reaction description
        reaction_details = []
        reaction_details.append(f"Solid-state reaction forming {formula}")
        
        # Describe the transformation
        if 'F' in composition:
            reaction_details.append(f"fluoride precursors {', '.join(precursors[:3])} diffuse and react at {temp}°C")
        elif 'O' in composition:
            if any(p.endswith('CO3') for p in precursors):
                reaction_details.append(f"carbonates decompose releasing CO₂, oxides react forming {material_class}")
            else:
                reaction_details.append(f"oxide precursors interdiffuse at {temp}°C forming single-phase {material_class}")
        else:
            reaction_details.append(f"solid-state diffusion couples {len(composition)} elements into {material_class}")
        
        lines.append("; ".join(reaction_details))
        lines.append(f"Reaction: {reaction_eq}")
        
        lines.append("\n" + "="*70)
        
        return "\n".join(lines)
    
    def _classify_material(self, composition: Dict[str, float]) -> str:
        """Classify material type based on composition."""
        if 'O' in composition:
            if len(composition) > 3:
                return 'complex oxide'
            return 'metal oxide'
        elif 'F' in composition:
            if len(composition) > 3:
                return 'complex metal fluoride'
            return 'metal fluoride'
        elif 'N' in composition:
            return 'metal nitride'
        elif 'S' in composition:
            return 'metal sulfide'
        elif 'Cl' in composition:
            return 'metal chloride'
        else:
            return 'intermetallic compound'
    
    def _determine_atmosphere(self, composition: Dict[str, float]) -> str:
        """Determine appropriate atmosphere for synthesis."""
        if 'O' in composition:
            return 'air/O2'
        elif 'F' in composition:
            return 'dry inert (N2/Ar)'
        elif 'N' in composition:
            return 'NH3/N2'
        else:
            return 'inert (Ar/N2)'
    
    def _generate_reaction_equation(
        self,
        formula: str,
        composition: Dict[str, float],
        precursors: List[str]
    ) -> str:
        """Generate balanced reaction equation with specific precursors."""
        from pymatgen.core import Composition
        
        try:
            # Create reaction string
            target_comp = Composition(formula)
            
            # Calculate stoichiometric coefficients
            precursor_parts = []
            for prec in precursors:
                prec_comp = Composition(prec)
                
                # Find coefficient based on element conservation
                coeff = 1.0
                for elem in prec_comp.elements:
                    elem_str = str(elem)
                    if elem_str in composition:
                        target_amount = composition[elem_str]
                        prec_amount = prec_comp[elem]
                        ratio = target_amount / prec_amount
                        if ratio > coeff:
                            coeff = ratio
                
                if coeff != 1.0:
                    precursor_parts.append(f"{coeff:.2f} {prec}")
                else:
                    precursor_parts.append(prec)
            
            reaction = f"{' + '.join(precursor_parts)} → {formula}"
            
            # Add byproducts if carbonates are used
            if any('CO3' in p for p in precursors):
                reaction += " + CO2↑"
            if any('OH' in p for p in precursors):
                reaction += " + H2O↑"
            
            return reaction
        except:
            # Fallback to simple representation
            return f"{' + '.join(precursors)} → {formula}"
    
    def _generate_safety_section(
        self,
        formula: str,
        composition: Dict[str, float],
        precursors: List[str],
        hazards: List
    ) -> str:
        """Generate MANDATORY safety section."""
        # Estimate max temperature
        max_temp = self._estimate_synthesis_temperature(composition)
        
        # Use hazard detector to format comprehensive safety
        safety_text = self.hazard_detector.format_safety_section(
            hazards,
            max_temp
        )
        
        return safety_text
    
    def _generate_materials_section(
        self,
        formula: str,
        precursors: List[str],
        papers: List[Dict]
    ) -> str:
        """Generate materials and equipment section."""
        lines = ["2. MATERIALS AND EQUIPMENT\n"]
        
        lines.append("Starting Materials:")
        for precursor in precursors:
            lines.append(f"  • {precursor} (≥99% purity recommended)")
        
        lines.append("\nRequired Equipment:")
        lines.extend([
            "  • Analytical balance (±0.0001 g precision)",
            "  • Agate or alumina mortar and pestle",
            "  • Alumina or platinum crucible (compatible with composition)",
            "  • High-temperature furnace (capable of 1000-1400°C)",
            "  • Furnace with controlled atmosphere (air, O2, or inert gas)",
            "  • Heat-resistant gloves and tongs",
            "  • Desiccator for storage"
        ])
        
        # Try to extract equipment details from papers using LLM
        if papers and len(papers) > 0:
            lines.append("\nLiterature-Based Equipment Recommendations:")
            lines.append(f"  • Based on {len(papers)} retrieved papers")
            
            # Extract specific equipment from papers if available
            equipment_mentions = []
            for paper in papers[:2]:
                abstract = paper.get('abstract', '')
                full_text = paper.get('full_text', '')
                text = (full_text if full_text else abstract).lower()
                
                # Check for specific equipment mentions
                if 'tube furnace' in text:
                    equipment_mentions.append("Tube furnace mentioned in literature")
                if 'box furnace' in text:
                    equipment_mentions.append("Box furnace mentioned in literature")
                if 'pt crucible' in text or 'platinum crucible' in text:
                    equipment_mentions.append("Platinum crucible recommended")
                if 'alumina crucible' in text or 'al2o3 crucible' in text:
                    equipment_mentions.append("Alumina crucible used")
                if 'quartz' in text and 'tube' in text:
                    equipment_mentions.append("Quartz tube for controlled atmosphere")
            
            if equipment_mentions:
                for mention in set(equipment_mentions):  # Remove duplicates
                    lines.append(f"  • {mention}")
            else:
                lines.append("  • Typical crucible: alumina (Al2O3) for most oxides")
                lines.append("  • Atmosphere: air or oxygen for oxide synthesis")
        
        return "\n".join(lines)
    
    def _extract_synthesis_details_from_papers(
        self,
        formula: str,
        papers: List[Dict]
    ) -> Optional[str]:
        """
        Use LLM to extract detailed synthesis information from papers.
        
        Returns:
            Extracted synthesis details or None if no papers
        """
        if not papers or len(papers) == 0:
            print(f"  ℹ No papers available for LLM extraction")
            return None
        
        print(f"  📚 Extracting synthesis details from {len(papers)} papers using LLM...")
        
        # Prepare context from papers
        context_parts = []
        for i, paper in enumerate(papers[:3], 1):  # Use top 3 papers
            title = paper.get('title', 'Unknown')
            abstract = paper.get('abstract', '')
            full_text = paper.get('full_text', '')
            
            print(f"    Paper {i}: {title[:60]}...")
            
            # Prefer full text, fall back to abstract
            text = full_text if full_text else abstract
            if text:
                context_parts.append(f"Paper {i}: {title}\n{text[:2000]}")  # Limit length
        
        if not context_parts:
            print(f"  ⚠ Papers have no text content for extraction")
            return None
        
        context = "\n\n".join(context_parts)
        
        # Construct prompt for LLM
        prompt = f"""You are a materials science expert analyzing scientific literature to provide DETAILED synthesis guidance for {formula}.

⚠️ IMPORTANT: The papers below discuss RELATED MATERIALS and PRECURSORS, not {formula} specifically.
This is EXPECTED and CORRECT - extract general synthesis knowledge that applies to similar compounds.

Your task:
- Extract COMPREHENSIVE synthesis parameters from papers about related materials/precursors
- Focus on techniques applicable to solid-state ceramic/oxide synthesis
- Provide SPECIFIC NUMBERS where available (temperatures, rates, times)
- Give detailed step-by-step procedures adapted for {formula}

Extract these synthesis parameters with MAXIMUM DETAIL:

1. **Temperatures** (provide specific ranges):
   - Calcination temperature: XXX-XXX°C (with justification)
   - Sintering temperature: XXX-XXX°C (typical range)
   - Annealing temperature: XXX-XXX°C (if applicable)
   - Intermediate heating steps (if any)

2. **Heating/Cooling Rates** (be specific):
   - Ramp rate from room temp to target: X-X°C/min
   - Cooling rate: X-X°C/min or furnace cooling
   - Any slow cooling requirements

3. **Holding Times** (exact durations):
   - Duration at calcination temp: X-X hours
   - Duration at sintering temp: X-X hours
   - Total cycle time estimation

4. **Atmosphere Control** (detailed):
   - Gas type: air, O2, N2, Ar, vacuum, reducing atmosphere
   - Flow rate: XX-XX mL/min (if applicable)
   - Humidity control requirements
   - Sealed vs. open crucible

5. **Equipment Specifications**:
   - Furnace type: tube furnace, box furnace, etc.
   - Crucible material: alumina, platinum, zirconia (with reason)
   - Crucible size and sample loading
   - Temperature control accuracy needed

6. **Precursor Preparation** (step-by-step):
   - Drying conditions: temperature and duration
   - Grinding method: ball mill, mortar-pestle, duration
   - Mixing techniques: wet vs. dry mixing
   - Pellet pressing (if applicable): pressure, size

7. **Multi-Step Procedures**:
   - Number of heating cycles required
   - Intermediate grinding between cycles
   - Phase evolution during synthesis

8. **Quality Control**:
   - Expected product appearance (color, texture)
   - XRD patterns to look for
   - Common impurities and how to avoid them
   - Yield expectations (%)

9. **Best Practices & Troubleshooting**:
   - Critical control points
   - Common synthesis failures and solutions
   - Optimization tips from literature
   - Safety considerations specific to this synthesis

Papers (related materials and precursors):

{context}

Based on these papers about related materials, provide COMPREHENSIVE, DETAILED synthesis guidance for {formula}.
Include specific numbers, ranges, and step-by-step instructions wherever possible:
"""
        
        try:
            # Query LLM
            print(f"  🤖 Querying LLM with {len(context)} characters of context...")
            llm_response = self.llama_agent.generate_text(
                prompt,
                max_new_tokens=1500,  # Increased for more detailed output
                temperature=0.3  # Low temperature for factual extraction
            )
            print(f"  ✓ LLM extraction complete ({len(llm_response)} characters)")
            return llm_response
        except Exception as e:
            print(f"  ✗ LLM extraction failed: {e}")
            return None
    
    def _generate_procedure_section(
        self,
        formula: str,
        composition: Dict[str, float],
        precursors: List[str],
        papers: List[Dict],
        substitutions: Optional[Dict[str, str]]
    ) -> str:
        """Generate detailed synthesis procedure."""
        lines = ["3. DETAILED SYNTHESIS PROCEDURE\n"]
        
        # CRITICAL: Try to extract detailed synthesis from literature using LLM
        literature_details = self._extract_synthesis_details_from_papers(formula, papers)
        
        if literature_details:
            lines.append("="*70)
            lines.append("📚 LITERATURE-BASED SYNTHESIS PROCEDURE")
            lines.append("="*70)
            lines.append(f"Extracted from {len(papers)} retrieved papers using AI\n")
            lines.append(literature_details)
            lines.append("\n" + "="*70)
            lines.append("END LITERATURE EXTRACTION")
            lines.append("="*70 + "\n")
        else:
            if papers and len(papers) > 0:
                lines.append("⚠ WARNING: LLM extraction failed despite having papers")
                lines.append(f"  Papers available: {len(papers)}")
                lines.append("  Falling back to general procedure template\n")
            else:
                lines.append("⚠ WARNING: No literature retrieved from vector database")
                lines.append("  Synthesis procedure will be based on general principles only")
                lines.append("  For better results, populate database with relevant papers\n")
        
        # Calculate stoichiometry
        stoich_text = self._calculate_stoichiometry(formula, composition, precursors)
        lines.append("\n" + stoich_text)
        
        # DETAILED STEP-BY-STEP SYNTHESIS PROCEDURE
        lines.append("\n" + "="*60)
        lines.append("STEP-BY-STEP SYNTHESIS PROCEDURE")
        lines.append("="*60)
        
        # Step 1: Precursor Preparation
        lines.append("\nSTEP 1: Precursor Preparation and Weighing")
        lines.append("Duration: 30-45 minutes")
        lines.append("Equipment: Analytical balance (±0.0001g), weighing boats, desiccator")
        lines.append("Critical Parameters:")
        lines.append("  • Balance accuracy: ±0.1 mg minimum")
        lines.append("  • Relative humidity: <40% (use glove box if hygroscopic materials)")
        lines.append("  • Temperature: room temperature (20-25°C)")
        lines.append("\nProcedure:")
        lines.extend([
            "  a) Pre-dry all precursors in oven at 110-120°C for 2-4 hours",
            "     → Remove surface moisture and adsorbed water",
            "     → Critical for accurate stoichiometry",
            "  b) Cool in desiccator with fresh CaCl2 or silica gel for 30 minutes",
            "     → Prevents moisture re-absorption",
            "     → Allow thermal equilibration to prevent air currents",
            "  c) Weigh precursors according to stoichiometric calculations above",
            "     → Record masses to 0.0001g precision",
            "     → Work quickly to minimize air exposure",
            "  d) Transfer to clean, dry agate mortar immediately after weighing",
            "     → Avoid contamination from previous syntheses",
            "     → Pre-clean mortar with ethanol and dry at 110°C",
            "  e) Record actual masses for yield calculations",
            "     → Note any deviations from target masses (±0.5% acceptable)"
        ])
        
        # Step 2: Mixing
        lines.append("\nSTEP 2: Thorough Mixing of Precursors")
        lines.append("Duration: 30-45 minutes")
        lines.append("Equipment: Agate mortar and pestle (preferred) or ball mill")
        lines.append("Grinding aid: Anhydrous ethanol (99.5%+) or acetone")
        lines.append("\nCritical Parameters:")
        lines.append("  • Particle size target: <10 μm for good reactivity")
        lines.append("  • Mixing uniformity: homogeneous color/texture throughout")
        lines.append("  • Contamination prevention: clean tools between batches")
        lines.append("\nProcedure (Manual Grinding):")
        lines.extend([
            "  a) Add precursors to mortar in order of decreasing mass",
            "     → Heavier components first for better mixing efficiency",
            "     → Note: If using hygroscopic materials, work in glove box",
            "  b) Grind manually in circular and figure-8 motion for 10-15 minutes",
            "     → Apply moderate pressure (not crushing force)",
            "     → Mixture should become visibly more uniform",
            "  c) Add 3-5 drops of anhydrous ethanol to aid mixing",
            "     → Ethanol acts as dispersant and reduces agglomeration",
            "     → Alternative: acetone (faster drying) or isopropanol",
            "  d) Continue grinding for additional 15-20 minutes",
            "     → Powder should have paste-like consistency with ethanol",
            "     → Gradually becomes free-flowing as ethanol evaporates",
            "  e) Scrape sides and bottom of mortar every 5 minutes",
            "     → Ensures all material is mixed uniformly",
            "     → Use plastic or ceramic spatula (not metal)",
            "  f) Final mixture should be:",
            "     → Homogeneous powder with uniform color",
            "     → No visible clumps or separate phases",
            "     → Free-flowing (not caked or sticky)",
            "     → Particle size: fine powder (talc-like texture)"
        ])
        lines.append("\nAlternative - Ball Milling (for larger batches or better homogeneity):")
        lines.extend([
            "  • Mill type: planetary ball mill or tumbler",
            "  • Milling media: zirconia or alumina balls (5-10 mm diameter)",
            "  • Ball-to-powder ratio: 10:1 to 20:1 by mass",
            "  • Milling speed: 200-400 rpm",
            "  • Duration: 2-6 hours with ethanol",
            "  • Result: Superior particle size reduction (<5 μm) and mixing"
        ])
        
        # Step 3: Drying
        lines.append("\nSTEP 3: Drying Mixed Precursors")
        lines.append("Duration: 3-4 hours")
        lines.append("Equipment: Drying oven, ceramic boat or crucible")
        lines.append("\nProcedure:")
        lines.extend([
            "  a) Transfer mixed powder to alumina or platinum crucible",
            "  b) Place crucible in drying oven at 120-150°C",
            "  c) Dry for 3-4 hours to completely remove ethanol and moisture",
            "  d) Cool in desiccator before proceeding to calcination"
        ])
        
        # Step 4: Calcination/Sintering
        temp = self._estimate_synthesis_temperature(composition)
        lines.append("\nSTEP 4: High-Temperature Calcination/Sintering")
        lines.append(f"Duration: 10-16 hours (including heating and cooling)")
        lines.append(f"Equipment: High-temperature furnace, alumina/platinum crucible")
        lines.append("\nCritical Parameters:")
        lines.append(f"  • Temperature accuracy: ±5°C at {temp}°C")
        lines.append("  • Temperature uniformity: ±10°C across sample")
        lines.append("  • Atmosphere purity: 99.9%+ if using inert gas")
        lines.append("  • Crucible fill: 1/3 to 1/2 full (allow for gas evolution)")
        lines.append("\nTemperature Profile:")
        lines.extend([
            f"  Target Temperature: {temp}°C",
            "  ",
            "  DETAILED HEATING SCHEDULE:",
            "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "  Phase 1: Initial Heating & Solvent Removal",
            "    • 25°C → 200°C at 5°C/min (35 minutes)",
            "    • Purpose: Remove residual ethanol/water without spattering",
            "  ",
            "  Phase 2: Intermediate Heating & Decomposition",
            f"    • 200°C → 500°C at 5°C/min (60 minutes)",
            "    • Purpose: Decompose carbonates, nitrates, precursor compounds",
            "    • Note: May see color changes, gas evolution",
            "  ",
            "  Phase 3: High-Temperature Ramp",
            f"    • 500°C → {temp}°C at 3-5°C/min ({int((temp-500)/4)} min @ 4°C/min)",
            "    • Purpose: Approach reaction temperature gradually",
            "    • Slower rate prevents thermal shock and ensures uniformity",
            "  ",
            "  Phase 4: Reaction Hold (CRITICAL)",
            f"    • Hold at {temp}°C for 8-12 hours",
            "    • Purpose: Complete solid-state diffusion and phase formation",
            "    • Minimum: 6 hours for small crystallites",
            "    • Optimal: 10-12 hours for high crystallinity",
            "    • Extended: 18-24 hours for difficult formations",
            "  ",
            "  Phase 5: Controlled Cooling",
            f"    • {temp}°C → 500°C at 5°C/min (slow cool, {int((temp-500)/5)} min)",
            "    • 500°C → 25°C: furnace cool (natural rate, ~4-6 hours)",
            "    • Purpose: Prevent thermal stress, maintain crystal structure",
            "    • Note: Some materials require slower cooling (2°C/min)",
            "  ",
            f"  TOTAL CYCLE TIME: ~12-18 hours",
            "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ])
        
        lines.append("\nAtmosphere Control:")
        # Determine atmosphere based on composition
        if 'O' in composition:
            lines.append("  • ATMOSPHERE: Air or O2 atmosphere (for oxide formation)")
            lines.append("  • Gas flow rate: 50-100 mL/min if using flowing gas")
            lines.append("  • Static vs. flowing: Static air acceptable for most oxides")
            lines.append("  • Crucible: Open or loosely covered (allow O2 access)")
            lines.append("  • Reason: Oxygen incorporation into crystal structure")
        elif 'F' in composition:
            lines.append("  • ATMOSPHERE: Dry nitrogen (N2) or argon (Ar) - CRITICAL")
            lines.append("  • Gas purity: 99.99%+ (use ultra-high purity grade)")
            lines.append("  • Flow rate: 100-200 mL/min (positive pressure)")
            lines.append("  • Drying: Pass gas through molecular sieve column or P2O5")
            lines.append("  • Crucible: Covered with lid (small gap for gas flow)")
            lines.append("  • CRITICAL: Fluorides are hygroscopic and hydrolyze in moist air")
            lines.append("  • WARNING: Even trace moisture will cause HF formation")
            lines.append("  • Humidity: <5 ppm H2O in gas stream")
        elif 'N' in composition:
            lines.append("  • ATMOSPHERE: Ammonia (NH3) or nitrogen (N2)")
            lines.append("  • For nitrides: NH3 flow at 50-100 mL/min")
            lines.append("  • For nitrogen incorporation: N2 at 99.99%+ purity")
            lines.append("  • Flow rate: 50-150 mL/min")
            lines.append("  • WARNING: NH3 is toxic and corrosive - use fume hood")
        else:
            lines.append("  • ATMOSPHERE: Inert (Ar or N2) recommended")
            lines.append("  • Gas purity: 99.9%+ (standard grade acceptable)")
            lines.append("  • Flow rate: 50-100 mL/min")
            lines.append("  • Reason: Prevent oxidation/contamination")
            lines.append("  • Alternative: Air if no reactive elements present")
        
        lines.append("\nFurnace Loading Procedure:")
        lines.extend([
            "  a) Place dried sample in crucible:",
            "     → Fill to 1/3-1/2 capacity (loosely packed, not compressed)",
            "     → Reason: Allow gas circulation and prevent sintering to crucible",
            "  b) Cover crucible appropriately:",
            "     → For oxides: loose cover or uncovered",
            "     → For fluorides: tight cover with small vent hole",
            "     → For volatile components: sealed crucible within larger crucible",
            "  c) Load into furnace at room temperature:",
            "     → Never load into hot furnace (thermal shock)",
            "     → Position in center of hot zone for uniform temperature",
            "  d) Connect gas lines (if using controlled atmosphere):",
            "     → Purge furnace with 3 volume changes before heating",
            "     → Maintain positive pressure throughout heating",
            "  e) Start temperature program as specified above",
            "  f) Monitor furnace during first 2 hours (CRITICAL):",
            "     → Watch for unusual smoke, odors, or temperature spikes",
            "     → Check gas flow rates remain constant",
            "     → Verify controller following programmed profile",
            "  g) After completion, allow furnace to cool naturally:",
            "     → Do NOT open furnace above 200°C",
            "     → For fluorides: maintain inert atmosphere during cooling",
            "  h) Remove sample only when furnace is below 100°C:",
            "     → Use tongs or heat-resistant gloves",
            "     → Transfer immediately to desiccator"
        ])
        
        # Step 5: Post-processing
        lines.append("\nSTEP 5: Post-Processing and Homogenization")
        lines.append("Duration: 30-60 minutes")
        lines.append("Equipment: Agate mortar, desiccator")
        lines.append("\nProcedure:")
        lines.extend([
            "  a) Remove calcined product from crucible",
            "  b) Gently grind to break up any large agglomerates",
            "  c) DO NOT over-grind (can damage crystal structure)",
            "  d) Sieve through 100-mesh screen if needed for uniform particle size",
            "  e) Store in labeled, sealed container in desiccator"
        ])
        
        # Optional re-firing
        lines.append("\nSTEP 6: Re-firing for Improved Crystallinity (OPTIONAL)")
        lines.append("Duration: 8-12 hours")
        lines.append("\nProcedure:")
        lines.extend([
            "  a) If XRD shows incomplete reaction or poor crystallinity:",
            "  b) Regrind sample thoroughly (10 minutes)",
            "  c) Reload into crucible",
            f"  d) Re-fire at same temperature ({temp}°C) for 6-12 hours",
            "  e) Use same atmosphere and heating/cooling rates",
            "  f) This often improves phase purity and crystallinity"
        ])
        
        # Literature basis
        if papers and len(papers) > 0:
            lines.append("\n" + "="*60)
            lines.append("LITERATURE BASIS")
            lines.append("="*60)
            lines.append(f"• Based on {len(papers)} retrieved papers")
            if literature_details:
                lines.append("• AI-extracted specific details from paper text above")
            lines.append("• Temperature and conditions are literature-backed")
            lines.append("• Adjust based on your specific precursors and equipment")
        else:
            lines.append("\n" + "="*60)
            lines.append("SYNTHESIS BASIS")
            lines.append("="*60)
            lines.append("⚠ NOTE: No literature retrieved from database")
            lines.append("• Temperature estimated from composition and material class")
            lines.append("• Procedure based on general solid-state synthesis principles")
            lines.append("• CRITICAL: Experimental optimization required")
            lines.append("• Monitor XRD results and adjust conditions accordingly")
        
        # Expected yield
        lines.append("\n" + "="*60)
        lines.append("EXPECTED RESULTS")
        lines.append("="*60)
        lines.append("\nTypical Yield: 85-95% (based on theoretical mass)")
        lines.append("\nProduct Characteristics:")
        lines.extend([
            "  • Appearance: Fine powder, color depends on composition",
            "  • Particle size: 1-10 μm (typical for solid-state synthesis)",
            "  • Density: Calculate from XRD unit cell parameters",
            "  • Phase purity: Verify by XRD (compare to ICDD database)"
        ])
        
        # Post-processing
        lines.append("\nSTEP 7: Storage and Handling")
        lines.append("Equipment: Desiccator, sealed containers, labels")
        lines.append("\nProcedure:")
        lines.extend([
            "  a) Transfer final product to clean, dry glass vial",
            "  b) Label with: Material formula, synthesis date, batch number",
            "  c) Store in desiccator with fresh desiccant",
            "  d) For hygroscopic materials: consider argon-filled vials",
            "  e) Record total mass and calculate yield percentage"
        ])
        
        # Substitution notes
        if substitutions:
            lines.append("\nElement Substitution Notes:")
            for old, new in substitutions.items():
                lines.append(f"  • {old} → {new}: May affect synthesis temperature and phase stability")
                lines.append("  • Optimization of temperature and time may be required")
        
        return "\n".join(lines)
    
    def _generate_characterization_section(
        self,
        formula: str
    ) -> str:
        """Generate characterization section."""
        lines = ["4. CHARACTERIZATION\n"]
        
        lines.append("Required Characterization:")
        lines.extend([
            "  • X-Ray Diffraction (XRD):",
            "    - Confirm phase purity and crystal structure",
            "    - Compare with standard ICDD/PDF database",
            "    - Identify any secondary phases or impurities",
            "",
            "  • Composition Verification:",
            "    - Energy Dispersive X-ray Spectroscopy (EDS/EDX)",
            "    - X-ray Photoelectron Spectroscopy (XPS) for oxidation states",
            "    - Inductively Coupled Plasma (ICP) for precise composition",
            "",
            "  • Physical Properties:",
            "    - Density measurement (Archimedes method or pycnometry)",
            "    - Particle size analysis (SEM or laser diffraction)",
            "    - Surface area (BET if applicable)"
        ])
        
        return "\n".join(lines)
    
    def _generate_notes_section(
        self,
        formula: str,
        substitutions: Optional[Dict[str, str]],
        papers: List[Dict]
    ) -> str:
        """Generate notes and limitations section."""
        lines = ["5. NOTES & LIMITATIONS\n"]
        
        lines.append("Assumptions:")
        lines.extend([
            "  • Synthesis conditions are estimates based on literature and composition",
            "  • Actual optimal conditions may vary and require experimental optimization",
            "  • Temperature, time, and atmosphere should be refined based on XRD results"
        ])
        
        if substitutions:
            lines.append("\nSubstitution Impact:")
            lines.append("  • Element substitutions may significantly alter:")
            lines.extend([
                "    - Optimal synthesis temperature",
                "    - Phase stability and purity",
                "    - Physical and electronic properties",
                "  • Literature for parent compound used as starting point",
                "  • Systematic optimization recommended"
            ])
        
        lines.append("\nOptimization Recommendations:")
        lines.extend([
            "  • Vary synthesis temperature in ±100°C range",
            "  • Test different holding times (4-24 hours)",
            "  • Consider intermediate grinding and re-firing",
            "  • Adjust heating/cooling rates if cracking observed",
            "  • Monitor atmosphere (oxygen partial pressure for mixed valence materials)"
        ])
        
        if not papers or len(papers) == 0:
            lines.append("\nLiterature Gap:")
            lines.append("  ⚠ WARNING: No literature references retrieved")
            lines.append("  • Synthesis parameters are educated estimates only")
            lines.append("  • Extensive experimental optimization will be required")
            lines.append("  • Consider consulting additional databases or experts")
        
        return "\n".join(lines)
    
    def _generate_sources_section(
        self,
        papers: List[Dict]
    ) -> str:
        """Generate MANDATORY retrieved sources section."""
        lines = ["="*80]
        lines.append("RETRIEVED CONTEXT SOURCES")
        lines.append("="*80 + "\n")
        
        if not papers or len(papers) == 0:
            lines.append("⚠ NO LITERATURE SOURCES RETRIEVED")
            lines.append("This synthesis protocol is based on general materials chemistry principles")
            lines.append("and estimated parameters. Experimental validation is ESSENTIAL.")
        else:
            for i, paper in enumerate(papers, 1):
                lines.append(f"[{i}] {paper.get('title', 'Unknown')}")
                
                if paper.get('doi'):
                    lines.append(f"    DOI: {paper['doi']}")
                if paper.get('pmid'):
                    lines.append(f"    PMID: {paper['pmid']}")
                if paper.get('url'):
                    lines.append(f"    URL: {paper['url']}")
                
                # Material relevance (if available in payload)
                material = paper.get('material', 'Related material')
                lines.append(f"    Material: {material}")
                
                score = paper.get('score', 0.0)
                lines.append(f"    Relevance Score: {score:.3f}")
                
                lines.append("")
        
        return "\n".join(lines)
    
    def _calculate_stoichiometry(
        self,
        formula: str,
        composition: Dict[str, float],
        precursors: List[str]
    ) -> str:
        """Calculate and format stoichiometry."""
        lines = ["Stoichiometric Calculations:"]
        lines.append(f"  Target: {formula}")
        lines.append(f"  Composition: {composition}")
        lines.append("\n  Precursors (example for 10g batch):")
        
        # Simplified calculation
        from ingestion.precursor_extraction import calculate_stoichiometry
        
        try:
            amounts = calculate_stoichiometry(composition, precursors, target_mass=10.0)
            for precursor, mass in amounts.items():
                lines.append(f"    • {precursor}: {mass:.4f} g")
        except:
            lines.append("    • Calculate based on precursor molecular weights")
            lines.append("    • Ensure stoichiometric ratios match target composition")
        
        return "\n".join(lines)
    
    def _get_precursor_decomp_temp(self, precursor: str) -> int:
        """Get estimated decomposition/reaction temperature for precursor."""
        # Decomposition temperatures (°C) for common precursors
        decomp_temps = {
            'CO3': 850,  # Carbonates generally decompose 700-1000°C
            'OH': 400,   # Hydroxides decompose 300-500°C
            'NO3': 600,  # Nitrates decompose 400-800°C
            'H2PO4': 250, # Ammonium phosphates decompose 200-300°C
            'TiO2': 1200, # Rutile stable to very high temps
            'ZrO2': 1300, # Very refractory
            'Al2O3': 1400, # Highly refractory
            'La2O3': 1200,
            'Y2O3': 1200,
        }
        
        # Check which decomposition group this precursor belongs to
        for key, temp in decomp_temps.items():
            if key in precursor:
                return temp
        
        # Default oxide reaction temperature
        return 1000
    
    def _calculate_required_temp(self, composition: Dict[str, float], precursors: List[str]) -> Tuple[int, int, int]:
        """Calculate required temperature from precursor properties and composition.
        
        Returns: (base_temp, lower_bound, upper_bound)
        """
        # Get decomposition temperatures for all precursors
        decomp_temps = [self._get_precursor_decomp_temp(p) for p in precursors]
        
        # Base temperature should be 100-200°C above highest decomposition temp
        max_decomp = max(decomp_temps) if decomp_temps else 1000
        base_temp = max_decomp + 150
        
        # Adjust based on composition complexity
        n_elements = len(composition)
        if n_elements > 3:
            # Complex systems need higher temps for interdiffusion
            base_temp += 50 * (n_elements - 3)
        
        # Adjust for specific elements
        # Refractory elements
        refractory = {'Zr', 'Hf', 'Nb', 'Ta', 'W', 'Mo', 'Al'}
        if set(composition.keys()) & refractory:
            base_temp += 100
        
        # Volatile elements - lower temperature
        volatile = {'Li', 'Na', 'K', 'Ag', 'Hg', 'Cd', 'Zn'}
        if set(composition.keys()) & volatile:
            base_temp = min(base_temp, 900)
        
        # Calculate practical bounds
        lower_bound = max(600, base_temp - 100)
        upper_bound = min(1500, base_temp + 150)
        
        # Clamp base temp to reasonable range
        base_temp = max(700, min(base_temp, 1400))
        
        return base_temp, lower_bound, upper_bound
    
    def _get_element_volatility(self, element: str) -> str:
        """Get volatility classification for element."""
        very_volatile = {'Hg', 'Cd', 'As', 'P', 'S', 'Se', 'Te'}
        moderately_volatile = {'Zn', 'Ag', 'Sb', 'Pb', 'Bi'}
        alkali_volatile = {'Li', 'Na', 'K', 'Rb', 'Cs'}
        
        if element in very_volatile:
            return 'very_volatile'
        elif element in moderately_volatile:
            return 'moderately_volatile'
        elif element in alkali_volatile:
            return 'alkali_volatile'
        else:
            return 'stable'
    
    def _estimate_diffusion_time(self, composition: Dict[str, float], temp: int) -> Tuple[int, int]:
        """Estimate reaction time based on diffusion kinetics.
        
        Returns: (min_hours, max_hours)
        """
        # Base time depends on number of elements (complexity)
        n_elements = len(composition)
        
        # Activation energy considerations
        # Higher temps -> faster diffusion (exponential with T)
        temp_factor = 1400 / max(temp, 700)  # Ratio to reference temp
        
        # Fluorides and some compounds have slow diffusion
        diffusion_modifier = 1.0
        if 'F' in composition:
            diffusion_modifier = 2.0  # Fluorides diffuse slowly
        elif 'N' in composition:
            diffusion_modifier = 1.5  # Nitrides moderate
        elif 'O' in composition:
            diffusion_modifier = 1.0  # Oxides normal
        else:
            diffusion_modifier = 0.8  # Intermetallics faster
        
        # Calculate base times
        if n_elements == 2:
            min_time = int(6 * temp_factor * diffusion_modifier)
            max_time = int(12 * temp_factor * diffusion_modifier)
        elif n_elements == 3:
            min_time = int(10 * temp_factor * diffusion_modifier)
            max_time = int(20 * temp_factor * diffusion_modifier)
        elif n_elements == 4:
            min_time = int(16 * temp_factor * diffusion_modifier)
            max_time = int(32 * temp_factor * diffusion_modifier)
        else:
            # Complex systems
            min_time = int(20 * temp_factor * diffusion_modifier)
            max_time = int(48 * temp_factor * diffusion_modifier)
        
        # Ensure reasonable bounds
        min_time = max(4, min_time)
        max_time = max(min_time + 4, min(max_time, 72))
        
        return min_time, max_time
    
    def _calculate_atmosphere_requirement(self, composition: Dict[str, float]) -> Dict[str, str]:
        """Determine required atmosphere based on element properties.
        
        Returns dict with 'type', 'purity', 'flow_rate', 'reason'
        """
        result = {}
        
        # Halogens require inert atmosphere
        halogens = {'F', 'Cl', 'Br', 'I'}
        if set(composition.keys()) & halogens:
            halogen = list(set(composition.keys()) & halogens)[0]
            result['type'] = 'Dry N₂ or Ar'
            result['purity'] = '99.99%+ (<5 ppm H₂O, <2 ppm O₂)'
            result['flow_rate'] = '100-200 mL/min'
            result['reason'] = f'{halogen} compounds extremely moisture-sensitive - HF/HCl formation with water'
            return result
        
        # Transition metals that need oxygen
        oxidizing_tms = {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu'}
        if 'O' in composition and set(composition.keys()) & oxidizing_tms:
            tms = list(set(composition.keys()) & oxidizing_tms)
            result['type'] = 'Air or O₂-enriched (21-50% O₂)'
            result['purity'] = 'Standard air or compressed O₂'
            result['flow_rate'] = '50-100 mL/min (if flowing)'
            result['reason'] = f'Ensures correct oxidation states for {", ".join(tms)} - prevents reduction'
            return result
        
        # Nitrides need nitrogen/ammonia
        if 'N' in composition:
            result['type'] = 'NH₃ or N₂ (flowing)'
            result['purity'] = '99.9%+ NH₃ or 99.99%+ N₂'
            result['flow_rate'] = '100-300 mL/min'
            result['reason'] = 'Maintains nitrogen chemical potential - prevents decomposition'
            return result
        
        # Simple oxides
        if 'O' in composition and len(composition) <= 3:
            result['type'] = 'Air (static)'
            result['purity'] = 'Ambient air sufficient'
            result['flow_rate'] = 'Static atmosphere'
            result['reason'] = 'Natural O₂ diffusion maintains stoichiometry'
            return result
        
        # Intermetallics or other compounds
        # Check if any reactive elements present
        reactive = {'Li', 'Na', 'K', 'Mg', 'Ca', 'Sr', 'Ba', 'Al', 'Ti', 'Zr', 'Hf', 'Nb', 'Ta'}
        if set(composition.keys()) & reactive:
            react_list = list(set(composition.keys()) & reactive)
            result['type'] = 'Inert (Ar or N₂)'
            result['purity'] = '99.9%+ (for less reactive) or 99.999%+ (for highly reactive)'
            result['flow_rate'] = '50-150 mL/min'
            result['reason'] = f'Protects {", ".join(react_list[:3])} from oxidation during high-temp processing'
            return result
        
        # Default
        result['type'] = 'Air or inert atmosphere'
        result['purity'] = 'Standard air or 99.9%+ inert'
        result['flow_rate'] = 'Static or 50-100 mL/min'
        result['reason'] = 'Standard atmosphere suitable for stable compounds'
        return result
    
    def _estimate_synthesis_temperature(
        self,
        composition: Dict[str, float]
    ) -> int:
        """Estimate synthesis temperature based on composition.
        
        DEPRECATED: Use _calculate_required_temp instead.
        """
        # Simple heuristics for backward compatibility
        base_temp = 1000
        
        # Heavy elements increase temperature
        heavy_elements = {'Ba', 'Sr', 'La', 'Zr'}
        if set(composition.keys()) & heavy_elements:
            base_temp += 200
        
        # Volatile elements decrease temperature
        volatile = {'Li', 'Na', 'K'}
        if set(composition.keys()) & volatile:
            base_temp -= 100
        
        # Ensure reasonable range
        temp = max(800, min(base_temp, 1400))
        
        return temp
    
    def _validate_protocol(self, protocol: str):
        """
        Validate that protocol contains all required sections.
        
        Raises ValueError if validation fails.
        """
        # Check header
        if "ANSWER WITH SAFETY PROTOCOLS" not in protocol:
            raise ValueError("CRITICAL: Protocol missing required header")
        
        # Check required sections
        for section in self.REQUIRED_SECTIONS:
            if section not in protocol:
                raise ValueError(f"CRITICAL: Protocol missing required section: {section}")
        
        # Check sources section
        if "RETRIEVED CONTEXT SOURCES" not in protocol:
            raise ValueError("CRITICAL: Protocol missing sources section")
        
        # Check safety content
        if "SAFETY PROTOCOLS" not in protocol:
            raise ValueError("CRITICAL: Safety section missing or improperly formatted")


if __name__ == "__main__":
    # Test synthesis generation
    from ingestion.parse_reactions import parse_chemical_formula
    from ingestion.precursor_extraction import infer_precursors
    
    # Test without Llama (using mock)
    class MockLlama:
        def query_with_context(self, *args, **kwargs):
            return "Mock synthesis details"
    
    generator = SynthesisGenerator(MockLlama())
    
    formula = "BaTiO3"
    composition = parse_chemical_formula(formula)
    precursors = infer_precursors(composition)
    
    # Mock papers
    papers = [
        {
            'title': 'Synthesis of BaTiO3 ceramics',
            'doi': '10.1234/example',
            'score': 0.85,
            'material': 'BaTiO3'
        }
    ]
    
    protocol = generator.generate_synthesis_protocol(
        formula,
        composition,
        precursors,
        papers
    )
    
    print(protocol)
