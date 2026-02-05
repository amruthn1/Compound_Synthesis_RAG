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
        
        # Temperature section with MATERIAL-SPECIFIC analysis
        lines.append("\n📊 Temperature:")
        lines.append(f"  Analyze thermal stability of {', '.join(precursors[:3])} - typically {temp-100}-{temp+100}°C")
        lines.append(f"  for {material_class} based on precursor decomposition temperatures")
        lines.append(f"  and solid-state diffusion requirements")
        lines.append(f"")
        lines.append(f"  Recommended target: {temp}°C")
        
        # Material-specific temperature considerations
        if 'F' in composition:
            lines.append(f"  ⚠ Fluoride chemistry: Consider thermal stability of fluoride precursors")
            if 'Ag' in composition:
                lines.append(f"  ⚠ AgF2 decomposes above ~200°C - use lower temperatures or sealed crucible")
            if 'Ba' in composition:
                lines.append(f"  BaF2 is stable but requires high temps (900-1100°C) for diffusion")
        
        if 'Li' in composition or 'Na' in composition or 'K' in composition:
            lines.append(f"  ⚠ Alkali metals: Lower temperature ({temp-200}-{temp}°C) to prevent volatilization")
        
        if 'Ti' in composition or 'Zr' in composition:
            lines.append(f"  ⚠ Transition metal oxides: Higher temperatures ({temp}-{temp+200}°C) needed")
        
        lines.append(f"")
        lines.append(f"  Heating rate: 5-10°C/min (prevents thermal shock)")
        lines.append(f"  Hold time: 12-48 hours for complete solid-state diffusion and phase formation")
        lines.append(f"  Cooling rate: 2-5°C/min (controlled to maintain phase purity)")
        
        # Pressure section - SPECIFIC to this material's precursors
        lines.append("\n🔧 Pressure:")
        lines.append("  Ambient pressure typically suitable for solid-state ceramic synthesis")
        
        # Check for volatile precursors
        volatile_elements = {'Ag', 'Hg', 'Cd', 'Zn', 'As', 'Sb', 'P'}
        has_volatile = any(elem in volatile_elements for elem in composition.keys())
        
        if has_volatile:
            volatile_list = [e for e in composition.keys() if e in volatile_elements]
            lines.append(f"  ⚠ Volatile elements present ({', '.join(volatile_list)})")
            lines.append(f"  unless precursors have high vapor pressure or oxidation sensitivity requires vacuum")
        
        if 'F' in composition or 'Cl' in composition:
            lines.append(f"  ⚠ Halide precursors may sublime - consider sealed crucible within larger crucible")
        
        # Atmosphere section - MATERIAL-SPECIFIC recommendations
        lines.append("\n🌬️ Atmosphere:")
        atmosphere_type = self._determine_atmosphere(composition)
        
        if 'O' in composition and ('Ti' in composition or 'Fe' in composition or 'Mn' in composition):
            lines.append(f"  Air or O2 atmosphere for {material_class}")
            lines.append(f"  Inert atmosphere recommended for fluorides (moisture-sensitive, HF formation risk)")
        elif 'F' in composition:
            lines.append(f"  Inert atmosphere recommended for fluorides (moisture-sensitive, HF formation risk)")
            lines.append(f"  Use dry N2 or Ar (99.99%+ purity) with flow rate 100-200 mL/min")
            lines.append(f"  ⚠ CRITICAL: Even trace moisture causes HF formation")
            lines.append(f"  Pass gas through molecular sieve or P2O5 drying column (<5 ppm H2O)")
        elif 'N' in composition:
            lines.append(f"  NH3 or N2 atmosphere for nitride formation")
            lines.append(f"  ⚠ WARNING: NH3 is toxic - use proper ventilation and scrubbing")
        elif 'O' in composition:
            lines.append(f"  Air or oxygen atmosphere suitable for oxide synthesis")
        else:
            lines.append(f"  Inert atmosphere (Ar or N2, 99.9%+) to prevent oxidation")
        
        # Time section with SPECIFIC breakdown for this material
        lines.append("\n⏱️ Time Required:")
        heating_time = int((temp - 25) / 5 / 60)  # hours at 5°C/min
        lines.append(f"  Heating phase: 3-5 hours to reach target temperature at controlled rate (5-10°C/min)")
        lines.append(f"                 ({heating_time}-{heating_time*2} hours to reach {temp}°C)")
        
        # Material-specific reaction time
        if 'F' in composition and len(composition) > 3:
            lines.append(f"  Reaction phase: 12-48 hours hold at temperature for complete solid-state diffusion")
            lines.append(f"                  and phase formation; Complex fluoride ({len(composition)} elements)")
            lines.append(f"                  requires longer time (24-48h recommended)")
        elif material_class == 'complex oxide':
            lines.append(f"  Reaction phase: 18-36 hours (complex oxide with multiple cations)")
        else:
            lines.append(f"  Reaction phase: 12-24 hours hold at temperature")
        
        lines.append(f"  Cooling phase: 6-12 hours controlled cooling (2-5°C/min)")
        lines.append(f"  Multiple cycles may be needed with intermediate grinding for phase purity")
        
        total_min = heating_time + 12 + 6
        total_max = heating_time*2 + 48 + 12
        lines.append(f"  Total estimated time: {total_min}-{total_max} hours (single cycle)")
        
        lines.append("\n" + "-"*70)
        
        # Method & Type section - SPECIFIC to this material
        lines.append("\n🧪 Synthesis Method:")
        lines.append(f"  Solid-state reaction method: Intimately mix precursor powders via grinding,")
        lines.append(f"  compact into pellet to maximize contact area, heat to enable solid-state")
        lines.append(f"  diffusion and reaction between phases")
        
        lines.append("\n🔬 Reaction Type:")
        # Generate ACTUAL reaction equation with specific precursors
        reaction_eq = self._generate_reaction_equation(formula, composition, precursors)
        lines.append(f"  Solid-state reaction forming {formula} from precursors")
        lines.append(f"  {', '.join(precursors)} via high-temperature diffusion and phase formation")
        lines.append(f"")
        lines.append(f"  Reaction: {reaction_eq}")
        
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
    
    def _estimate_synthesis_temperature(
        self,
        composition: Dict[str, float]
    ) -> int:
        """Estimate synthesis temperature based on composition."""
        # Simple heuristics
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
