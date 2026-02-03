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
        sections.append(f"\n## Synthesis Protocol for {formula}\n")
        
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
        prompt = f"""Extract detailed synthesis information for {formula} from the following papers.

Focus on:
1. Specific temperatures (calcination, sintering, annealing)
2. Exact heating rates and cooling rates
3. Holding times at each temperature
4. Atmosphere requirements (air, oxygen, nitrogen, argon, vacuum)
5. Equipment used (furnace type, crucible material)
6. Step-by-step procedure
7. Precursor materials and their ratios
8. Any special techniques or precautions

Provide a concise but detailed synthesis procedure based on this literature.

{context}

SYNTHESIS DETAILS:"""
        
        try:
            # Query LLM
            print(f"  🤖 Querying LLM with {len(context)} characters of context...")
            llm_response = self.llama_agent.generate_text(
                prompt,
                max_new_tokens=800,
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
        lines.append("\nProcedure:")
        lines.extend([
            "  a) Dry all precursors in oven at 110°C for 2 hours to remove moisture",
            "  b) Cool in desiccator for 30 minutes before weighing",
            "  c) Weigh precursors according to stoichiometric calculations above",
            "  d) Transfer to clean, dry agate mortar immediately after weighing",
            "  e) Record actual masses for yield calculations"
        ])
        
        # Step 2: Mixing
        lines.append("\nSTEP 2: Thorough Mixing of Precursors")
        lines.append("Duration: 20-30 minutes")
        lines.append("Equipment: Agate mortar and pestle, ethanol (anhydrous, 99.5%)")
        lines.append("\nProcedure:")
        lines.extend([
            "  a) Add precursors to mortar in order of decreasing mass",
            "  b) Grind manually in circular motion for 10 minutes",
            "  c) Add 2-3 drops of anhydrous ethanol to aid mixing",
            "  d) Continue grinding for additional 10-15 minutes",
            "  e) Scrape sides and bottom of mortar to ensure complete mixing",
            "  f) Final mixture should be homogeneous powder with no visible clumps"
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
        lines.append(f"Duration: 8-14 hours (including heating and cooling)")
        lines.append(f"Equipment: High-temperature furnace, alumina/platinum crucible")
        lines.append("\nTemperature Profile:")
        lines.extend([
            f"  Target Temperature: {temp}°C",
            "  ",
            "  Detailed Heating Schedule:",
            f"    • 25°C → 300°C at 5°C/min (55 min)",
            f"    • 300°C → {temp}°C at 3-5°C/min (variable)",
            f"    • Hold at {temp}°C for 6-12 hours",
            f"    • {temp}°C → 25°C at 5°C/min (furnace cool, natural rate)",
            "  ",
            "  Total cycle time: ~10-16 hours"
        ])
        
        lines.append("\nAtmosphere Control:")
        # Determine atmosphere based on composition
        if 'O' in composition:
            lines.append("  • Air or O2 atmosphere (for oxide formation)")
            lines.append("  • Flow rate: 50-100 mL/min if using flowing gas")
        elif 'F' in composition:
            lines.append("  • Dry nitrogen or argon atmosphere (prevent hydrolysis)")
            lines.append("  • Flow rate: 100 mL/min")
            lines.append("  • CRITICAL: Ensure atmosphere is DRY (use drying column)")
        elif 'N' in composition:
            lines.append("  • Ammonia (NH3) or nitrogen atmosphere")
            lines.append("  • Flow rate: 50-100 mL/min")
        else:
            lines.append("  • Inert atmosphere (Ar or N2) recommended")
            lines.append("  • Flow rate: 50-100 mL/min")
        
        lines.append("\nProcedure:")
        lines.extend([
            "  a) Place dried sample in crucible (loosely packed, not compressed)",
            "  b) Cover crucible with lid (leave small gap for gas exchange)",
            "  c) Load into furnace at room temperature",
            "  d) Start temperature program as specified above",
            "  e) Monitor furnace periodically (first 2 hours critical)",
            "  f) After completion, allow furnace to cool naturally",
            "  g) Remove sample only when furnace is below 100°C"
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
