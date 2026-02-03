"""
SINGLE SHARED PIPELINE - THE ONLY SOURCE OF TRUTH

This module contains the ONLY pipeline function that performs all materials operations.
Both Colab and Streamlit MUST call this function. No logic duplication allowed.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

# Imports from our modules
from ingestion.parse_reactions import parse_chemical_formula, composition_to_formula
from ingestion.precursor_extraction import infer_precursors
from ingestion.paper_scraper import PaperScraper
from ingestion.embed_and_store import PaperEmbedder
from rag.retriever import MaterialsRetriever
from rag.llama_agent import LlamaAgent
from crystal.composition_editing import CompositionEditor
from crystal.cif_generation import CIFGenerator
from prediction.alignff_predict import AlignFFPredictor
from prediction.matgl_predict import MatGLPredictor
from synthesis.hazard_detection import HazardDetector
from synthesis.synthesis_generator import SynthesisGenerator


@dataclass
class PipelineResult:
    """Complete results from materials pipeline."""
    # Input
    original_formula: str
    final_formula: str
    composition: Dict[str, float]
    substitutions: Optional[Dict[str, str]]
    
    # Precursors
    precursors: List[str]
    
    # Literature
    retrieved_papers: List[Dict]
    
    # CIF
    cif_content: Optional[str]
    cif_metadata: Optional[Dict]
    
    # Properties
    predicted_properties: Optional[Dict]
    property_method: Optional[str]
    
    # Synthesis
    synthesis_protocol: Optional[str]
    hazards_detected: List[Dict]
    
    # Status
    success: bool
    errors: List[str]
    warnings: List[str]


class MaterialsPipeline:
    """
    THE ONLY materials discovery pipeline.
    
    This is the single source of truth for all materials operations.
    Colab and Streamlit must both use this class - no exceptions.
    """
    
    def __init__(
        self,
        llama_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        qdrant_path: str = "./qdrant_storage",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_4bit: bool = True
    ):
        """
        Initialize pipeline components.
        
        Args:
            llama_model_name: Model identifier (Qwen2.5, Mistral, Phi-3 recommended)
            qdrant_path: Path for Qdrant storage
            embedding_model: SentenceTransformer model name
            use_4bit: Use 4-bit quantization (recommended for GPU)
        """
        print("="*80)
        print("INITIALIZING MATERIALS DISCOVERY PIPELINE")
        print("="*80)
        
        # Initialize components
        print("\n[1/8] Loading embedding model...")
        self.embedder = PaperEmbedder(
            model_name=embedding_model,
            qdrant_path=qdrant_path
        )
        
        print("[2/8] Initializing retriever...")
        self.retriever = MaterialsRetriever(self.embedder)
        
        print("[3/8] Loading Llama model (this may take a few minutes)...")
        self.llama_agent = LlamaAgent(
            model_name=llama_model_name,
            load_in_4bit=use_4bit
        )
        
        print("[4/8] Initializing composition editor...")
        self.composition_editor = CompositionEditor()
        
        print("[5/8] Initializing CIF generator...")
        self.cif_generator = CIFGenerator()
        
        print("[6/8] Initializing property predictors...")
        self.alignff_predictor = AlignFFPredictor()
        try:
            self.matgl_predictor = MatGLPredictor()
        except:
            print("  MatGL not available - will use AlignFF fallback only")
            self.matgl_predictor = None
        
        print("[7/8] Initializing hazard detector...")
        self.hazard_detector = HazardDetector()
        
        print("[8/8] Initializing synthesis generator...")
        self.synthesis_generator = SynthesisGenerator(
            self.llama_agent,
            self.hazard_detector
        )
        
        # Load sample reactions into vector database
        print("[9/9] Populating vector database with sample reactions...")
        self._load_sample_reactions_to_db()
        
        print("\n" + "="*80)
        print("PIPELINE READY")
        print("="*80 + "\n")
    
    def populate_database_from_reactions(self, force_reload: bool = False):
        """
        Manually populate vector database from reaction.csv.
        
        Args:
            force_reload: If True, reload even if database has entries
        
        Call this if automatic loading during init failed.
        """
        print("\n" + "="*80)
        print("MANUAL DATABASE POPULATION")
        print("="*80 + "\n")
        
        # Check current status
        stats = self.embedder.get_collection_stats()
        current_count = stats.get('points_count', 0)
        
        if current_count > 0 and not force_reload:
            print(f"Database already has {current_count} entries.")
            print(f"Set force_reload=True to reload anyway.")
            return current_count
        
        print(f"Current database entries: {current_count}")
        print(f"Starting paper scraping...")
        
        # Force reload by temporarily clearing the check
        self._load_sample_reactions_to_db()
        
        # Check final status
        final_stats = self.embedder.get_collection_stats()
        final_count = final_stats.get('points_count', 0)
        
        print("\n" + "="*80)
        print(f"Database now has {final_count} entries")
        print("="*80 + "\n")
        
        return final_count
    
    def check_database_status(self):
        """Check and display vector database status."""
        stats = self.embedder.get_collection_stats()
        
        print("\n" + "="*80)
        print("VECTOR DATABASE STATUS")
        print("="*80)
        print(f"Collection: {stats.get('name', 'Unknown')}")
        print(f"Total papers: {stats.get('points_count', 0)}")
        print(f"Vector count: {stats.get('vectors_count', 0)}")
        
        if stats.get('points_count', 0) == 0:
            print("\n⚠ WARNING: Database is EMPTY!")
            print("Literature retrieval will NOT work.")
            print("\nTo populate the database, run:")
            print("  pipeline.populate_database_from_reactions()")
        else:
            print(f"\n✓ Database is populated and ready for queries")
        
        print("="*80 + "\n")
        
        return stats
    
    def run_materials_pipeline(
        self,
        composition: str,
        substitutions: Optional[Dict[str, str]] = None,
        generate_cif: bool = True,
        predict_properties: bool = True,
        generate_synthesis: bool = True,
        scrape_papers: bool = False,
        retrieve_top_k: int = 5
    ) -> PipelineResult:
        """
        THE ONLY PIPELINE FUNCTION.
        
        This function performs ALL materials discovery operations.
        Both Colab and Streamlit MUST call this function.
        
        Args:
            composition: Chemical formula (e.g., "BaTiO3")
            substitutions: Element substitutions (e.g., {"Ba": "Sr"})
            generate_cif: Whether to generate CIF file
            predict_properties: Whether to predict properties
            generate_synthesis: Whether to generate synthesis protocol
            scrape_papers: Whether to scrape new papers (slow)
            retrieve_top_k: Number of papers to retrieve
            
        Returns:
            PipelineResult with all outputs
        """
        errors = []
        warnings = []
        
        print(f"\n{'='*80}")
        print(f"RUNNING PIPELINE FOR: {composition}")
        if substitutions:
            print(f"SUBSTITUTIONS: {substitutions}")
        print('='*80)
        
        # ============================================================
        # STEP 1: Parse composition
        # ============================================================
        print("\n[STEP 1/7] Parsing composition...")
        try:
            parsed_composition = parse_chemical_formula(composition)
            original_formula = composition
            print(f"  Parsed: {parsed_composition}")
        except Exception as e:
            errors.append(f"Failed to parse composition: {e}")
            return self._error_result(original_formula, errors)
        
        # ============================================================
        # STEP 2: Apply substitutions
        # ============================================================
        print("\n[STEP 2/7] Applying substitutions...")
        if substitutions:
            try:
                final_formula, final_composition = self.composition_editor.apply_substitution(
                    original_formula,
                    substitutions
                )
                print(f"  Original: {original_formula}")
                print(f"  Final: {final_formula}")
                
                # Validate substitution
                validation = self.composition_editor.validate_substitution(
                    original_formula,
                    substitutions
                )
                for warning in validation.get('warnings', []):
                    warnings.append(warning)
                    print(f"  ⚠ {warning}")
                    
            except Exception as e:
                errors.append(f"Substitution failed: {e}")
                return self._error_result(original_formula, errors)
        else:
            final_formula = original_formula
            final_composition = parsed_composition
            print("  No substitutions")
        
        # ============================================================
        # STEP 3: Infer precursors
        # ============================================================
        print("\n[STEP 3/7] Inferring precursors...")
        try:
            precursors = infer_precursors(final_composition)
            print(f"  Precursors: {', '.join(precursors)}")
        except Exception as e:
            warnings.append(f"Precursor inference failed: {e}")
            precursors = []
        
        # ============================================================
        # STEP 4: Retrieve/scrape literature
        # ============================================================
        print("\n[STEP 4/7] Retrieving literature...")
        retrieved_papers = []
        
        if scrape_papers:
            print("  Scraping new papers (this may take a minute)...")
            try:
                scraper = PaperScraper()
                papers = scraper.scrape_papers_for_material(
                    final_formula,
                    precursors,
                    max_per_source=retrieve_top_k
                )
                
                # Add to vector database
                for paper in papers:
                    self.embedder.add_paper(
                        paper_id=paper.pmid or paper.doi or f"scraped_{len(retrieved_papers)}",
                        title=paper.title,
                        abstract=paper.abstract,
                        metadata={
                            'doi': paper.doi,
                            'pmid': paper.pmid,
                            'url': paper.url,
                            'material': final_formula
                        }
                    )
                
                print(f"  Scraped and indexed {len(papers)} papers")
            except Exception as e:
                warnings.append(f"Paper scraping failed: {e}")
        
        # Retrieve from vector database
        try:
            retrieved_papers = self.retriever.retrieve_for_synthesis(
                final_formula,
                precursors,
                top_k=retrieve_top_k
            )
            print(f"  Retrieved {len(retrieved_papers)} papers from database")
            
            if retrieved_papers and len(retrieved_papers) > 0:
                print(f"  Top result: {retrieved_papers[0]['title'][:60]}...")
                print(f"  Score: {retrieved_papers[0]['score']:.3f}")
            else:
                print(f"  ⚠ No papers found in database - synthesis will use general templates only")
                warnings.append("No literature found in vector database")
        except Exception as e:
            print(f"  ✗ Literature retrieval error: {e}")
            warnings.append(f"Literature retrieval failed: {e}")
            retrieved_papers = []  # Ensure it's at least an empty list
        
        # ============================================================
        # STEP 5: Generate CIF
        # ============================================================
        cif_content = None
        cif_metadata = None
        
        if generate_cif:
            print("\n[STEP 5/7] Generating CIF file...")
            try:
                # Get reference DOI from top paper if available
                ref_doi = None
                if retrieved_papers and len(retrieved_papers) > 0:
                    ref_doi = retrieved_papers[0].get('doi')
                
                cif_content = self.cif_generator.generate_cif(
                    final_formula,
                    final_composition,
                    reference_doi=ref_doi
                )
                
                cif_metadata = {
                    'formula': final_formula,
                    'reference': ref_doi or "Generated structure",
                    'source': 'Crystal-Text-LLM-inspired generation'
                }
                
                print("  ✓ CIF generated successfully")
                print(f"  Lines: {len(cif_content.split(chr(10)))}")
                
            except Exception as e:
                errors.append(f"CIF generation failed: {e}")
                warnings.append("Continuing without CIF")
        else:
            print("\n[STEP 5/7] Skipping CIF generation")
        
        # ============================================================
        # STEP 6: Predict properties
        # ============================================================
        predicted_properties = None
        property_method = None
        
        if predict_properties:
            print("\n[STEP 6/7] Predicting properties...")
            try:
                # Try MatGL first if available and we have CIF
                if self.matgl_predictor and cif_content:
                    try:
                        predicted_properties = self.matgl_predictor.predict_from_cif(
                            cif_content
                        )
                        property_method = "MatGL M3GNet"
                        print("  Using MatGL M3GNet")
                    except:
                        predicted_properties = None
                
                # Fallback to AlignFF
                if predicted_properties is None:
                    predicted_properties = self.alignff_predictor.predict(
                        final_formula,
                        final_composition,
                        cif_content
                    )
                    property_method = "AlignFF (composition-based)"
                    print("  Using AlignFF fallback")
                
                print("  ✓ Properties predicted")
                if 'band_gap_eV' in predicted_properties:
                    print(f"  Band gap: {predicted_properties['band_gap_eV']:.2f} eV")
                if 'formation_energy_eV_atom' in predicted_properties:
                    print(f"  Formation energy: {predicted_properties['formation_energy_eV_atom']:.3f} eV/atom")
                    
            except Exception as e:
                errors.append(f"Property prediction failed: {e}")
                warnings.append("Continuing without properties")
        else:
            print("\n[STEP 6/7] Skipping property prediction")
        
        # ============================================================
        # STEP 7: Generate synthesis protocol with MANDATORY safety
        # ============================================================
        synthesis_protocol = None
        hazards_detected = []
        
        if generate_synthesis:
            print("\n[STEP 7/7] Generating synthesis protocol with safety...")
            try:
                # Detect hazards first
                hazards = self.hazard_detector.detect_hazards(
                    final_composition,
                    precursors
                )
                hazards_detected = [
                    {
                        'element': h.element,
                        'type': h.hazard_type,
                        'severity': h.severity,
                        'description': h.description
                    }
                    for h in hazards
                ]
                
                print(f"  Hazards detected: {len(hazards)}")
                for h in hazards:
                    print(f"    • {h.element}: {h.severity.upper()} - {h.hazard_type}")
                
                # Generate protocol with ENFORCED safety
                synthesis_protocol = self.synthesis_generator.generate_synthesis_protocol(
                    final_formula,
                    final_composition,
                    precursors,
                    retrieved_papers,
                    substitutions
                )
                
                print("  ✓ Synthesis protocol generated with mandatory safety")
                print(f"  Protocol length: {len(synthesis_protocol)} characters")
                
            except Exception as e:
                errors.append(f"Synthesis generation failed: {e}")
                warnings.append("Cannot provide synthesis without safety validation")
        else:
            print("\n[STEP 7/7] Skipping synthesis generation")
        
        # ============================================================
        # COMPLETE
        # ============================================================
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        
        result = PipelineResult(
            original_formula=original_formula,
            final_formula=final_formula,
            composition=final_composition,
            substitutions=substitutions,
            precursors=precursors,
            retrieved_papers=retrieved_papers,
            cif_content=cif_content,
            cif_metadata=cif_metadata,
            predicted_properties=predicted_properties,
            property_method=property_method,
            synthesis_protocol=synthesis_protocol,
            hazards_detected=hazards_detected,
            success=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
        return result
    
    def _error_result(self, formula: str, errors: List[str]) -> PipelineResult:
        """Create error result."""
        return PipelineResult(
            original_formula=formula,
            final_formula=formula,
            composition={},
            substitutions=None,
            precursors=[],
            retrieved_papers=[],
            cif_content=None,
            cif_metadata=None,
            predicted_properties=None,
            property_method=None,
            synthesis_protocol=None,
            hazards_detected=[],
            success=False,
            errors=errors,
            warnings=[]
        )
    
    def _load_sample_reactions_to_db(self):
        """Scrape and load real papers from PubMed/arXiv for reactions in reaction.csv."""
        import os
        import pandas as pd
        from ingestion.paper_scraper import PaperScraper
        
        # Check if reaction.csv exists
        csv_path = 'reaction.csv'
        if not os.path.exists(csv_path):
            # Try parent directory
            csv_path = '../reaction.csv'
            if not os.path.exists(csv_path):
                print("  ⚠ reaction.csv not found, skipping sample data load")
                return
        
        try:
            # Load reactions
            df = pd.read_csv(csv_path)
            
            # Check if database is already populated (skip auto-load if yes)
            stats = self.embedder.get_collection_stats()
            existing_count = stats.get('points_count', 0)
            
            if existing_count > 0:
                print(f"  Database already has {existing_count} entries")
                print(f"  Skipping auto-scraping to save time")
                print(f"  To reload: pipeline.populate_database_from_reactions(force_reload=True)")
                return
            
            # Initialize scraper
            try:
                scraper = PaperScraper(email="research@materialsrag.ai")
            except Exception as e:
                print(f"  ✗ Failed to initialize scraper: {e}")
                return
            
            paper_count = 0
            materials_processed = 0
            
            print(f"  Starting to scrape papers for {len(df)} materials...")
            print(f"  This will take several minutes due to API rate limits...")
            print(f"  (NCBI allows max 3 requests/second, ~0.34s delay per request)")
            print(f"  You can monitor progress below...\n")
            
            # Scrape papers for ALL reactions
            for idx, row in df.iterrows():
                composition = row.get('composition', '')
                precursors_str = row.get('precursors', '')
                
                # Parse precursors list
                precursors_list = [p.strip() for p in precursors_str.split(',')]
                
                print(f"  [{idx+1}/{len(df)}] Scraping papers for {composition}...")
                
                try:
                    # Scrape papers for this material and its precursors
                    papers = scraper.scrape_papers_for_material(
                        material=composition,
                        precursors=precursors_list,
                        max_per_source=2  # Limit to 2 papers per source to avoid long wait
                    )
                    
                    # Add scraped papers to database
                    for paper in papers:
                        if paper.abstract:  # Only add if has abstract
                            self.embedder.add_paper(
                                paper_id=paper.pmid or paper.doi or f"scraped_{idx}_{paper_count}",
                                title=paper.title,
                                abstract=paper.abstract,
                                full_text=paper.full_text,
                                metadata={
                                    'source': 'pubmed' if paper.pmid else 'arxiv',
                                    'doi': paper.doi,
                                    'pmid': paper.pmid,
                                    'authors': ', '.join(paper.authors[:5]),
                                    'journal': paper.journal,
                                    'year': paper.year,
                                    'url': paper.url,
                                    'related_material': composition,
                                    'related_precursors': precursors_str
                                }
                            )
                            paper_count += 1
                    
                    print(f"    ✓ Added {len(papers)} papers")
                    materials_processed += 1
                    
                except Exception as e:
                    print(f"    ✗ Scraping error: {str(e)[:100]}")
                    continue
            
            print(f"\n  {'='*70}")
            
            if paper_count > 0:
                print(f"  ✓ Successfully scraped and indexed {paper_count} real papers from PubMed/arXiv")
                print(f"  ✓ Processed {materials_processed}/{len(df)} materials from reaction.csv")
                print(f"  ✓ Average: {paper_count/max(materials_processed, 1):.1f} papers per material")
            else:
                print(f"  ✗ WARNING: No papers were scraped - vector database is EMPTY")
                print(f"  ✗ Literature retrieval will NOT work")
                print(f"  ✗ Synthesis protocols will use generic templates only")
                print(f"  ")
                print(f"  Possible causes:")
                print(f"    - Network connection issues")
                print(f"    - PubMed API rate limiting or blocking")
                print(f"    - Materials not found in scientific literature")
                print(f"    - Invalid chemical formulas")
            
        except Exception as e:
            print(f"  ⚠ Failed to load and scrape papers: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'vector_db_stats': self.embedder.get_collection_stats(),
            'models_loaded': {
                'llama': True,
                'embeddings': True,
                'matgl': self.matgl_predictor is not None
            }
        }


def save_result_to_json(result: PipelineResult, output_path: str):
    """Save pipeline result to JSON file."""
    result_dict = asdict(result)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    # Test the pipeline
    print("Testing Materials Pipeline...")
    
    # Initialize (will load models)
    pipeline = MaterialsPipeline(use_4bit=True)
    
    # Test run
    result = pipeline.run_materials_pipeline(
        composition="BaTiO3",
        substitutions={"Ti": "Zr"},
        generate_cif=True,
        predict_properties=True,
        generate_synthesis=True,
        scrape_papers=False,
        retrieve_top_k=3
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Success: {result.success}")
    print(f"Final formula: {result.final_formula}")
    print(f"Precursors: {', '.join(result.precursors)}")
    print(f"Papers retrieved: {len(result.retrieved_papers)}")
    print(f"CIF generated: {'Yes' if result.cif_content else 'No'}")
    print(f"Properties predicted: {'Yes' if result.predicted_properties else 'No'}")
    print(f"Synthesis protocol: {'Yes' if result.synthesis_protocol else 'No'}")
    
    if result.warnings:
        print(f"\nWarnings: {len(result.warnings)}")
        for w in result.warnings:
            print(f"  ⚠ {w}")
    
    if result.errors:
        print(f"\nErrors: {len(result.errors)}")
        for e in result.errors:
            print(f"  ✗ {e}")
