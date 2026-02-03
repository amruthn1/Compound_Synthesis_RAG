"""Retriever for materials science RAG system."""

from typing import List, Dict, Optional
from ingestion.embed_and_store import PaperEmbedder


class MaterialsRetriever:
    """Retrieve relevant papers for materials queries."""
    
    def __init__(
        self,
        embedder: PaperEmbedder,
        default_top_k: int = 5,
        score_threshold: float = 0.2
    ):
        """
        Initialize retriever.
        
        Args:
            embedder: PaperEmbedder instance with loaded papers
            default_top_k: Default number of results to return
            score_threshold: Minimum similarity score (0.2 = filters noise, allows precursor papers)
        """
        self.embedder = embedder
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold
    
    def retrieve_for_material(
        self,
        material: str,
        query_type: str = "synthesis",
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve papers for a specific material.
        
        Args:
            material: Material formula
            query_type: Type of query (synthesis, properties, characterization)
            top_k: Number of results (uses default if None)
            
        Returns:
            List of relevant papers with scores
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # Construct query
        query = f"{material} {query_type}"
        
        # Retrieve
        results = self.embedder.search(
            query=query,
            top_k=k,
            score_threshold=self.score_threshold
        )
        
        return results
    
    def retrieve_for_precursors(
        self,
        precursors: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve papers mentioning specific precursors.
        
        Args:
            precursors: List of precursor formulas
            top_k: Number of results
            
        Returns:
            List of relevant papers
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # Construct query from precursors
        query = " ".join(precursors) + " solid state synthesis"
        
        results = self.embedder.search(
            query=query,
            top_k=k,
            score_threshold=self.score_threshold
        )
        
        return results
    
    def retrieve_for_synthesis(
        self,
        material: str,
        precursors: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve papers for synthesis of a material from specific precursors.
        
        Args:
            material: Target material formula
            precursors: List of precursor formulas
            top_k: Number of results
            
        Returns:
            List of relevant papers
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # Construct comprehensive query focusing on precursors AND material elements
        # Extract base elements from precursors for better matching
        import re
        
        precursor_elements = []
        for prec in precursors[:3]:
            if prec:
                # Extract element symbols (pattern: Capital letter followed by optional lowercase)
                elements = re.findall(r'[A-Z][a-z]?', prec)
                # Take first 1-2 elements (usually the metal/main component)
                precursor_elements.extend(elements[:2])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_elements = [x for x in precursor_elements if not (x in seen or seen.add(x))]
        
        # Build query with material, elements, precursors, and relevant keywords
        precursor_str = " ".join(precursors[:3])
        element_str = " ".join(unique_elements[:4])  # Limit to 4 elements
        
        # Multi-part query for better retrieval
        query = f"{material} {element_str} synthesis {precursor_str} solid-state reaction ceramic oxide preparation"
        
        # Debug: print query being used
        print(f"  📝 Query: '{query}'")
        print(f"  🔍 Searching with top_k={k}, score_threshold={self.score_threshold}")
        
        results = self.embedder.search(
            query=query,
            top_k=k,
            score_threshold=self.score_threshold
        )
        
        # If threshold is too restrictive and we got no results, try with lower threshold
        if len(results) == 0 and self.score_threshold > 0.1:
            print(f"  ⚠ No results with threshold {self.score_threshold}, retrying with 0.15...")
            results = self.embedder.search(
                query=query,
                top_k=k,
                score_threshold=0.15  # More permissive fallback
            )
        
        print(f"  📊 Results found: {len(results)}")
        if results:
            print(f"  📄 Top result score: {results[0].get('score', 0):.4f}")
            if len(results) > 1:
                print(f"  📄 Lowest result score: {results[-1].get('score', 0):.4f}")
        
        return results
    
    def retrieve_for_properties(
        self,
        material: str,
        properties: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve papers about specific properties of a material.
        
        Args:
            material: Material formula
            properties: List of properties (e.g., ['band gap', 'conductivity'])
            top_k: Number of results
            
        Returns:
            List of relevant papers
        """
        k = top_k if top_k is not None else self.default_top_k
        
        # Construct query
        prop_str = " ".join(properties)
        query = f"{material} {prop_str} characterization properties"
        
        results = self.embedder.search(
            query=query,
            top_k=k,
            score_threshold=self.score_threshold
        )
        
        return results
    
    def format_context(self, papers: List[Dict], max_papers: int = 5) -> str:
        """
        Format retrieved papers into context for LLM.
        
        Args:
            papers: List of paper dictionaries
            max_papers: Maximum number of papers to include
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, paper in enumerate(papers[:max_papers], 1):
            context_part = f"[{i}] {paper['title']}\n"
            
            # Add metadata
            if paper.get('doi'):
                context_part += f"DOI: {paper['doi']}\n"
            if paper.get('pmid'):
                context_part += f"PMID: {paper['pmid']}\n"
            if paper.get('year'):
                context_part += f"Year: {paper['year']}\n"
            
            context_part += f"Relevance Score: {paper['score']:.3f}\n"
            context_part += f"\nAbstract:\n{paper['abstract']}\n"
            
            if paper.get('full_text'):
                context_part += f"\nExcerpt:\n{paper['full_text'][:500]}...\n"
            
            context_parts.append(context_part)
        
        return "\n" + "="*80 + "\n".join(context_parts)


if __name__ == "__main__":
    # Test retriever
    from ingestion.embed_and_store import PaperEmbedder
    
    embedder = PaperEmbedder(qdrant_path="./qdrant_storage")
    retriever = MaterialsRetriever(embedder)
    
    # Test retrieval
    results = retriever.retrieve_for_material("BaTiO3", "synthesis")
    print(f"Found {len(results)} results for BaTiO3 synthesis")
    
    context = retriever.format_context(results)
    print(context[:500])
