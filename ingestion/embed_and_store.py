"""Embed papers and store in Qdrant vector database."""

import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import json


class PaperEmbedder:
    """Embed papers and store in vector database."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        qdrant_path: Optional[str] = None,
        collection_name: str = "materials_papers"
    ):
        """
        Initialize embedder and vector database.
        
        Args:
            model_name: SentenceTransformer model name
            qdrant_path: Path for local Qdrant storage (None for in-memory)
            collection_name: Name of Qdrant collection
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize Qdrant client
        if qdrant_path and qdrant_path != ":memory:":
            # Use persistent storage on disk
            self.client = QdrantClient(path=qdrant_path)
        else:
            # Use in-memory storage (for cloud deployments)
            self.client = QdrantClient(":memory:")
        
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create Qdrant collection."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                print(f"  Creating new Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"  ✓ Collection created")
            else:
                print(f"  ✓ Collection already exists: {self.collection_name}")
        except Exception as e:
            print(f"  Warning: Could not initialize collection: {e}")
            # Continue anyway - will create on first add

    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def add_paper(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        full_text: str = "",
        metadata: Optional[Dict] = None
    ):
        """
        Add a paper to the vector database.
        
        Args:
            paper_id: Unique paper identifier
            title: Paper title
            abstract: Paper abstract
            full_text: Full text (if available)
            metadata: Additional metadata
        """
        # Combine text for embedding
        combined_text = f"{title}\n\n{abstract}"
        if full_text:
            combined_text += f"\n\n{full_text[:5000]}"  # Limit full text
        
        # Generate embedding
        embedding = self.embed_text(combined_text)
        
        # Prepare payload
        payload = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "full_text": full_text[:5000] if full_text else "",
        }
        if metadata:
            payload.update(metadata)
        
        # Store in Qdrant
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=payload
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
    
    def add_papers_from_directory(self, papers_dir: str):
        """
        Load and embed all papers from a directory.
        
        Args:
            papers_dir: Directory containing paper JSON files
        """
        if not os.path.exists(papers_dir):
            print(f"Papers directory not found: {papers_dir}")
            return
        
        paper_files = [f for f in os.listdir(papers_dir) if f.endswith('.json')]
        
        for filename in paper_files:
            filepath = os.path.join(papers_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            # Add to vector database
            self.add_paper(
                paper_id=paper_data.get('pmid') or paper_data.get('doi') or filename,
                title=paper_data.get('title', ''),
                abstract=paper_data.get('abstract', ''),
                full_text=paper_data.get('full_text', ''),
                metadata={
                    'doi': paper_data.get('doi', ''),
                    'pmid': paper_data.get('pmid', ''),
                    'authors': paper_data.get('authors', []),
                    'journal': paper_data.get('journal', ''),
                    'year': paper_data.get('year', ''),
                    'url': paper_data.get('url', '')
                }
            )
        
        print(f"Added {len(paper_files)} papers to vector database")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for relevant papers.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of results with papers and scores
        """
        # Embed query
        query_embedding = self.embed_text(query)
        
        # Search - try both new and old API
        try:
            # New Qdrant API (v1.8+)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            ).points
        except (AttributeError, TypeError):
            # Old Qdrant API (v1.7 and earlier)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'score': result.score,
                'paper_id': result.payload.get('paper_id'),
                'title': result.payload.get('title'),
                'abstract': result.payload.get('abstract'),
                'doi': result.payload.get('doi'),
                'pmid': result.payload.get('pmid'),
                'url': result.payload.get('url'),
                'full_text': result.payload.get('full_text', '')
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Handle different Qdrant API versions
            try:
                # Newer Qdrant API (v1.8+) - nested in config
                if hasattr(collection_info, 'points_count'):
                    points_count = collection_info.points_count
                elif hasattr(collection_info, 'vectors_count'):
                    points_count = collection_info.vectors_count
                else:
                    # Fallback: count via scroll
                    points, _ = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=1
                    )
                    points_count = len(points)
            except AttributeError:
                # Oldest fallback
                points_count = 0
            
            return {
                'name': self.collection_name,
                'points_count': points_count,
                'vectors_count': points_count  # Same as points_count in most cases
            }
        except Exception as e:
            print(f"Warning: Could not get collection stats: {e}")
            return {
                'name': self.collection_name,
                'points_count': 0,
                'vectors_count': 0
            }


if __name__ == "__main__":
    # Test embedding and storage
    embedder = PaperEmbedder(qdrant_path="./qdrant_storage")
    
    # Add a test paper
    embedder.add_paper(
        paper_id="test_001",
        title="Synthesis of BaTiO3 ceramics",
        abstract="We report the synthesis of BaTiO3 using solid-state reaction...",
        metadata={'doi': '10.1234/test'}
    )
    
    # Search
    results = embedder.search("BaTiO3 synthesis methods", top_k=3)
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  Score: {r['score']:.3f} - {r['title']}")
