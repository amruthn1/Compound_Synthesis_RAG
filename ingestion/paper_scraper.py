"""Scrape scientific papers from open-access sources."""

import requests
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import xml.etree.ElementTree as ET


@dataclass
class Paper:
    """Represents a scientific paper."""
    title: str
    abstract: str
    doi: str = ""
    pmid: str = ""
    authors: List[str] = None
    journal: str = ""
    year: str = ""
    full_text: str = ""
    url: str = ""
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []


class PaperScraper:
    """Scrape papers from PubMed Central and arXiv."""
    
    def __init__(self, email: str = "user@example.com"):
        """
        Initialize scraper.
        
        Args:
            email: Email for PubMed API (required by NCBI guidelines)
        """
        self.email = email
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.pmc_base = "https://www.ncbi.nlm.nih.gov/pmc/articles/"
        self.arxiv_base = "http://export.arxiv.org/api/query"
        
    def search_pubmed(self, query: str, max_results: int = 5) -> List[str]:
        """
        Search PubMed for paper IDs.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of PubMed IDs
        """
        search_url = f"{self.pubmed_base}esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'email': self.email
        }
        
        try:
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                return data['esearchresult']['idlist']
            return []
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []
    
    def fetch_pubmed_paper(self, pmid: str) -> Optional[Paper]:
        """
        Fetch paper details from PubMed.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            Paper object or None
        """
        fetch_url = f"{self.pubmed_base}efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml',
            'email': self.email
        }
        
        try:
            time.sleep(0.34)  # NCBI rate limit: max 3 requests/second
            response = requests.get(fetch_url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            article = root.find('.//PubmedArticle')
            
            if article is None:
                return None
            
            # Extract metadata
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            abstract_elem = article.find('.//Abstract/AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Authors
            authors = []
            for author in article.findall('.//Author'):
                lastname = author.find('LastName')
                forename = author.find('ForeName')
                if lastname is not None and forename is not None:
                    authors.append(f"{forename.text} {lastname.text}")
            
            # Journal and year
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            year_elem = article.find('.//PubDate/Year')
            year = year_elem.text if year_elem is not None else ""
            
            # DOI
            doi = ""
            for article_id in article.findall('.//ArticleId'):
                if article_id.get('IdType') == 'doi':
                    doi = article_id.text
                    break
            
            return Paper(
                title=title,
                abstract=abstract,
                pmid=pmid,
                doi=doi,
                authors=authors,
                journal=journal,
                year=year,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            )
            
        except Exception as e:
            print(f"Error fetching PMID {pmid}: {e}")
            return None
    
    def search_arxiv(self, query: str, max_results: int = 5) -> List[Paper]:
        """
        Search arXiv for papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of Paper objects
        """
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results
        }
        
        try:
            response = requests.get(self.arxiv_base, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse Atom XML
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            papers = []
            for entry in root.findall('atom:entry', ns):
                title_elem = entry.find('atom:title', ns)
                title = title_elem.text.strip() if title_elem is not None else ""
                
                summary_elem = entry.find('atom:summary', ns)
                abstract = summary_elem.text.strip() if summary_elem is not None else ""
                
                # Authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None:
                        authors.append(name_elem.text)
                
                # URL
                url_elem = entry.find('atom:id', ns)
                url = url_elem.text if url_elem is not None else ""
                
                # Published date
                published_elem = entry.find('atom:published', ns)
                year = ""
                if published_elem is not None:
                    year = published_elem.text[:4]
                
                papers.append(Paper(
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    journal="arXiv",
                    year=year,
                    url=url
                ))
            
            return papers
            
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []
    
    def scrape_papers_for_material(
        self,
        material: str,
        precursors: List[str],
        max_per_source: int = 3
    ) -> List[Paper]:
        """
        Scrape papers related to a material and its precursors.
        
        Args:
            material: Material formula
            precursors: List of precursor formulas
            max_per_source: Maximum papers per source
            
        Returns:
            List of Paper objects
        """
        all_papers = []
        
        # Search for main material
        queries = [
            f"{material} synthesis",
            f"{material} solid state reaction",
        ]
        
        # Add individual precursor queries for better coverage
        for precursor in precursors[:3]:  # Search for each precursor
            queries.append(f"{precursor} synthesis preparation")
            queries.append(f"{material} {precursor}")
        
        # Search PubMed with diverse queries
        for query in queries[:5]:  # Limit to 5 queries to avoid rate limits
            pmids = self.search_pubmed(query, max_results=max_per_source)
            for pmid in pmids[:max_per_source]:
                paper = self.fetch_pubmed_paper(pmid)
                if paper and paper.abstract:
                    all_papers.append(paper)
                    if len(all_papers) >= max_per_source * 3:  # Get more papers
                        break
            if len(all_papers) >= max_per_source * 3:
                break
        
        # Search arXiv for additional papers
        arxiv_papers = self.search_arxiv(f"{material} synthesis", max_results=max_per_source)
        all_papers.extend(arxiv_papers[:max_per_source])
        
        return all_papers


def save_papers(papers: List[Paper], output_dir: str):
    """Save scraped papers to JSON files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, paper in enumerate(papers):
        filename = f"paper_{i:03d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'title': paper.title,
                'abstract': paper.abstract,
                'doi': paper.doi,
                'pmid': paper.pmid,
                'authors': paper.authors,
                'journal': paper.journal,
                'year': paper.year,
                'url': paper.url,
                'full_text': paper.full_text
            }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Test scraping
    scraper = PaperScraper()
    papers = scraper.scrape_papers_for_material("BaTiO3", ["BaCO3", "TiO2"], max_per_source=2)
    
    print(f"Found {len(papers)} papers:")
    for paper in papers:
        print(f"\nTitle: {paper.title}")
        print(f"Authors: {', '.join(paper.authors[:3])}")
        print(f"Abstract: {paper.abstract[:200]}...")
