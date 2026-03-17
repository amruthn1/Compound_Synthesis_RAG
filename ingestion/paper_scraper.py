"""Scrape scientific papers from multiple scholarly APIs with one normalized schema."""

import os
import requests
import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import json
import xml.etree.ElementTree as ET


@dataclass
class Paper:
    """Represents a normalized synthesis paper record."""
    title: str
    abstract: str
    doi: str = ""
    pmid: str = ""
    source: str = ""
    source_id: str = ""
    authors: List[str] = None
    journal: str = ""
    year: str = ""
    target_material: str = ""
    precursors: List[str] = None
    full_text: str = ""
    url: str = ""
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.precursors is None:
            self.precursors = []


class PaperScraper:
    """Scrape papers from CrossRef, PubMed, arXiv, DOAJ, Europe PMC, Springer, CORE, and Semantic Scholar."""
    
    def __init__(self, email: str = "user@example.com"):
        """
        Initialize scraper.
        
        Args:
            email: Email for PubMed API (required by NCBI guidelines)
        """
        self.email = email
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.crossref_base = "https://api.crossref.org/works"
        self.arxiv_base = "https://export.arxiv.org/api/query"
        self.doaj_base = "https://doaj.org/api/search/articles"
        self.europepmc_base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        self.springer_base = "https://api.springernature.com/meta/v2/json"
        self.core_base = "https://api.core.ac.uk/v3/search/works"
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1/paper/search"

        self.springer_api_key = os.getenv("SPRINGER_API_KEY", "")
        self.core_api_key = os.getenv("CORE_API_KEY", "")
        self.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

    def _safe_get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: int = 20):
        """HTTP GET with common safety defaults and error propagation."""
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize extra whitespace and newlines."""
        return " ".join((text or "").split())

    def _paper_key(self, paper: Paper) -> str:
        """Create a dedupe key preferring DOI, then PMID, then title+year."""
        if paper.doi:
            return f"doi:{paper.doi.lower()}"
        if paper.pmid:
            return f"pmid:{paper.pmid}"
        return f"title:{paper.title.lower()}|year:{paper.year}"

    def _attach_context(self, papers: List[Paper], material: str, precursors: List[str]) -> List[Paper]:
        """Attach material/precursor context to each normalized paper."""
        for paper in papers:
            paper.target_material = material
            paper.precursors = list(precursors)
        return papers

    def _extract_crossref_abstract(self, raw_abstract: str) -> str:
        """CrossRef abstracts are often JATS fragments; strip simple tags."""
        if not raw_abstract:
            return ""
        cleaned = raw_abstract.replace("<jats:p>", " ").replace("</jats:p>", " ")
        cleaned = cleaned.replace("<jats:title>", " ").replace("</jats:title>", " ")
        return self._normalize_whitespace(cleaned)

    def search_crossref(self, query: str, max_results: int = 5) -> List[Paper]:
        """Search CrossRef and normalize JSON metadata into Paper objects."""
        params = {
            "query": query,
            "rows": max_results,
            "mailto": self.email,
            "select": "DOI,title,author,container-title,published-print,published-online,issued,abstract,URL,type"
        }

        papers: List[Paper] = []
        try:
            response = self._safe_get(self.crossref_base, params=params)
            data = response.json()
            items = data.get("message", {}).get("items", [])

            for item in items:
                title = self._normalize_whitespace(" ".join(item.get("title", [])))
                abstract = self._extract_crossref_abstract(item.get("abstract", ""))
                doi = item.get("DOI", "") or ""
                url = item.get("URL", "") or (f"https://doi.org/{doi}" if doi else "")

                authors = []
                for author in item.get("author", []):
                    given = (author.get("given") or "").strip()
                    family = (author.get("family") or "").strip()
                    full_name = f"{given} {family}".strip()
                    if full_name:
                        authors.append(full_name)

                journal = self._normalize_whitespace(" ".join(item.get("container-title", [])))
                year = ""
                for date_key in ("published-print", "published-online", "issued"):
                    date_parts = item.get(date_key, {}).get("date-parts", [])
                    if date_parts and date_parts[0]:
                        year = str(date_parts[0][0])
                        break

                if title:
                    papers.append(Paper(
                        title=title,
                        abstract=abstract,
                        doi=doi,
                        source="crossref",
                        source_id=doi or url,
                        authors=authors,
                        journal=journal,
                        year=year,
                        url=url
                    ))
        except Exception as e:
            print(f"CrossRef search error: {e}")

        return papers
        
    def search_pubmed(self, query: str, max_results: int = 5) -> List[str]:
        """
        Search PubMed for paper IDs.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of PubMed IDs
        """
        search_url = f"{self.pubmed_base}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'email': self.email
        }
        
        try:
            response = self._safe_get(search_url, params=params)
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
        fetch_url = f"{self.pubmed_base}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml',
            'email': self.email
        }
        
        try:
            time.sleep(0.34)  # NCBI rate limit: max 3 requests/second
            response = self._safe_get(fetch_url, params=params)
            
            # Parse XML
            root = ET.fromstring(response.content)
            article = root.find('.//PubmedArticle')
            
            if article is None:
                return None
            
            # Extract metadata
            title_elem = article.find('.//ArticleTitle')
            title = self._normalize_whitespace(title_elem.text if title_elem is not None else "")
            
            abstract_elem = article.find('.//Abstract/AbstractText')
            abstract = self._normalize_whitespace(abstract_elem.text if abstract_elem is not None else "")
            
            # Authors
            authors = []
            for author in article.findall('.//Author'):
                lastname = author.find('LastName')
                forename = author.find('ForeName')
                if lastname is not None and forename is not None:
                    authors.append(f"{forename.text} {lastname.text}")
            
            # Journal and year
            journal_elem = article.find('.//Journal/Title')
            journal = self._normalize_whitespace(journal_elem.text if journal_elem is not None else "")
            
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
                source="pubmed",
                source_id=pmid,
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
            response = self._safe_get(self.arxiv_base, params=params)
            
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
                    source="arxiv",
                    source_id=url,
                    authors=authors,
                    journal="arXiv",
                    year=year,
                    url=url
                ))
            
            return papers
            
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []

    def search_doaj(self, query: str, max_results: int = 5) -> List[Paper]:
        """Search DOAJ article index."""
        params = {
            "page": 1,
            "pageSize": max_results,
            "q": query,
        }

        papers: List[Paper] = []
        try:
            response = self._safe_get(self.doaj_base, params=params)
            data = response.json()
            for item in data.get("results", []):
                bibjson = item.get("bibjson", {})
                title = self._normalize_whitespace(bibjson.get("title", ""))
                abstract = self._normalize_whitespace(bibjson.get("abstract", ""))
                journal = self._normalize_whitespace(bibjson.get("journal", {}).get("title", ""))
                year = str(bibjson.get("year", "") or "")

                doi = ""
                url = ""
                for identifier in bibjson.get("identifier", []):
                    if (identifier.get("type") or "").lower() == "doi":
                        doi = identifier.get("id", "")
                        break
                links = bibjson.get("link", [])
                if links:
                    url = links[0].get("url", "")
                if not url and doi:
                    url = f"https://doi.org/{doi}"

                authors = [self._normalize_whitespace(a.get("name", "")) for a in bibjson.get("author", []) if a.get("name")]

                if title:
                    papers.append(Paper(
                        title=title,
                        abstract=abstract,
                        doi=doi,
                        source="doaj",
                        source_id=doi or url,
                        authors=authors,
                        journal=journal,
                        year=year,
                        url=url
                    ))
        except Exception as e:
            print(f"DOAJ search error: {e}")

        return papers

    def search_europepmc(self, query: str, max_results: int = 5) -> List[Paper]:
        """Search Europe PMC for abstracts and metadata."""
        params = {
            "query": query,
            "format": "json",
            "pageSize": max_results,
            "resultType": "core"
        }

        papers: List[Paper] = []
        try:
            response = self._safe_get(self.europepmc_base, params=params)
            data = response.json()
            results = data.get("resultList", {}).get("result", [])

            for item in results:
                title = self._normalize_whitespace(item.get("title", ""))
                abstract = self._normalize_whitespace(item.get("abstractText", ""))
                doi = item.get("doi", "") or ""
                pmid = item.get("pmid", "") or ""
                journal = self._normalize_whitespace(item.get("journalTitle", ""))
                year = str(item.get("pubYear", "") or "")
                authors = [self._normalize_whitespace(name) for name in (item.get("authorString", "") or "").split(",") if name.strip()]

                url = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
                article_url = ""
                if url:
                    article_url = url[0].get("url", "")
                if not article_url:
                    if pmid:
                        article_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    elif doi:
                        article_url = f"https://doi.org/{doi}"

                if title:
                    papers.append(Paper(
                        title=title,
                        abstract=abstract,
                        doi=doi,
                        pmid=pmid,
                        source="europepmc",
                        source_id=pmid or doi or article_url,
                        authors=authors,
                        journal=journal,
                        year=year,
                        url=article_url
                    ))
        except Exception as e:
            print(f"Europe PMC search error: {e}")

        return papers

    def search_springer(self, query: str, max_results: int = 5) -> List[Paper]:
        """Search Springer Nature metadata API. Requires SPRINGER_API_KEY."""
        if not self.springer_api_key:
            return []

        params = {
            "q": query,
            "p": max_results,
            "api_key": self.springer_api_key,
        }

        papers: List[Paper] = []
        try:
            response = self._safe_get(self.springer_base, params=params)
            data = response.json()
            for item in data.get("records", []):
                title = self._normalize_whitespace(item.get("title", ""))
                abstract = self._normalize_whitespace(item.get("abstract", ""))
                doi = item.get("doi", "") or ""
                journal = self._normalize_whitespace(item.get("publicationName", ""))
                year = str(item.get("publicationDate", "") or "")[:4]
                authors = [self._normalize_whitespace(a.get("creator", "")) for a in item.get("creators", []) if a.get("creator")]

                url = ""
                for link in item.get("url", []):
                    candidate = link.get("value", "")
                    if candidate:
                        url = candidate
                        break
                if not url and doi:
                    url = f"https://doi.org/{doi}"

                if title:
                    papers.append(Paper(
                        title=title,
                        abstract=abstract,
                        doi=doi,
                        source="springer",
                        source_id=doi or url,
                        authors=authors,
                        journal=journal,
                        year=year,
                        url=url
                    ))
        except Exception as e:
            print(f"Springer search error: {e}")

        return papers

    def search_core(self, query: str, max_results: int = 5) -> List[Paper]:
        """Search CORE works API. Requires CORE_API_KEY."""
        if not self.core_api_key:
            return []

        headers = {"Authorization": f"Bearer {self.core_api_key}"}
        payload = {
            "q": query,
            "limit": max_results,
        }

        papers: List[Paper] = []
        try:
            response = requests.post(self.core_base, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()

            for item in data.get("results", []):
                title = self._normalize_whitespace(item.get("title", ""))
                abstract = self._normalize_whitespace(item.get("abstract", ""))
                doi = item.get("doi", "") or ""
                year = str(item.get("yearPublished", "") or "")
                journal = self._normalize_whitespace(item.get("publisher", ""))
                source_id = str(item.get("id", "") or "")

                authors = []
                for author in item.get("authors", []):
                    name = self._normalize_whitespace(author.get("name", ""))
                    if name:
                        authors.append(name)

                url = item.get("downloadUrl", "") or item.get("sourceFulltextUrls", [""])[0]
                if not url and doi:
                    url = f"https://doi.org/{doi}"

                if title:
                    papers.append(Paper(
                        title=title,
                        abstract=abstract,
                        doi=doi,
                        source="core",
                        source_id=source_id or doi or url,
                        authors=authors,
                        journal=journal,
                        year=year,
                        url=url
                    ))
        except Exception as e:
            print(f"CORE search error: {e}")

        return papers

    def search_semantic_scholar(self, query: str, max_results: int = 5) -> List[Paper]:
        """Search Semantic Scholar Graph API."""
        headers = {}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key

        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,year,authors,externalIds,journal,url"
        }

        papers: List[Paper] = []
        try:
            response = self._safe_get(self.semantic_scholar_base, params=params, headers=headers)
            data = response.json()

            for item in data.get("data", []):
                title = self._normalize_whitespace(item.get("title", ""))
                abstract = self._normalize_whitespace(item.get("abstract", ""))
                year = str(item.get("year", "") or "")
                doi = (item.get("externalIds", {}) or {}).get("DOI", "") or ""
                journal = self._normalize_whitespace((item.get("journal", {}) or {}).get("name", ""))
                url = item.get("url", "") or (f"https://doi.org/{doi}" if doi else "")
                source_id = (item.get("externalIds", {}) or {}).get("CorpusId", "") or url
                authors = [self._normalize_whitespace(a.get("name", "")) for a in item.get("authors", []) if a.get("name")]

                if title:
                    papers.append(Paper(
                        title=title,
                        abstract=abstract,
                        doi=doi,
                        source="semantic_scholar",
                        source_id=str(source_id),
                        authors=authors,
                        journal=journal,
                        year=year,
                        url=url
                    ))
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")

        return papers

    def search_pubmed_papers(self, query: str, max_results: int = 5) -> List[Paper]:
        """Search PubMed and fetch full metadata records by PMID."""
        pmids = self.search_pubmed(query, max_results=max_results)
        papers: List[Paper] = []
        for pmid in pmids[:max_results]:
            paper = self.fetch_pubmed_paper(pmid)
            if paper:
                papers.append(paper)
        return papers

    def _run_source(self, fn: Callable[[str, int], List[Paper]], query: str, max_results: int) -> List[Paper]:
        """Run one source function safely."""
        try:
            return fn(query, max_results)
        except Exception as e:
            print(f"Source execution error for query '{query}': {e}")
            return []

    def _get_enabled_source_order(self) -> List[Callable[[str, int], List[Paper]]]:
        """Build source order while skipping APIs that require missing keys."""
        source_order: List[Callable[[str, int], List[Paper]]] = [
            self.search_crossref,
            self.search_pubmed_papers,
            self.search_arxiv,
            self.search_doaj,
            self.search_europepmc,
        ]

        # Optional keyed sources are only enabled when credentials exist.
        if self.springer_api_key:
            source_order.append(self.search_springer)
        if self.core_api_key:
            source_order.append(self.search_core)

        # Semantic Scholar is usable with or without key; keep enabled.
        source_order.append(self.search_semantic_scholar)
        return source_order
    
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
        all_papers: List[Paper] = []
        seen_keys = set()
        
        # Search for main material
        queries = [
            f"{material} synthesis",
            f"{material} solid state reaction",
        ]
        
        # Add individual precursor queries for better coverage
        for precursor in precursors[:3]:  # Search for each precursor
            queries.append(f"{precursor} synthesis preparation")
            queries.append(f"{material} {precursor}")
        
        # CrossRef is primary; missing-key sources are skipped automatically.
        source_order = self._get_enabled_source_order()

        max_total = max_per_source * len(source_order)

        for query in queries[:5]:
            for source_fn in source_order:
                source_papers = self._run_source(source_fn, query, max_per_source)
                source_papers = self._attach_context(source_papers, material, precursors)

                for paper in source_papers:
                    if not paper.title:
                        continue
                    key = self._paper_key(paper)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    all_papers.append(paper)

                    if len(all_papers) >= max_total:
                        return all_papers

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
                'source': paper.source,
                'source_id': paper.source_id,
                'authors': paper.authors,
                'journal': paper.journal,
                'year': paper.year,
                'target_material': paper.target_material,
                'precursors': paper.precursors,
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
