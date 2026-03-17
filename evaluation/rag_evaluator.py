"""RAG retrieval evaluation helpers with precision/recall/F1 scoring."""

from typing import Dict, List, Optional, Set
import json


class RAGEvaluator:
    """Evaluate retrieval quality for the materials RAG retriever."""

    def __init__(self, retriever):
        """Initialize with a retriever implementing retrieve_for_synthesis."""
        self.retriever = retriever

    @staticmethod
    def _normalize_identifier(raw_id: str) -> str:
        """Normalize user/provided paper identifiers into a common namespace."""
        value = (raw_id or "").strip().lower()
        if not value:
            return ""

        if value.startswith("doi:"):
            return f"doi:{value[4:].strip()}"
        if value.startswith("pmid:"):
            return f"pmid:{value[5:].strip()}"
        if value.startswith("paper:"):
            return f"paper:{value[6:].strip()}"

        if value.isdigit():
            return f"pmid:{value}"

        # DOI-like fallback
        if "/" in value and " " not in value:
            return f"doi:{value}"

        return f"paper:{value}"

    @classmethod
    def _predicted_identifier(cls, paper: Dict) -> str:
        """Pick a canonical identifier from a retrieved paper payload."""
        doi = cls._normalize_identifier(paper.get("doi", ""))
        if doi and doi.startswith("doi:"):
            return doi

        pmid = cls._normalize_identifier(paper.get("pmid", ""))
        if pmid and pmid.startswith("pmid:"):
            return pmid

        paper_id = cls._normalize_identifier(str(paper.get("paper_id", "")))
        if paper_id:
            return paper_id

        title = (paper.get("title", "") or "").strip().lower()
        if title:
            return f"paper:title:{title}"

        return ""

    @classmethod
    def _normalize_relevant_set(cls, relevant_ids: List[str]) -> Set[str]:
        """Normalize benchmark relevant IDs."""
        normalized = set()
        for raw_id in relevant_ids:
            nid = cls._normalize_identifier(raw_id)
            if nid:
                normalized.add(nid)
        return normalized

    def evaluate_retrieval(
        self,
        benchmark_cases: List[Dict],
        default_top_k: int = 5
    ) -> Dict:
        """
        Evaluate retrieval with precision/recall/F1.

        Benchmark case format:
        {
          "material": "BaTiO3",
          "precursors": ["BaCO3", "TiO2"],
          "relevant_ids": ["doi:10.1000/xyz", "pmid:123456"],
          "top_k": 5,
          "name": "optional label"
        }
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0

        per_case = []

        for index, case in enumerate(benchmark_cases):
            material = case.get("material", "")
            precursors = case.get("precursors", [])
            relevant_ids = case.get("relevant_ids", [])
            top_k = int(case.get("top_k", default_top_k))
            case_name = case.get("name", f"case_{index + 1}")

            if not material:
                per_case.append({
                    "name": case_name,
                    "error": "Missing required field: material",
                })
                continue

            retrieved = self.retriever.retrieve_for_synthesis(
                material=material,
                precursors=precursors,
                top_k=top_k,
            )

            predicted_ids = set()
            for paper in retrieved:
                pid = self._predicted_identifier(paper)
                if pid:
                    predicted_ids.add(pid)

            relevant_set = self._normalize_relevant_set(relevant_ids)

            tp = len(predicted_ids.intersection(relevant_set))
            fp = len(predicted_ids - relevant_set)
            fn = len(relevant_set - predicted_ids)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            total_tp += tp
            total_fp += fp
            total_fn += fn

            per_case.append({
                "name": case_name,
                "material": material,
                "top_k": top_k,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "predicted_count": len(predicted_ids),
                "relevant_count": len(relevant_set),
            })

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        scored_cases = [c for c in per_case if "f1" in c]
        macro_precision = sum(c["precision"] for c in scored_cases) / len(scored_cases) if scored_cases else 0.0
        macro_recall = sum(c["recall"] for c in scored_cases) / len(scored_cases) if scored_cases else 0.0
        macro_f1 = sum(c["f1"] for c in scored_cases) / len(scored_cases) if scored_cases else 0.0

        return {
            "summary": {
                "evaluated_cases": len(scored_cases),
                "total_cases": len(benchmark_cases),
                "micro_precision": round(micro_precision, 4),
                "micro_recall": round(micro_recall, 4),
                "micro_f1": round(micro_f1, 4),
                "macro_precision": round(macro_precision, 4),
                "macro_recall": round(macro_recall, 4),
                "macro_f1": round(macro_f1, 4),
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
            },
            "cases": per_case,
        }

    def evaluate_retrieval_from_file(self, benchmark_path: str, default_top_k: int = 5) -> Dict:
        """Load benchmark cases from JSON file and evaluate."""
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark_cases = json.load(f)

        if not isinstance(benchmark_cases, list):
            raise ValueError("Benchmark file must contain a JSON list of cases.")

        return self.evaluate_retrieval(benchmark_cases=benchmark_cases, default_top_k=default_top_k)
