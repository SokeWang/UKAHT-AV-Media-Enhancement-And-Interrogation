"""
experiments/tian/skills/ukaht-polar-navigator/scripts/polar_skill_engine.py
Progressive Disclosure Skill Engine for UKAHT Historical Archive.

Implements a 4-Stage Progressive Disclosure Pipeline:
  Stage 1: Intent Discovery (Trigger matching)
  Stage 2: Entity & Era Normalization (Heuristic resolution via polar ontology)
  Stage 3: Adaptive Execution with Dynamic Fallback (Zero-result recovery)
  Stage 4: Fact-Grounding & Evidence Citation Validation
"""

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONTOLOGY_PATH = os.path.join(SKILL_DIR, "references", "polar_heritage_ontology.json")

class PolarSkillEngine:
    """
    Modular Skill Engine implementing Progressive Disclosure.
    """

    def __init__(self, ontology_path: str = ONTOLOGY_PATH):
        if os.path.exists(ontology_path):
            with open(ontology_path, "r", encoding="utf-8") as f:
                self.ontology = json.load(f)
        else:
            self.ontology = {}

    def phase1_discover_trigger(self, query: str) -> bool:
        """
        Level 1: Minimal trigger check (~0 token overhead).
        """
        keywords = ["base", "photo", "image", "antarctic", "hut", "station", "seal", "year", "archive", "glacier"]
        q_lower = query.lower()
        return any(k in q_lower for k in keywords)

    def phase2_normalize_entities(self, query: str) -> Dict[str, Any]:
        """
        Level 2: Progressive entity & temporal resolution using polar ontology.
        """
        normalized = {
            "query_text": query,
            "base_code": None,
            "shooting_year": None,
            "category": None,
            "subject_type": None,
            "copyright": None,
            "detected_intent": "visual_semantic",
            "semantic_keywords": []
        }
        q_lower = query.lower()

        # 1. Base code resolution
        stations = self.ontology.get("stations", {})
        for st_name, st_info in stations.items():
            for alias in st_info.get("aliases", []):
                if re.search(r'\b' + re.escape(alias) + r'\b', q_lower):
                    normalized["base_code"] = st_info["code"]
                    break
            if normalized["base_code"]:
                break

        # Fallback base detection
        if not normalized["base_code"]:
            if "base e" in q_lower or "stonington" in q_lower: normalized["base_code"] = "E"
            elif "base w" in q_lower or "detaille" in q_lower: normalized["base_code"] = "W"
            elif "base a" in q_lower or "port lockroy" in q_lower: normalized["base_code"] = "A"

        # 2. Year & Decade resolution
        # Inequalities
        before_match = re.search(r'before\s+(\d{4})', q_lower)
        if before_match:
            normalized["shooting_year"] = f"<{before_match.group(1)}"
        after_match = re.search(r'after\s+(\d{4})', q_lower)
        if after_match:
            normalized["shooting_year"] = f">{after_match.group(1)}"
        # Hyphenated years (2021-22)
        hyphen_year = re.search(r'\b(\d{4}-\d{2,4})\b', q_lower)
        if hyphen_year:
            normalized["shooting_year"] = hyphen_year.group(1)
        # Decade
        decade_match = re.search(r'\b(19\d0)s\b', q_lower)
        if decade_match:
            normalized["shooting_year"] = f"{decade_match.group(1)}s"
        # Exact year
        if not normalized["shooting_year"]:
            exact_year = re.search(r'\b(19\d{2}|20\d{2})\b', q_lower)
            if exact_year:
                normalized["shooting_year"] = exact_year.group(1)

        # 3. Photographer resolution
        photographers = self.ontology.get("photographers", ["Mike Cousins", "Neil Marsden", "Gordon MacDonald"])
        for p in photographers:
            if p.lower() in q_lower:
                normalized["copyright"] = p
                break

        # 4. Category and Subject Type
        if "landscape" in q_lower or "glacier" in q_lower:
            normalized["category"] = "Polar Landscape & Glaciers"
        elif "artifact" in q_lower or "museum" in q_lower or "stove" in q_lower:
            normalized["category"] = "Artifacts & Museum Display"
        elif "hut" in q_lower or "exterior" in q_lower:
            normalized["category"] = "Exterior Heritage & Huts"
        elif "equipment" in q_lower or "vessel" in q_lower or "ship" in q_lower:
            normalized["category"] = "Expedition Equipment & Vessels"

        if "interior" in q_lower: normalized["subject_type"] = "Interior"
        elif "exterior" in q_lower: normalized["subject_type"] = "Exterior"
        elif "main hut" in q_lower: normalized["subject_type"] = "Main Hut"
        elif "sfm" in q_lower: normalized["subject_type"] = "SfM"

        # 5. Intent Classification
        if "sql:" in q_lower or "select" in q_lower or "where_clause" in q_lower or "order by" in q_lower:
            normalized["detected_intent"] = "direct_sql"
        elif "how has" in q_lower or "changed from" in q_lower or "compare" in q_lower:
            normalized["detected_intent"] = "complex_timeline"
        elif normalized["base_code"] or normalized["shooting_year"] or normalized["copyright"]:
            normalized["detected_intent"] = "metadata_filter"
        else:
            normalized["detected_intent"] = "visual_semantic"

        return normalized

    def phase3_execute_plan(self, normalized: Dict[str, Any], execute_fn=None) -> Dict[str, Any]:
        """
        Level 3: Adaptive tool execution with dynamic fallback.
        """
        intent = normalized["detected_intent"]
        tool_name = "semantic_search"
        tool_args = {}

        if intent == "direct_sql":
            tool_name = "sql_query"
            where_parts = []
            if normalized["base_code"]: where_parts.append(f"base_code = '{normalized['base_code']}'")
            if normalized["shooting_year"]:
                yr = normalized["shooting_year"]
                if yr.startswith("<"): where_parts.append(f"shooting_year < '{yr[1:]}'")
                elif yr.startswith(">"): where_parts.append(f"shooting_year > '{yr[1:]}'")
                else: where_parts.append(f"shooting_year = '{yr}'")
            where_clause = " AND ".join(where_parts) if where_parts else "1=1"
            if "order by" in normalized["query_text"].lower():
                where_clause += " ORDER BY shooting_year ASC"
            tool_args = {"where_clause": where_clause}

        elif intent in ["metadata_filter", "complex_timeline"]:
            tool_name = "sql_filter"
            if normalized["base_code"]: tool_args["base_code"] = normalized["base_code"]
            if normalized["shooting_year"]: tool_args["shooting_year"] = normalized["shooting_year"]
            if normalized["category"]: tool_args["category"] = normalized["category"]
            if normalized["subject_type"]: tool_args["subject_type"] = normalized["subject_type"]
            if normalized["copyright"]: tool_args["copyright"] = normalized["copyright"]

        else:
            tool_name = "semantic_search"
            tool_args = {"query": normalized["query_text"]}
            if normalized["category"]: tool_args["category"] = normalized["category"]

        # Mock / Real Execution with Dynamic Fallback
        result_count = 6
        fallback_triggered = False

        if execute_fn:
            res = execute_fn(tool_name, tool_args)
            if not res or len(res) == 0:
                # Dynamic Fallback to Semantic Search
                fallback_triggered = True
                tool_name = "semantic_search"
                tool_args = {"query": normalized["query_text"]}
                res = execute_fn(tool_name, tool_args)
            result_count = len(res)

        return {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "fallback_triggered": fallback_triggered,
            "result_count": result_count
        }

    def phase4_validate_citations(self, response_text: str, retrieved_ids: List[str]) -> Tuple[bool, List[str]]:
        """
        Level 4: Fact-grounding and citation validation.
        """
        cited_ids = re.findall(r'\[ID:\s*([\w_]+)\]', response_text)
        valid_citations = [cid for cid in cited_ids if cid in retrieved_ids]
        is_valid = len(valid_citations) > 0 if retrieved_ids else True
        return is_valid, valid_citations
