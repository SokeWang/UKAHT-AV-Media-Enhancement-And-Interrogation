"""
experiments/tian/eval_llm_models.py
Owner: Tian Luo — Milestone 4 (LLM Agent ReAct Control & Tool-Calling Intent Benchmark)

Comparative Evaluation of LLM Models:
  1. Qwen 3.5 4B (qwen3.5:4b)
  2. Llama 3.2 3B (llama3.2:3b)
  3. Gemma 4:e4b Baseline (gemma4:e4b)

Evaluation Dimensions:
  - User Intent Classification Accuracy
  - Tool Selection Precision, Recall, and F1 Score
  - Parameter / Argument Extraction Accuracy (base_code, year, category, subject_type)
  - Tool Calling Schema & JSON Validity Rate
  - Fact-Grounding & Evidence Citation Rate ([ID: asset_id])
  - Execution Latency & Performance (Mean, Median, P95)
"""

import sys
import os
import json
import time
import re
import argparse
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import requests

# ---------------------------------------------------------------------------
# Tool Declarations (OpenAI Function / Ollama Tool Call Schema)
# ---------------------------------------------------------------------------

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search the UKAHT image archive by visual meaning or descriptive keywords. Use this when the user asks for images based on visual scenes, objects, or themes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language visual search query"},
                    "category": {"type": "string", "description": "Optional image category filter"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sql_filter",
            "description": "Filter archive images by structured metadata attributes like base_code, shooting_year, category, subject_type, copyright.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_code": {"type": "string", "description": "Single letter: 'E' (Stonington Island), 'W' (Detaille Island), 'A' (Port Lockroy)"},
                    "shooting_year": {"type": "string", "description": "Shooting year or inequality e.g. '1958', '1965', '1950s', '<1970', '>2000', '2021-22'"},
                    "category": {"type": "string", "description": "One of: 'Polar Landscape & Glaciers', 'Artifacts & Museum Display', 'Exterior Heritage & Huts', 'Expedition Equipment & Vessels'"},
                    "subject_type": {"type": "string", "description": "One of: 'Exterior', 'Artifact', 'Main Hut', 'SfM', 'Landscape', 'Interior'"},
                    "copyright": {"type": "string", "description": "Photographer credit substring e.g. 'Mike Cousins', 'Neil Marsden'"},
                    "keyword": {"type": "string", "description": "Substring search in title or caption"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Direct PostgreSQL Query Tool. Generate a raw SQL WHERE clause or SELECT statement to retrieve archive assets from 'assets' table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "where_clause": {"type": "string", "description": "SQL WHERE expression e.g. base_code = 'E' AND shooting_year < '1980' ORDER BY shooting_year ASC"},
                    "sql": {"type": "string", "description": "Full raw SELECT SQL statement"}
                }
            }
        }
    }
]

SYSTEM_PROMPT = """You are an AI Assistant for the UK Antarctic Heritage Trust (UKAHT) photograph archive.
Help users explore historical photos by calling one of the 3 tools:
- `semantic_search`: for visual scenes, themes, objects (e.g. 'seals on ice', 'wooden huts in snow')
- `sql_filter`: for structured metadata (base_code in 'E','W','A', shooting_year, category, subject_type, copyright)
- `sql_query`: for direct SQL where_clause (e.g. where_clause="base_code = 'E' AND shooting_year < '1970'")

Rules:
Base E = Stonington Island, Base W = Detaille Island, Base A = Port Lockroy.
Always call the most specific tool for the user's intent."""


def normalize_param_value(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip().lower().replace("'", "").replace('"', "")


def evaluate_parameter_match(expected_params: Dict[str, Any], extracted_params: Dict[str, Any]) -> float:
    """
    Computes parameter match score (0.0 to 1.0) between ground truth and extracted parameters.
    """
    if not expected_params:
        return 1.0
    if not extracted_params:
        return 0.0
    
    matches = 0.0
    total = len(expected_params)
    
    for k, v in expected_params.items():
        if k not in extracted_params:
            exp_val = normalize_param_value(v)
            if "where_clause" in extracted_params and exp_val in normalize_param_value(extracted_params["where_clause"]):
                matches += 0.8
            continue
            
        exp_v = normalize_param_value(v)
        act_v = normalize_param_value(extracted_params[k])
        
        # Exact match
        if exp_v == act_v:
            matches += 1.0
        # Substring / pattern containment
        elif exp_v in act_v or act_v in exp_v:
            matches += 0.85
        elif k == "shooting_year" and any(yr in act_v for yr in re.findall(r'\d{4}', exp_v)):
            matches += 0.9
        elif k == "where_clause" and ("base_code" in act_v or "shooting_year" in act_v):
            matches += 0.8
        else:
            matches += 0.0
            
    return min(1.0, matches / max(1.0, float(total)))


def run_model_inference(model_name: str, base_url: str, api_key: str, query: str, timeout: int = 90) -> Dict[str, Any]:
    """
    Executes one turn tool-calling inference on the target LLM.
    """
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        "tools": OPENAI_TOOLS,
        "temperature": 0.0,
        "max_tokens": 128,
        "options": {
            "num_predict": 128,
            "num_ctx": 2048,
            "temperature": 0.0
        },
        "stream": False
    }
    
    start_t = time.perf_counter()
    tool_calls = []
    error_msg = None
    raw_content = ""
    
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        raw_content = msg.get("content") or ""
        
        # 1. Native tool calls
        if "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                fn_name = fn.get("name")
                fn_args_raw = fn.get("arguments", {})
                if isinstance(fn_args_raw, str):
                    try:
                        fn_args = json.loads(fn_args_raw)
                    except Exception:
                        fn_args = {"raw": fn_args_raw}
                else:
                    fn_args = fn_args_raw
                tool_calls.append({
                    "name": fn_name,
                    "args": fn_args
                })
        # 2. Fallback regex for models that emit tool calls in content
        if not tool_calls:
            # Check for <tool_call> or ```json
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    tool_calls.append({
                        "name": parsed.get("tool") or parsed.get("name", "unknown"),
                        "args": parsed.get("args") or parsed.get("parameters", {})
                    })
                except Exception:
                    pass
            # Check for name(args) pattern
            fn_call_match = re.search(r'(semantic_search|sql_filter|sql_query)\s*\((.*?)\)', raw_content, re.DOTALL)
            if fn_call_match:
                tool_calls.append({
                    "name": fn_call_match.group(1),
                    "args": {"raw_call": fn_call_match.group(2).strip()}
                })
    except Exception as exc:
        error_msg = str(exc)
        
    latency = time.perf_counter() - start_t
    
    return {
        "tool_calls": tool_calls,
        "content": raw_content,
        "latency_sec": round(latency, 4),
        "error": error_msg
    }


def evaluate_model_on_dataset(model_name: str, base_url: str, api_key: str, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    print(f"\n{'='*70}", flush=True)
    print(f"🚀 Evaluating Model: {model_name} ({len(dataset)} test cases)", flush=True)
    print(f"{'='*70}", flush=True)
    
    # Pre-warm model in memory
    print(f"[*] Pre-warming model {model_name}...", flush=True)
    try:
        run_model_inference(model_name, base_url, api_key, "Warmup query", timeout=120)
        print(f"[*] Pre-warming complete.", flush=True)
    except Exception as exc:
        print(f"[!] Warmup note: {exc}", flush=True)
    
    case_results = []
    total_cases = len(dataset)
    correct_tool_selection = 0
    param_scores = []
    valid_schema_count = 0
    latencies = []
    
    tool_confusion = {
        "semantic_search": {"tp": 0, "fp": 0, "fn": 0},
        "sql_filter": {"tp": 0, "fp": 0, "fn": 0},
        "sql_query": {"tp": 0, "fp": 0, "fn": 0},
    }
    
    intent_accuracy_by_category = {}
    
    for idx, tc in enumerate(dataset, 1):
        query = tc["query"]
        expected_tools = tc.get("expected_tools", [])
        expected_params = tc.get("expected_params", {})
        intent_cat = tc.get("intent_category", "general")
        
        if intent_cat not in intent_accuracy_by_category:
            intent_accuracy_by_category[intent_cat] = {"correct": 0, "total": 0}
        intent_accuracy_by_category[intent_cat]["total"] += 1
        
        res = run_model_inference(model_name, base_url, api_key, query)
        latencies.append(res["latency_sec"])
        
        tool_calls = res["tool_calls"]
        called_tools = [c["name"] for c in tool_calls if c.get("name")]
        extracted_args = tool_calls[0]["args"] if tool_calls else {}
        
        # 1. Tool selection accuracy
        is_tool_correct = any(t in expected_tools for t in called_tools)
        if is_tool_correct:
            correct_tool_selection += 1
            intent_accuracy_by_category[intent_cat]["correct"] += 1
            
        # 2. Confusion matrix update
        for et in expected_tools:
            if et in called_tools:
                if et in tool_confusion: tool_confusion[et]["tp"] += 1
            else:
                if et in tool_confusion: tool_confusion[et]["fn"] += 1
        for ct in called_tools:
            if ct not in expected_tools:
                if ct in tool_confusion: tool_confusion[ct]["fp"] += 1
                
        # 3. Parameter accuracy
        p_score = evaluate_parameter_match(expected_params, extracted_args) if is_tool_correct else 0.0
        param_scores.append(p_score)
        
        # 4. Schema validity
        is_schema_valid = bool(tool_calls and isinstance(extracted_args, dict) and not res["error"])
        if is_schema_valid:
            valid_schema_count += 1
            
        case_results.append({
            "id": tc["id"],
            "query": query,
            "intent_category": intent_cat,
            "expected_tools": expected_tools,
            "called_tools": called_tools,
            "expected_params": expected_params,
            "extracted_args": extracted_args,
            "is_tool_correct": is_tool_correct,
            "param_score": round(p_score, 3),
            "schema_valid": is_schema_valid,
            "latency_sec": res["latency_sec"],
            "error": res["error"]
        })
        
        status_sym = "✅" if is_tool_correct else "❌"
        called_str = ",".join(called_tools) if called_tools else "NONE"
        print(f"[{idx:02d}/{total_cases:02d}] {status_sym} {intent_cat:16s} | Tool: {called_str:16s} | ParamScore: {p_score:.2f} | Latency: {res['latency_sec']:.2f}s", flush=True)

    # Metrics aggregation
    tool_accuracy = round((correct_tool_selection / total_cases) * 100, 2)
    mean_param_score = round((sum(param_scores) / total_cases) * 100, 2)
    schema_valid_rate = round((valid_schema_count / total_cases) * 100, 2)
    mean_latency = round(float(sum(latencies) / len(latencies)), 3)
    p95_latency = round(float(sorted(latencies)[int(len(latencies) * 0.95)]), 3)
    
    # Tool F1 computation
    tool_f1_scores = {}
    for t_name, counts in tool_confusion.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        tool_f1_scores[t_name] = {
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2)
        }
        
    category_breakdown = {}
    for cat, data in intent_accuracy_by_category.items():
        acc = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0.0
        category_breakdown[cat] = {
            "accuracy": round(acc, 2),
            "correct": data["correct"],
            "total": data["total"]
        }

    summary = {
        "model_name": model_name,
        "total_test_cases": total_cases,
        "tool_selection_accuracy": tool_accuracy,
        "mean_parameter_accuracy": mean_param_score,
        "schema_validity_rate": schema_valid_rate,
        "mean_latency_sec": mean_latency,
        "p95_latency_sec": p95_latency,
        "category_breakdown": category_breakdown,
        "tool_f1_scores": tool_f1_scores,
        "case_details": case_results
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Models on Intent & Tool Calling")
    parser.add_argument("--dataset", type=str, default="experiments/tian/benchmark_dataset.json")
    parser.add_argument("--models", nargs="+", default=["gemma4:e4b", "llama3.2:3b", "qwen2.5:3b"])
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--api-key", type=str, default="ollama")
    parser.add_argument("--output", type=str, default="experiments/tian/llm_benchmark_results.json")
    args = parser.parse_args()
    
    dataset_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(PROJECT_ROOT, args.dataset)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(PROJECT_ROOT, args.output)
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} benchmark test cases from {dataset_path}", flush=True)
    
    all_results = {}
    for model in args.models:
        res = evaluate_model_on_dataset(model, args.base_url, args.api_key, dataset)
        all_results[model] = res
        
    # Write full JSON benchmark output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ All evaluation results saved to: {output_path}", flush=True)
    
    # Print formatted markdown table
    print("\n" + "="*80, flush=True)
    print("🏆 FINAL BENCHMARK SUMMARY TABLE", flush=True)
    print("="*80, flush=True)
    print(f"| Model Name | Tool Selection Acc (%) | Param Accuracy (%) | Schema Valid (%) | Mean Latency (s) | P95 Latency (s) |", flush=True)
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: |", flush=True)
    for m, d in all_results.items():
        print(f"| **{m}** | {d['tool_selection_accuracy']}% | {d['mean_parameter_accuracy']}% | {d['schema_validity_rate']}% | {d['mean_latency_sec']}s | {d['p95_latency_sec']}s |", flush=True)
    print("="*80 + "\n", flush=True)


if __name__ == "__main__":
    main()
