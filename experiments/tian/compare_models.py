import json
import time
from pathlib import Path

import requests


MODELS = ["gemma4:e4b", "qwen3:8b"]
OLLAMA_URL = "http://localhost:11434/api/chat"
RUNS_PER_CASE = 2

EVALUATION_DIRECTORY = Path(__file__).parent
TEST_CASES_PATH = EVALUATION_DIRECTORY / "test_cases.json"
RESULTS_DIRECTORY = (
    EVALUATION_DIRECTORY
    / "results_gemma4_e4b_vs_qwen3_8b"
)

SYSTEM_PROMPT = """
You are the query-understanding component of a UK Antarctic Heritage Trust
image retrieval system.

Return only valid JSON in this exact structure:
{
  "intent": "image_search | metadata_filter | unsupported",
  "semantic_query": "string or null",
  "filters": {
    "category": null,
    "date": null,
    "location": null
  },
  "selected_tools": [],
  "needs_clarification": false
}

Available tools:
- semantic_search: searches images by visual or semantic meaning.
- sql_filter: filters images using exact metadata.

Rules:
1. Use semantic_search when the request describes the visual or semantic
   content of an image.
2. Use sql_filter for category, date, or location metadata.
3. Use both tools when the request contains semantic content and metadata.
4. Use metadata_filter only when the request contains metadata conditions
   without semantic image content.
5. If the request is too vague or depends on missing conversation context,
   set needs_clarification to true and selected_tools to [].
6. If the request is unrelated to archive image retrieval, use unsupported.
7. Do not invent filters or information not stated by the user.
8. Return JSON only.
"""


def load_test_cases():
    """Load the 15 test cases and expected answers."""
    with TEST_CASES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(value):
    """Normalize text for case-insensitive comparison."""
    if value is None:
        return ""

    return str(value).lower().strip().rstrip(".")


def compare_filters(actual_filters, expected_filters):
    """Compare category, date and location filters."""
    if not isinstance(actual_filters, dict):
        return False

    for key in ["category", "date", "location"]:
        actual_value = normalize_text(actual_filters.get(key))
        expected_value = normalize_text(expected_filters.get(key))

        if actual_value != expected_value:
            return False

    return True


def compare_tools(actual_tools, expected_tools):
    """Compare selected tools without considering their order."""
    if not isinstance(actual_tools, list):
        return False

    return set(actual_tools) == set(expected_tools)


def compare_semantic_concepts(semantic_query, concept_groups):
    """
    Check whether the semantic query contains at least one accepted
    expression from every required concept group.
    """
    normalized_query = normalize_text(semantic_query)

    if not concept_groups:
        return normalized_query == ""

    if not normalized_query:
        return False

    for accepted_expressions in concept_groups:
        concept_found = any(
            normalize_text(expression) in normalized_query
            for expression in accepted_expressions
        )

        if not concept_found:
            return False

    return True


def evaluate_output(model_output, expected):
    """Evaluate every required part of one model response."""
    if not isinstance(model_output, dict):
        return {
            "complete_success": False,
            "intent_correct": False,
            "semantic_correct": False,
            "filters_correct": False,
            "tools_correct": False,
            "clarification_correct": False
        }

    intent_correct = (
        model_output.get("intent")
        == expected["intent"]
    )

    semantic_correct = compare_semantic_concepts(
        model_output.get("semantic_query"),
        expected["semantic_concepts"]
    )

    filters_correct = compare_filters(
        model_output.get("filters"),
        expected["filters"]
    )

    tools_correct = compare_tools(
        model_output.get("selected_tools"),
        expected["selected_tools"]
    )

    clarification_correct = (
        model_output.get("needs_clarification")
        == expected["needs_clarification"]
    )

    complete_success = all([
        intent_correct,
        semantic_correct,
        filters_correct,
        tools_correct,
        clarification_correct
    ])

    return {
        "complete_success": complete_success,
        "intent_correct": intent_correct,
        "semantic_correct": semantic_correct,
        "filters_correct": filters_correct,
        "tools_correct": tools_correct,
        "clarification_correct": clarification_correct
    }


def create_payload(model, query):
    """Create an Ollama request payload."""
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_ctx": 4096
        },
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }

    # Disable Qwen's thinking mode for a fair JSON task comparison.
    if model.startswith(("qwen", "gemma4")):
        payload["think"] = False

    return payload


def warm_up_model(model):
    """Warm up the model without including the result in the benchmark."""
    print(f"\nWarming up {model}...")

    payload = create_payload(
        model,
        "Find photographs of people inside a tent."
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        print(f"{model} warm-up completed.")

    except requests.RequestException as error:
        print(f"{model} warm-up failed: {error}")


def run_test(model, test_case, run_number):
    """Run one test case once."""
    payload = create_payload(
        model,
        test_case["query"]
    )

    start_time = time.perf_counter()

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        result = response.json()
        response_time = time.perf_counter() - start_time

        raw_output = result["message"]["content"]

        try:
            parsed_output = json.loads(raw_output)
            json_valid = True
        except json.JSONDecodeError:
            parsed_output = None
            json_valid = False

        evaluation = evaluate_output(
            parsed_output,
            test_case["expected"]
        )

        eval_count = result.get("eval_count", 0)
        eval_duration = result.get("eval_duration", 0)

        generation_speed = (
            eval_count / (eval_duration / 1_000_000_000)
            if eval_duration > 0
            else 0
        )

        return {
            "case_id": test_case["id"],
            "case_type": test_case["type"],
            "query": test_case["query"],
            "run": run_number,
            "response_time_seconds": round(response_time, 3),
            "generation_speed_tokens_per_second": round(
                generation_speed,
                2
            ),
            "json_valid": json_valid,
            "task_success": evaluation["complete_success"],
            "evaluation": evaluation,
            "model_output": (
                parsed_output
                if parsed_output is not None
                else raw_output
            )
        }

    except (requests.RequestException, KeyError, ValueError) as error:
        return {
            "case_id": test_case["id"],
            "case_type": test_case["type"],
            "query": test_case["query"],
            "run": run_number,
            "response_time_seconds": round(
                time.perf_counter() - start_time,
                3
            ),
            "generation_speed_tokens_per_second": 0,
            "json_valid": False,
            "task_success": False,
            "error": str(error)
        }


def calculate_model_summary(model, results):
    """Calculate the three main benchmark metrics."""
    total_runs = len(results)

    average_response_time = sum(
        result["response_time_seconds"]
        for result in results
    ) / total_runs

    average_generation_speed = sum(
        result["generation_speed_tokens_per_second"]
        for result in results
    ) / total_runs

    successful_runs = sum(
        result["task_success"]
        for result in results
    )

    success_rate = (
        successful_runs / total_runs
    ) * 100

    return {
        "model": model,
        "number_of_test_cases": len(results) // RUNS_PER_CASE,
        "runs_per_case": RUNS_PER_CASE,
        "total_runs": total_runs,
        "average_response_time_seconds": round(
            average_response_time,
            3
        ),
        "average_generation_speed_tokens_per_second": round(
            average_generation_speed,
            2
        ),
        "complete_task_success_rate_percent": round(
            success_rate,
            2
        ),
        "successful_runs": successful_runs,
        "failed_runs": total_runs - successful_runs
    }


def main():
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    test_cases = load_test_cases()
    all_results = {}
    summaries = []

    print(
        f"Loaded {len(test_cases)} test cases. "
        f"Each case will run {RUNS_PER_CASE} times."
    )

    for model in MODELS:
        warm_up_model(model)
        model_results = []

        total_model_runs = len(test_cases) * RUNS_PER_CASE
        current_run = 0

        for test_case in test_cases:
            for run_number in range(1, RUNS_PER_CASE + 1):
                current_run += 1

                print(
                    f"[{model}] "
                    f"{current_run}/{total_model_runs} - "
                    f"{test_case['id']} "
                    f"run {run_number}/{RUNS_PER_CASE}"
                )

                result = run_test(
                    model,
                    test_case,
                    run_number
                )

                model_results.append(result)

                status = (
                    "PASS"
                    if result["task_success"]
                    else "FAIL"
                )
                print(f"Result: {status}")

        model_summary = calculate_model_summary(
            model,
            model_results
        )

        all_results[model] = {
            "summary": model_summary,
            "individual_results": model_results
        }

        summaries.append(model_summary)

        model_filename = model.replace(":", "_")

        model_output_path = (
            RESULTS_DIRECTORY
            / f"{model_filename}_full_results.json"
        )

        model_output_path.write_text(
            json.dumps(
                all_results[model],
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    comparison = {
        "benchmark_configuration": {
            "models": MODELS,
            "number_of_test_cases": len(test_cases),
            "runs_per_case": RUNS_PER_CASE,
            "total_requests": (
                len(MODELS)
                * len(test_cases)
                * RUNS_PER_CASE
            ),
            "warm_up_included_in_results": False
        },
        "model_summaries": summaries,
        "full_results": all_results
    }

    comparison_path = (
        RESULTS_DIRECTORY
        / "model_comparison_results.json"
    )

    comparison_path.write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nFinal comparison:")

    for summary in summaries:
        print(f"\nModel: {summary['model']}")
        print(
            "Average response time: "
            f"{summary['average_response_time_seconds']} seconds"
        )
        print(
            "Average generation speed: "
            f"{summary['average_generation_speed_tokens_per_second']} "
            "tokens/s"
        )
        print(
            "Complete task success rate: "
            f"{summary['complete_task_success_rate_percent']}%"
        )
        print(
            f"Successful runs: {summary['successful_runs']}/"
            f"{summary['total_runs']}"
        )

    print(f"\nFull comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()
