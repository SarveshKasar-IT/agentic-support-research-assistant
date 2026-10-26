import json

from langchain_core.messages import HumanMessage
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent.graph import graph


EVAL_FILE = "evaluation/eval_cases.json"
RESULT_FILE = "evaluation/evaluation_results.json"


def load_cases():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tool_names(messages):
    tools = []

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            for call in tool_calls:
                name = call.get("name")

                if name:
                    tools.append(name)

    return tools


def get_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")

                if text:
                    parts.append(text)

            elif isinstance(item, str):
                parts.append(item)

        return "\n".join(parts)

    return str(content)


def run_case(case):
    thread_id = f"eval-{case['id']}"

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=case["input"])
            ]
        },
        {
            "configurable": {
                "thread_id": thread_id
            }
        },
    )

    messages = result["messages"]

    return {
        "tools": get_tool_names(messages),
        "answer": get_text(messages[-1].content),
    }


def run_memory_case(case):
    thread_id = f"eval-{case['id']}"

    graph.invoke(
        {
            "messages": [
                HumanMessage(content=case["conversation"][0])
            ]
        },
        {
            "configurable": {
                "thread_id": thread_id
            }
        },
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=case["conversation"][1])
            ]
        },
        {
            "configurable": {
                "thread_id": thread_id
            }
        },
    )

    messages = result["messages"]

    return {
        "tools": get_tool_names(messages),
        "answer": get_text(messages[-1].content),
    }


def evaluate_case(case, result):
    category = case["category"]

    tools = result["tools"]
    answer = result["answer"].lower()

    # --------------------------------------------------
    # Calculator evaluation
    # --------------------------------------------------

    if category == "calculator":
        expected_tool = case["expected_tool"]

        tool_pass = expected_tool in tools

        expected_result = case["expected_result"]

        result_pass = expected_result in answer

        return tool_pass and result_pass

    # --------------------------------------------------
    # Web search evaluation
    # --------------------------------------------------

    if category == "web_search":
        return "web_search" in tools

    # --------------------------------------------------
    # RAG evaluation
    # --------------------------------------------------

    if category == "rag":
        if "reset your password" in answer:
            return True

        if "spam" in answer or "junk" in answer:
            return True

        return False

    # --------------------------------------------------
    # Memory evaluation
    # --------------------------------------------------

    if category == "memory":
        if case["id"] == "MEM-01":
            return "alex" in answer

        if case["id"] == "MEM-02":
            return (
                "password" in answer
                or "login" in answer
            )

        return False

    # --------------------------------------------------
    # Fallback evaluation
    # --------------------------------------------------

    if category == "fallback":
        fallback_phrases = [
            "does not contain",
            "not enough information",
            "don't have enough information",
            "cannot find",
            "no information",
            "not available",
            "i don't know",
        ]

        return any(
            phrase in answer
            for phrase in fallback_phrases
        )

    return False


def main():

    cases = load_cases()

    results = []

    passed = 0
    failed = 0
    errors = 0

    print("=" * 70)
    print("AGENT EVALUATION")
    print("=" * 70)

    for case in cases:

        print(f"\nRunning {case['id']}...")

        try:

            if case["category"] == "memory":
                result = run_memory_case(case)
            else:
                result = run_case(case)

            passed_case = evaluate_case(
                case,
                result,
            )

            status = "PASS" if passed_case else "FAIL"

            if passed_case:
                passed += 1
            else:
                failed += 1

            evaluation_result = {
                "id": case["id"],
                "category": case["category"],
                "status": status,
                "tools_used": result["tools"],
                "answer": result["answer"],
            }

            results.append(evaluation_result)

            print("Tools:", result["tools"])
            print("Status:", status)
            print("Answer:", result["answer"][:300])

        except Exception as e:

            errors += 1

            evaluation_result = {
                "id": case["id"],
                "category": case["category"],
                "status": "ERROR",
                "error": str(e),
            }

            results.append(evaluation_result)

            print("Status: ERROR")
            print("Error:", e)

    total = len(cases)

    executed = passed + failed

    if executed > 0:
        score = (passed / executed) * 100
    else:
        score = 0

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total cases : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Errors      : {errors}")
    print(f"Score       : {score:.1f}%")

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "score": score,
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nResults saved to:")
    print(RESULT_FILE)


if __name__ == "__main__":
    main()