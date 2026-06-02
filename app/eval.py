import json
import os
from app.query import query_documents

TEST_QUESTIONS = [
    {
        "question": "What are the key risk management practices for fund management companies?",
        "ground_truth": "FMCs should establish governance structures, risk management frameworks, conflicts of interest management, and ongoing monitoring of investments."
    },
    {
        "question": "What governance structure should FMCs have?",
        "ground_truth": "FMCs should have Board and Senior Management oversight with committees across portfolio management, dealing, operations, risk management, compliance and legal functions."
    },
    {
        "question": "What is required under Regulation 13B for fund management companies?",
        "ground_truth": "FMCs must put in place a risk management framework to identify, address and monitor the risks associated with customer assets that they manage."
    },
    {
        "question": "How should FMCs handle conflicts of interest?",
        "ground_truth": "FMCs should identify actual, potential or perceived conflicts of interest, implement effective controls and segregation of duties, and disclose conflicts to customers where appropriate."
    },
    {
        "question": "What should FMCs do for ongoing monitoring of investments?",
        "ground_truth": "FMCs should independently monitor performance and risks at suitable intervals with appropriate metrics and thresholds, ensure deviations from risk parameters are approved, and provide accurate information to investors."
    }
]

def run_simple_eval():
    print("Running FinRAG evaluation...\n")
    results = []

    for i, item in enumerate(TEST_QUESTIONS):
        print(f"Question {i+1}/{len(TEST_QUESTIONS)}: {item['question'][:60]}...")
        
        try:
            response = query_documents(item["question"], top_k=5)
            
            answer = response["answer"]
            sources = response["sources"]
            latency = response["latency_ms"]
            
            # Simple faithfulness check — does answer reference sources?
            has_citations = any(
                f"Source {j+1}" in answer or 
                sources[j]["source"].split(".")[0][:10] in answer
                for j in range(len(sources))
            )
            
            # Simple relevancy check — do key ground truth words appear?
            ground_truth_words = set(item["ground_truth"].lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(ground_truth_words & answer_words)
            relevancy_score = round(min(overlap / max(len(ground_truth_words), 1), 1.0), 3)
            
            # Average rerank score as context precision proxy
            rerank_scores = [s.get("rerank_score", 0) for s in sources if s.get("rerank_score")]
            avg_rerank = round(sum(rerank_scores) / len(rerank_scores), 3) if rerank_scores else 0
            
            result = {
                "question": item["question"],
                "answer_length": len(answer),
                "has_citations": has_citations,
                "relevancy_score": relevancy_score,
                "avg_rerank_score": avg_rerank,
                "latency_ms": latency,
                "num_sources": len(sources)
            }
            
            results.append(result)
            print(f"  ✓ Relevancy: {relevancy_score} | Rerank: {avg_rerank} | Latency: {latency}ms")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({"question": item["question"], "error": str(e)})

    # Summary
    successful = [r for r in results if "error" not in r]
    if successful:
        avg_relevancy = round(sum(r["relevancy_score"] for r in successful) / len(successful), 3)
        avg_rerank = round(sum(r["avg_rerank_score"] for r in successful) / len(successful), 3)
        avg_latency = round(sum(r["latency_ms"] for r in successful) / len(successful))
        all_cited = all(r["has_citations"] for r in successful)

        summary = {
            "total_questions": len(TEST_QUESTIONS),
            "successful": len(successful),
            "avg_relevancy_score": avg_relevancy,
            "avg_rerank_score": avg_rerank,
            "avg_latency_ms": avg_latency,
            "citation_rate": f"{sum(r['has_citations'] for r in successful)}/{len(successful)}"
        }

        print(f"\n{'='*50}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Questions evaluated : {summary['total_questions']}")
        print(f"Successful          : {summary['successful']}")
        print(f"Avg relevancy score : {summary['avg_relevancy_score']}")
        print(f"Avg rerank score    : {summary['avg_rerank_score']}")
        print(f"Avg latency         : {summary['avg_latency_ms']}ms")
        print(f"Citation rate       : {summary['citation_rate']}")
        print(f"{'='*50}")

        # Save results
        with open("eval_results.json", "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"\nFull results saved to eval_results.json")

    return results

if __name__ == "__main__":
    run_simple_eval()