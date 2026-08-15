"""
RAG Evaluation Harness

Measures retrieval AND generation quality of the RAG chatbot so you can
quantify how good the system is (and prove it in interviews).

Metrics (RAGAS-style):
  Retrieval
    - hit_rate@k  : fraction of questions where at least one retrieved chunk
                    contains enough key phrases from the golden answer.
    - MRR@k       : Mean Reciprocal Rank - average of 1/rank of the first
                    chunk that counts as a hit.
  Generation (LLM-as-judge via Groq, the same model the app uses)
    - faithfulness    : is the answer grounded ONLY in the retrieved context?
                        (0-1, higher = less hallucination)
    - answer_relevancy: does the answer actually address the question?
                        (0-1, higher = more on-topic)

Usage:
    py -3.11 evaluate.py                    # run with hybrid+rerank (default)
    py -3.11 evaluate.py --compare          # also run a dense-only baseline and
                                            # print a side-by-side comparison
    py -3.11 evaluate.py --dataset my_eval.json
"""

import argparse
import json
import os
import re
import sys

from rag_chatbot import RAGChatbot


def load_dataset(path="eval_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def key_phrases(golden_answer):
    """Split the golden answer into distinctive alphanumeric phrases (>=3 chars)."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_.-]{2,}", golden_answer)
    # Drop very common words that add no signal
    stop = {"the", "and", "for", "with", "that", "this", "from", "are",
            "has", "was", "were", "its", "into", "also", "used", "uses",
            "via", "like", "their", "over", "what", "how", "when", "who"}
    return [w for w in words if w.lower() not in stop]


def phrase_overlap(text, phrases):
    """Count how many key phrases appear in the chunk text (case-insensitive)."""
    low = text.lower()
    return sum(1 for p in phrases if p.lower() in low)


def compute_hit_rate_mrr(search_results_list, questions):
    """Compute hit_rate@k and MRR@k from per-question search results."""
    hit_count = 0
    reciprocal_ranks = []
    for results, item in zip(search_results_list, questions):
        phrases = key_phrases(item["golden_answer"])
        min_facts = item.get("min_facts", 1)
        for rank, chunk in enumerate(results):
            if phrase_overlap(chunk["text"], phrases) >= min_facts:
                hit_count += 1
                reciprocal_ranks.append(1.0 / (rank + 1))
                break
        else:
            reciprocal_ranks.append(0.0)
    n = len(questions)
    return {
        "hit_rate@k": hit_count / n if n else 0.0,
        "mrr@k": sum(reciprocal_ranks) / n if n else 0.0,
    }


def build_rag_prompt(query, results):
    context = "\n\n".join(
        f"[Source {i+1}]\n{r['text']}" for i, r in enumerate(results)
    )
    return f"""You are an AI assistant that answers questions using ONLY the provided retrieved document excerpts.

Rules:
- Answer ONLY from the [Source N] excerpts below.
- Do NOT mention sources, citations, "Source 1", or brackets in your answer.
- If the excerpts do not contain the answer, say "The uploaded documents do not contain this information." Do not guess.
- Be concise, use bullet points where helpful.

RETRIEVED DOCUMENTS:
{context}

Question: {query}

Answer:"""


def judge_with_groq(chatbot, prompt):
    """Ask Groq for a structured 0-10 judgement, returns (score01, raw_text)."""
    try:
        reply = chatbot._call_groq(prompt)
    except Exception as e:
        return 0.0, f"JUDGE ERROR: {e}"
    m = re.search(r"(\d{1,2})(?:\s*/\s*10)?", reply)
    if m:
        val = min(10, max(0, int(m.group(1))))
        return val / 10.0, reply
    return 0.0, reply


def judge_faithfulness(chatbot, answer, context):
    prompt = f"""You are an evaluation judge for a RAG system.

On a scale of 0 to 10, how FAITHFUL is the answer to the given context?
Faithful = every claim in the answer is supported by the context. Give 0 if the
answer invents facts not in the context (hallucination), 10 if fully grounded.

CONTEXT:
{context[:4000]}

ANSWER:
{answer}

Respond with ONLY a number 0-10 and nothing else."""
    return judge_with_groq(chatbot, prompt)


def judge_answer_relevancy(chatbot, question, answer):
    prompt = f"""You are an evaluation judge for a RAG system.

On a scale of 0 to 10, how RELEVANT is the answer to the question?
Relevant = the answer directly addresses what was asked. Give 0 if it is
off-topic or unhelpful, 10 if it fully answers the question.

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a number 0-10 and nothing else."""
    return judge_with_groq(chatbot, prompt)


def run_eval(mode="hybrid", dataset_path="eval_dataset.json", top_k=5, max_q=None):
    print(f"=== RAG Evaluation ({mode} mode, top_k={top_k}) ===")
    chatbot = RAGChatbot()
    questions = load_dataset(dataset_path)
    if max_q:
        questions = questions[:max_q]

    retrieval_results = []
    generations = []
    faithfulness_scores = []
    relevancy_scores = []
    raw_judgements = []

    for i, item in enumerate(questions):
        q = item["question"]
        print(f"\n[{i+1}/{len(questions)}] Q: {q}")

        # Hybrid search (or dense-only baseline if requested)
        if mode == "dense":
            results = _dense_only_search(chatbot, q, top_k)
        else:
            results = chatbot.search(q, top_k=top_k)

        retrieval_results.append(results)
        print(f"    retrieved {len(results)} chunks")

        # Generate an answer using the same RAG prompt as the app
        prompt = build_rag_prompt(q, results)
        try:
            answer = chatbot._call_groq(prompt)
        except Exception as e:
            answer = f"GEN ERROR: {e}"
        generations.append(answer)
        print(f"    answer: {answer[:120]}...")

        # LLM-as-judge metrics
        context = "\n\n".join(r["text"] for r in results)
        f_score, f_raw = judge_faithfulness(chatbot, answer, context)
        r_score, r_raw = judge_answer_relevancy(chatbot, q, answer)
        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        raw_judgements.append({"faithfulness": f_raw, "relevancy": r_raw})

    # ---- Aggregate ----
    ret = compute_hit_rate_mrr(retrieval_results, questions)
    n = len(questions)
    gen = {
        "faithfulness": sum(faithfulness_scores) / n if n else 0.0,
        "answer_relevancy": sum(relevancy_scores) / n if n else 0.0,
    }
    report = {
        "mode": mode,
        "top_k": top_k,
        "n_questions": n,
        "retrieval": ret,
        "generation": gen,
    }
    print("\n========== SUMMARY ==========")
    print(f"  mode              : {mode}")
    print(f"  hit_rate@k        : {ret['hit_rate@k']:.3f}")
    print(f"  mrr@k             : {ret['mrr@k']:.3f}")
    print(f"  faithfulness      : {gen['faithfulness']:.3f}")
    print(f"  answer_relevancy  : {gen['answer_relevancy']:.3f}")
    return report, {
        "questions": [i["question"] for i in questions],
        "retrieved_texts": [[r["text"] for r in rs] for rs in retrieval_results],
        "answers": generations,
        "judgements": raw_judgements,
    }


def _dense_only_search(chatbot, query, top_k):
    """Dense-only baseline: skip BM25 fusion + rerank (single Pinecone query)."""
    query_embedding = chatbot.embed_query(query)
    matches = chatbot.pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    ).matches
    return [
        {
            "id": m.id,
            "text": m.metadata.get("text", "") if m.metadata else "",
            "score": float(m.score) if m.score is not None else 0.0,
            "metadata": {k: v for k, v in (m.metadata or {}).items() if k != "text"},
        }
        for m in matches
    ]


def main():
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--dataset", default="eval_dataset.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-q", type=int, default=None,
                        help="Limit to the first N questions (saves API calls)")
    parser.add_argument("--compare", action="store_true",
                        help="Also run a dense-only baseline and print comparison")
    parser.add_argument("--mode", choices=["hybrid", "dense"], default="hybrid")
    args = parser.parse_args()

    report_hybrid, _ = run_eval(
        mode=args.mode, dataset_path=args.dataset,
        top_k=args.top_k, max_q=args.max_q,
    )

    if args.compare:
        print("\n\n=== Running dense-only baseline for comparison ===")
        report_dense, _ = run_eval(
            mode="dense", dataset_path=args.dataset,
            top_k=args.top_k, max_q=args.max_q,
        )
        print("\n========== COMPARISON ==========")
        print(f"{'metric':<20} {'hybrid+rerank':>16} {'dense-only':>16} {'delta':>10}")
        for metric in ("hit_rate@k", "mrr@k", "faithfulness", "answer_relevancy"):
            h = report_hybrid["retrieval"].get(metric) or report_hybrid["generation"].get(metric)
            d = report_dense["retrieval"].get(metric) or report_dense["generation"].get(metric)
            print(f"{metric:<20} {h:>16.3f} {d:>16.3f} {h-d:>+10.3f}")

    # Save the report
    out_name = f"eval_report_{report_hybrid['mode']}.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(report_hybrid, f, indent=2)
    print(f"\nReport saved to {out_name}")


if __name__ == "__main__":
    main()