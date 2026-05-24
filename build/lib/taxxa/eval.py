"""
Eval Harness — pytest-compatible evaluation against question_bank.json.

- Loads question_bank.json
- Runs each question through the agent pipeline
- Compares generated answer against key_facts using semantic similarity
- Produces a per-question pass/fail table via rich
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .schema import EvalResult, AgentAnswer


# ---------------------------------------------------------------------------
# Question bank loader
# ---------------------------------------------------------------------------


def load_question_bank(path: str | Path) -> list[dict]:
    """Load question_bank.json and return entries as a list of dicts."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("entries", [])


# ---------------------------------------------------------------------------
# Key fact evaluation using semantic similarity
# ---------------------------------------------------------------------------


class FactEvaluator:
    """Evaluate key fact coverage using semantic similarity.

    Uses sentence-transformers to compute similarity between
    expected key facts and the generated answer. A fact is
    considered "covered" if similarity >= threshold.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        self.threshold = similarity_threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("BAAI/bge-m3")
            except ImportError:
                print("[WARN] sentence-transformers not available, using exact matching")
                self._model = None
        return self._model

    def evaluate(
        self,
        expected_answer: str,
        generated_answer: str,
        key_facts: list[str],
    ) -> tuple[list[str], list[str], float]:
        """Evaluate key fact coverage.

        Returns:
            covered: list of key facts that were covered
            missed: list of key facts that were missed
            overall_similarity: semantic similarity between full answers
        """
        if not generated_answer:
            return [], list(key_facts), 0.0

        if self.model:
            return self._semantic_eval(expected_answer, generated_answer, key_facts)
        else:
            return self._exact_eval(expected_answer, generated_answer, key_facts)

    def _semantic_eval(
        self,
        expected_answer: str,
        generated_answer: str,
        key_facts: list[str],
    ) -> tuple[list[str], list[str], float]:
        """Use sentence embeddings to check fact coverage."""
        import numpy as np

        # Embed the generated answer once
        gen_embedding = self.model.encode(
            [generated_answer], normalize_embeddings=True
        )[0]

        # Compute overall similarity with expected answer
        exp_embedding = self.model.encode(
            [expected_answer], normalize_embeddings=True
        )[0]
        overall_sim = float(np.dot(gen_embedding, exp_embedding))

        # Check each key fact
        covered = []
        missed = []

        if key_facts:
            fact_embeddings = self.model.encode(
                key_facts, normalize_embeddings=True
            )
            for i, fact in enumerate(key_facts):
                sim = float(np.dot(gen_embedding, fact_embeddings[i]))
                if sim >= self.threshold:
                    covered.append(fact)
                else:
                    missed.append(fact)

        # If no key_facts provided, fall back to overall similarity only
        if not key_facts:
            if overall_sim >= 0.6:
                covered.append(expected_answer[:100])
            else:
                missed.append(expected_answer[:100])

        return covered, missed, overall_sim

    def _exact_eval(
        self,
        expected_answer: str,
        generated_answer: str,
        key_facts: list[str],
    ) -> tuple[list[str], list[str], float]:
        """Fallback: substring matching for key facts."""
        gen_lower = generated_answer.lower()
        covered = []
        missed = []

        for fact in key_facts:
            # Check if key tokens from the fact appear in the generated answer
            fact_lower = fact.lower()
            # Count how many words match
            fact_words = set(fact_lower.split())
            gen_words = set(gen_lower.split())
            overlap = fact_words & gen_words
            if fact_words and len(overlap) / len(fact_words) >= 0.5:
                covered.append(fact)
            else:
                missed.append(fact)

        # Overall similarity: Jaccard
        exp_words = set(expected_answer.lower().split())
        if exp_words | gen_words:
            overall_sim = len(exp_words & gen_words) / len(exp_words | gen_words)
        else:
            overall_sim = 0.0

        if not key_facts:
            if overall_sim >= 0.3:
                covered.append(expected_answer[:100])
            else:
                missed.append(expected_answer[:100])

        return covered, missed, overall_sim


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Runs the full evaluation pipeline over the question bank.

    Usage:
        runner = EvalRunner(agent)
        results = runner.run_all("data/question_bank.json")
        runner.print_table(results)
    """

    def __init__(
        self,
        agent,  # TaxxaAgent instance
        evaluator: Optional[FactEvaluator] = None,
    ):
        self.agent = agent
        self.evaluator = evaluator or FactEvaluator()

    def run_all(self, question_bank_path: str | Path) -> list[EvalResult]:
        """Run all questions through the agent and evaluate."""
        entries = load_question_bank(question_bank_path)
        results = []

        for i, entry in enumerate(entries):
            qid = entry.get("id", f"Q{i}")
            tier = entry.get("tier", entry.get("difficulty", ""))
            question = entry.get("question", "")
            expected_answer = entry.get("answer", "")
            key_facts = entry.get("answer_key_facts", [])

            print(f"\n[{i+1}/{len(entries)}] {qid} ({tier})")
            print(f"  Q: {question[:120]}...")

            # Run agent
            agent_answer: AgentAnswer = self.agent.answer(
                question_id=qid,
                question=question,
                tier=str(tier),
            )

            # Evaluate
            covered, missed, similarity = self.evaluator.evaluate(
                expected_answer,
                agent_answer.answer,
                key_facts,
            )

            passed = len(missed) == 0 and similarity >= 0.5
            citations_present = len(agent_answer.citations) > 0

            result = EvalResult(
                question_id=qid,
                tier=str(tier),
                question=question,
                expected_answer=expected_answer,
                generated_answer=agent_answer.answer,
                key_facts_covered=covered,
                key_facts_missed=missed,
                semantic_similarity=similarity,
                passed=passed,
                citations_present=citations_present,
            )
            results.append(result)

            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} | sim={similarity:.3f} | covered={len(covered)}/{len(key_facts) if key_facts else 1} | cited={citations_present}")

        return results

    def print_table(self, results: list[EvalResult]) -> None:
        """Print a formatted pass/fail table using rich."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Taxxa QA Evaluation Results")

            table.add_column("ID", style="cyan", width=8)
            table.add_column("Tier", style="magenta", width=12)
            table.add_column("Similarity", style="yellow", width=10)
            table.add_column("Facts", style="green", width=10)
            table.add_column("Cited", style="blue", width=8)
            table.add_column("Result", style="bold", width=10)

            for r in results:
                sim_str = f"{r.semantic_similarity:.3f}"
                facts_denom = len(r.key_facts_covered) + len(r.key_facts_missed)
                facts_str = f"{len(r.key_facts_covered)}/{facts_denom}" if facts_denom > 0 else "N/A"
                cited_str = "✓" if r.citations_present else "✗"
                result_str = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"

                table.add_row(
                    r.question_id,
                    r.tier,
                    sim_str,
                    facts_str,
                    cited_str,
                    result_str,
                )

            console.print(table)

            # Summary
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            avg_sim = sum(r.semantic_similarity for r in results) / total if total > 0 else 0

            console.print(f"\n[bold]Summary:[/bold] {passed}/{total} passed ({100*passed/total:.1f}%)")
            console.print(f"[bold]Average similarity:[/bold] {avg_sim:.3f}")

            # By tier
            tiers: dict[str, list[EvalResult]] = {}
            for r in results:
                t = r.tier or "unknown"
                if t not in tiers:
                    tiers[t] = []
                tiers[t].append(r)

            for tier_name, tier_results in sorted(tiers.items()):
                tier_pass = sum(1 for r in tier_results if r.passed)
                tier_total = len(tier_results)
                console.print(f"  {tier_name}: {tier_pass}/{tier_total} passed")

        except ImportError:
            # Fallback without rich — plain text table
            print("\n" + "=" * 60)
            print("Taxxa QA Evaluation Results")
            print("=" * 60)
            print(f"{'ID':8s} {'Tier':12s} {'Sim':8s} {'Facts':10s} {'Cited':6s} {'Result':8s}")
            print("-" * 60)

            for r in results:
                sim_str = f"{r.semantic_similarity:.3f}"
                facts_denom = len(r.key_facts_covered) + len(r.key_facts_missed)
                facts_str = f"{len(r.key_facts_covered)}/{facts_denom}" if facts_denom > 0 else "N/A"
                cited_str = "yes" if r.citations_present else "no"
                result_str = "PASS" if r.passed else "FAIL"

                print(f"{r.question_id:8s} {r.tier:12s} {sim_str:8s} {facts_str:10s} {cited_str:6s} {result_str:8s}")

            # Summary
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            avg_sim = sum(r.semantic_similarity for r in results) / total if total > 0 else 0

            print("-" * 60)
            print(f"Summary: {passed}/{total} passed ({100*passed/total:.1f}%)")
            print(f"Average similarity: {avg_sim:.3f}")
