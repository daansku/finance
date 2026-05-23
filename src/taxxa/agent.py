"""
Agent Loop — LangGraph-based decompose→retrieve→draft→verify pipeline.

Models the QA workflow as an explicit state machine with conditional edges.
- Decompose: break the question into sub-questions
- Retrieve: run hybrid retrieval per sub-question
- Draft: generate answer with citations from retrieved passages
- Verify: check each claim against source nodes, re-retrieve if confidence low
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Callable, Literal, TypedDict

from langgraph.graph import StateGraph, END

from .schema import (
    SubQuestion,
    RetrievedPassage,
    AgentAnswer,
)
from .retrieval import HybridRetriever


# ---------------------------------------------------------------------------
# LLM Client abstraction
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin wrapper around OpenAI-compatible LLM APIs.

    Supports OpenRouter, Ollama, Featherless AI, and any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.1,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature

    def generate(self, messages: list[dict], json_mode: bool = False) -> str:
        """Send a chat completion request and return the text response."""
        import httpx

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = httpx.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120,
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                print(f"  [LLM ERROR] status={response.status_code}: {response.text[:200]}")
                return ""
        except Exception as e:
            print(f"  [LLM ERROR] {e}")
            return ""

    def generate_structured(self, messages: list[dict], schema: dict) -> Optional[dict]:
        """Generate a structured JSON response matching the given schema."""
        system_msg = messages[0]["content"] if messages else ""
        full_prompt = (
            f"{system_msg}\n\n"
            "You MUST respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            "Return ONLY the JSON object, no other text."
        )
        response = self.generate(
            [{"role": "user", "content": full_prompt}],
            json_mode=True,
        )
        if not response:
            return None
        try:
            # Handle markdown code fences
            response = re.sub(r"^```(?:json)?\s*", "", response.strip())
            response = re.sub(r"\s*```$", "", response)
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"  [WARN] Failed to parse structured response: {response[:200]}")
            return None


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """State carried through the agent graph."""

    question_id: str
    question: str
    tier: str
    sub_questions: list[dict]
    current_sub_idx: int
    retrieved_passages: list[dict]
    draft_answer: str
    citations: list[dict]
    verification_results: list[dict]
    verified: bool
    confidence: float
    iteration: int
    error: str


# Default initial state
INITIAL_STATE: AgentState = {
    "question_id": "",
    "question": "",
    "tier": "",
    "sub_questions": [],
    "current_sub_idx": 0,
    "retrieved_passages": [],
    "draft_answer": "",
    "citations": [],
    "verification_results": [],
    "verified": False,
    "confidence": 0.0,
    "iteration": 0,
    "error": "",
}


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------


class TaxxaAgent:
    """LangGraph agent for Finnish tax QA.

    State machine:
      decompose → retrieve → [loop sub-questions] → draft → verify → [retry?] → END
    """

    MAX_ITERATIONS = 3

    def __init__(
        self,
        llm: LLMClient,
        retriever: HybridRetriever,
    ):
        self.llm = llm
        self.retriever = retriever
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Build the LangGraph
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Construct the agent state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("decompose", self._decompose)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("draft", self._draft)
        workflow.add_node("verify", self._verify)

        # Edges
        workflow.set_entry_point("decompose")
        workflow.add_edge("decompose", "retrieve")
        workflow.add_edge("retrieve", "draft")
        workflow.add_edge("draft", "verify")

        # Conditional: retry if not verified and under max iterations
        workflow.add_conditional_edges(
            "verify",
            self._should_retry,
            {
                "retrieve": "retrieve",
                "end": END,
            },
        )

        return workflow.compile()

    # ------------------------------------------------------------------
    # Node: Decompose
    # ------------------------------------------------------------------

    def _decompose(self, state: AgentState) -> AgentState:
        """Break the question into sub-questions using structured LLM output."""
        question = state["question"]
        tier = state.get("tier", "")

        prompt = f"""You are a Finnish tax law expert. Break the following question into sub-questions,
each targeting a specific fact that needs to be retrieved. Consider:
- Rates and thresholds
- Effective dates and temporal scope
- Conditions and exceptions
- Cross-references between statutes and guidance

Question: {question}
Difficulty tier: {tier}

Return a JSON object with a 'sub_questions' array. Each sub-question has:
- text: the sub-question string
- reasoning: why this sub-question is needed
- priority: 1 (critical), 2 (important), or 3 (nice-to-have)

Example format:
{{"sub_questions": [{{"text": "What is the capital income tax rate above 30 000 euros?", "reasoning": "Need to find the rate threshold", "priority": 1}}]}}
"""

        schema = {
            "sub_questions": [
                {"text": "string", "reasoning": "string", "priority": 1}
            ]
        }

        result = self.llm.generate_structured(
            [{"role": "system", "content": prompt}],
            schema,
        )

        sub_questions = []
        if result and "sub_questions" in result:
            sub_questions = result["sub_questions"]
        else:
            # Fallback: treat the whole question as one sub-question
            sub_questions = [{"text": question, "reasoning": "Direct question", "priority": 1}]

        state["sub_questions"] = sub_questions
        state["current_sub_idx"] = 0
        return state

    # ------------------------------------------------------------------
    # Node: Retrieve
    # ------------------------------------------------------------------

    def _retrieve(self, state: AgentState) -> AgentState:
        """Run hybrid retrieval for the current sub-question."""
        sub_idx = state.get("current_sub_idx", 0)
        sub_questions = state.get("sub_questions", [])

        if sub_idx >= len(sub_questions):
            return state

        sub_q = sub_questions[sub_idx]
        query = sub_q.get("text", "")

        try:
            passages = self.retriever.retrieve(query)
            state["retrieved_passages"] = [
                {
                    "node_id": p.node_id,
                    "text": p.text,
                    "node_type": p.node_type,
                    "title": p.title,
                    "section_number": p.section_number,
                    "score": p.score,
                }
                for p in passages
            ]
        except Exception as e:
            state["error"] = f"Retrieval error: {e}"
            state["retrieved_passages"] = []

        state["current_sub_idx"] = sub_idx + 1
        return state

    # ------------------------------------------------------------------
    # Node: Draft
    # ------------------------------------------------------------------

    def _draft(self, state: AgentState) -> AgentState:
        """Generate the answer with citations from retrieved passages."""
        question = state["question"]
        passages = state.get("retrieved_passages", [])

        if not passages:
            state["draft_answer"] = (
                "Unable to find relevant source passages for this question. "
                "Please ensure the corpus is indexed."
            )
            state["confidence"] = 0.0
            return state

        # Format passages for the LLM
        passages_text = ""
        for i, p in enumerate(passages[:15]):
            passages_text += (
                f"\n[SOURCE {i+1}] "
                f"Title: {p.get('title', 'N/A')} | "
                f"Section: {p.get('section_number', 'N/A')} | "
                f"Type: {p.get('node_type', 'N/A')}\n"
                f"Text: {p.get('text', '')[:1500]}\n"
            )

        prompt = f"""You are a Finnish tax law expert. Answer the following question using ONLY the provided source passages.
Cite each claim with the source number in brackets [SOURCE N].

Question: {question}

Source passages:
{passages_text}

Instructions:
1. Answer ONLY from the provided sources. Do not use external knowledge.
2. Cite every factual claim with [SOURCE N].
3. If the sources contradict each other, note the conflict.
4. If a key fact isn't in the sources, say so explicitly.
5. Be precise with numbers, rates, dates, and conditions.
6. Structure your answer clearly — start with the direct answer, then provide reasoning.

Return your answer as a JSON object with:
- answer: the full answer text
- citations: array of sources used [{{source_id, title, excerpt}}]
- confidence: 0.0-1.0 rating

Example:
{{"answer": "The rate is 34% for amounts exceeding 30 000 euros [SOURCE 1].", "citations": [{{"source_id": 1, "title": "...", "excerpt": "..."}}], "confidence": 0.9}}
"""

        result = self.llm.generate_structured(
            [{"role": "system", "content": prompt}],
            {
                "answer": "string",
                "citations": [{"source_id": 1, "title": "string", "excerpt": "string"}],
                "confidence": 0.5,
            },
        )

        if result:
            state["draft_answer"] = result.get("answer", "")
            state["citations"] = result.get("citations", [])
            state["confidence"] = result.get("confidence", 0.5)
        else:
            # Fallback: concatenate passages
            state["draft_answer"] = "\n\n".join(
                f"[SOURCE {i+1}] {p.get('text', '')[:500]}"
                for i, p in enumerate(passages[:5])
            )
            state["confidence"] = 0.3

        return state

    # ------------------------------------------------------------------
    # Node: Verify
    # ------------------------------------------------------------------

    def _verify(self, state: AgentState) -> AgentState:
        """Verify each claim in the draft answer against the source passages."""
        draft = state.get("draft_answer", "")
        passages = state.get("retrieved_passages", [])

        if not draft or not passages:
            state["verified"] = False
            state["confidence"] = 0.0
            return state

        # Use LLM to verify claims
        passages_text = "\n\n".join(
            f"[{i+1}] {p.get('text', '')[:800]}"
            for i, p in enumerate(passages[:10])
        )

        verify_prompt = f"""Verify whether the following answer is fully supported by the source passages.
For each claim in the answer, check if it appears in the sources.

Answer to verify:
{draft}

Sources:
{passages_text}

Return JSON:
{{
  "verification_results": [
    {{"claim": "claim text", "supported": true/false, "source_index": 1, "explanation": "..."}}
  ],
  "overall_supported": true/false,
  "confidence": 0.0-1.0,
  "needs_retry": true/false,
  "retry_reasoning": "if needs_retry, what additional information is needed"
}}
"""

        result = self.llm.generate_structured(
            [{"role": "system", "content": verify_prompt}],
            {
                "verification_results": [{"claim": "string", "supported": True, "source_index": 1, "explanation": "string"}],
                "overall_supported": True,
                "confidence": 0.5,
                "needs_retry": False,
                "retry_reasoning": "string",
            },
        )

        if result:
            state["verification_results"] = result.get("verification_results", [])
            state["verified"] = result.get("overall_supported", False)
            state["confidence"] = result.get("confidence", state.get("confidence", 0.5))
        else:
            state["verified"] = state.get("confidence", 0) >= 0.7

        return state

    # ------------------------------------------------------------------
    # Conditional edge: should we retry?
    # ------------------------------------------------------------------

    def _should_retry(self, state: AgentState) -> Literal["retrieve", "end"]:
        """Decide whether to retry retrieval or finish."""
        verified = state.get("verified", False)
        confidence = state.get("confidence", 0.0)
        iteration = state.get("iteration", 0)

        # Retry if not verified and under max iterations
        if not verified and iteration < self.MAX_ITERATIONS and confidence < 0.7:
            state["iteration"] = iteration + 1
            # Check if we have more sub-questions to process
            sub_idx = state.get("current_sub_idx", 0)
            sub_questions = state.get("sub_questions", [])
            if sub_idx < len(sub_questions):
                return "retrieve"
            # If all sub-questions processed but still not verified, retry with refined query
            state["sub_questions"] = state.get("sub_questions", [])
            if state["sub_questions"]:
                state["current_sub_idx"] = 0  # restart from first sub-question
                return "retrieve"

        return "end"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question_id: str,
        question: str,
        tier: str = "",
    ) -> AgentAnswer:
        """Run the full agent pipeline and return an AgentAnswer."""
        initial_state: AgentState = {
            **INITIAL_STATE,
            "question_id": question_id,
            "question": question,
            "tier": tier,
        }

        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            print(f"  [AGENT ERROR] {e}")
            final_state = {
                **initial_state,
                "draft_answer": f"Agent error: {e}",
                "confidence": 0.0,
                "verified": False,
            }

        return AgentAnswer(
            question_id=question_id,
            question=question,
            answer=final_state.get("draft_answer", ""),
            citations=final_state.get("citations", []),
            sub_answers=final_state.get("sub_questions", []),
            confidence=final_state.get("confidence", 0.0),
            verified=final_state.get("verified", False),
        )

    def answer_batch(
        self,
        questions: list[dict],
    ) -> list[AgentAnswer]:
        """Answer a batch of questions."""
        answers = []
        for q in questions:
            print(f"  Processing {q['id']} ({q.get('tier', '')})...")
            answer = self.answer(
                question_id=q.get("id", ""),
                question=q.get("question", ""),
                tier=q.get("tier", q.get("difficulty", "")),
            )
            answers.append(answer)
        return answers