from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import requests
import streamlit as st

MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

GENERATOR_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "required": ["explanation", "mcqs"],
    "properties": {
        "explanation": {"type": "STRING"},
        "mcqs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["question", "options", "answer"],
                "properties": {
                    "question": {"type": "STRING"},
                    "options": {
                        "type": "ARRAY",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "STRING"},
                    },
                    "answer": {
                        "type": "STRING",
                        "enum": ["A", "B", "C", "D"],
                    },
                },
            },
        },
    },
}

REVIEWER_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "required": ["status", "feedback"],
    "properties": {
        "status": {"type": "STRING", "enum": ["pass", "fail"]},
        "feedback": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
}


@dataclass
class GeneratorInput:
    grade: int
    topic: str
    feedback: List[str] = field(default_factory=list)


@dataclass
class MCQ:
    question: str
    options: List[str]
    answer: Literal["A", "B", "C", "D"]


@dataclass
class GeneratorOutput:
    explanation: str
    mcqs: List[MCQ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation": self.explanation,
            "mcqs": [
                {
                    "question": mcq.question,
                    "options": mcq.options,
                    "answer": mcq.answer,
                }
                for mcq in self.mcqs
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratorOutput":
        explanation = str(data.get("explanation", "")).strip()
        raw_mcqs = data.get("mcqs", [])
        if not isinstance(raw_mcqs, list):
            raw_mcqs = []

        mcqs: List[MCQ] = []
        for raw_item in raw_mcqs:
            if not isinstance(raw_item, dict):
                continue

            question = str(raw_item.get("question", "")).strip()
            raw_options = raw_item.get("options", [])
            if not isinstance(raw_options, list):
                raw_options = []
            options = [str(option).strip() for option in raw_options][:4]
            while len(options) < 4:
                options.append(f"Option {len(options) + 1}")

            answer = str(raw_item.get("answer", "A")).strip().upper()
            if answer not in {"A", "B", "C", "D"}:
                answer = "A"

            if question:
                mcqs.append(MCQ(question=question, options=options, answer=answer))

        if not explanation:
            raise ValueError("Generator output is missing explanation.")
        if not mcqs:
            raise ValueError("Generator output is missing MCQs.")

        return cls(explanation=explanation, mcqs=mcqs)


@dataclass
class ReviewerOutput:
    status: Literal["pass", "fail"]
    feedback: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "feedback": self.feedback}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewerOutput":
        status = str(data.get("status", "fail")).strip().lower()
        if status not in {"pass", "fail"}:
            status = "fail"

        raw_feedback = data.get("feedback", [])
        if isinstance(raw_feedback, str):
            raw_feedback = [raw_feedback]
        if not isinstance(raw_feedback, list):
            raw_feedback = []

        feedback = [str(item).strip() for item in raw_feedback if str(item).strip()]
        if status == "fail" and not feedback:
            feedback = ["Reviewer marked fail but gave no details."]
        if status == "pass" and not feedback:
            feedback = ["Content meets age appropriateness, correctness, and clarity checks."]

        return cls(status=status, feedback=feedback)


class GeminiRestClient:
    def __init__(self, api_key: str, model_name: str = MODEL_NAME) -> None:
        self.api_key = api_key
        self.model_name = model_name

    def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        url = f"{GEMINI_API_BASE}/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
        except requests.HTTPError as exc:
            details = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f"Gemini API error: {details}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Network error when calling Gemini API: {exc}") from exc

        response_json = response.json()
        text = self._extract_text(response_json)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini did not return valid JSON.") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Gemini JSON response must be an object.")

        return data

    @staticmethod
    def _extract_text(response_json: Dict[str, Any]) -> str:
        candidates = response_json.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini API returned no candidates.")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            if "text" in part and part["text"]:
                return str(part["text"])

        raise RuntimeError("Gemini API candidate had no text output.")


class GeneratorAgent:
    def __init__(self, client: GeminiRestClient) -> None:
        self.client = client

    def run(self, agent_input: GeneratorInput) -> GeneratorOutput:
        feedback_block = ""
        if agent_input.feedback:
            bullets = "\n".join(f"- {item}" for item in agent_input.feedback)
            feedback_block = (
                "Reviewer feedback to apply in this new draft:\n"
                f"{bullets}\n"
            )

        prompt = (
            "You are Generator Agent.\n"
            "Goal: Generate draft educational content.\n\n"
            f"Input JSON:\n{json.dumps({'grade': agent_input.grade, 'topic': agent_input.topic}, indent=2)}\n\n"
            f"{feedback_block}"
            "Requirements:\n"
            "- Keep language appropriate for the given grade.\n"
            "- Ensure conceptual correctness.\n"
            "- Explanation must be 3 to 5 short sentences.\n"
            "- Create exactly 3 multiple choice questions.\n"
            "- Each MCQ must have exactly 4 options.\n"
            "- The answer must be one of A, B, C, D and match the correct option.\n"
            "- Test only ideas introduced in the explanation.\n"
            "Return JSON only.\n"
        )

        output_json = self.client.generate_json(
            prompt=prompt,
            schema=GENERATOR_SCHEMA,
            temperature=0.1,
        )
        return GeneratorOutput.from_dict(output_json)


class ReviewerAgent:
    def __init__(self, client: GeminiRestClient) -> None:
        self.client = client

    def run(
        self,
        generated_content: GeneratorOutput,
        grade: int,
        topic: str,
    ) -> ReviewerOutput:
        prompt = (
            "You are Reviewer Agent.\n"
            "Goal: Evaluate generated educational content.\n"
            "Evaluation criteria:\n"
            "- Age appropriateness\n"
            "- Conceptual correctness\n"
            "- Clarity\n\n"
            f"Grade: {grade}\n"
            f"Topic: {topic}\n"
            f"Generator output JSON:\n{json.dumps(generated_content.to_dict(), indent=2)}\n\n"
            "Rules:\n"
            "- Return status as pass or fail.\n"
            "- If status is fail, provide concrete feedback list items.\n"
            "- If status is pass, feedback can briefly confirm quality.\n"
            "Return JSON only.\n"
        )

        output_json = self.client.generate_json(
            prompt=prompt,
            schema=REVIEWER_SCHEMA,
            temperature=0.0,
        )
        return ReviewerOutput.from_dict(output_json)


@dataclass
class PipelineResult:
    generator_input: GeneratorInput
    draft_output: GeneratorOutput
    review_output: ReviewerOutput
    refined_output: Optional[GeneratorOutput]


def run_pipeline(grade: int, topic: str, api_key: str) -> PipelineResult:
    client = GeminiRestClient(api_key=api_key, model_name=MODEL_NAME)
    generator = GeneratorAgent(client)
    reviewer = ReviewerAgent(client)

    generator_input = GeneratorInput(grade=grade, topic=topic)
    draft = generator.run(generator_input)
    review = reviewer.run(generated_content=draft, grade=grade, topic=topic)

    refined: Optional[GeneratorOutput] = None
    if review.status == "fail":
        refined_input = GeneratorInput(grade=grade, topic=topic, feedback=review.feedback)
        refined = generator.run(refined_input)

    return PipelineResult(
        generator_input=generator_input,
        draft_output=draft,
        review_output=review,
        refined_output=refined,
    )


def resolve_api_key() -> str:
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key.strip()

    secret_key = ""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        secret_key = ""

    return str(secret_key).strip() if secret_key else ""


def render_flow_summary() -> None:
    st.markdown("### Agent Flow")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "**1) Generator Agent**\n"
            "Input: `{grade, topic}`\n"
            "Output: `{explanation, mcqs[]}`"
        )
    with col2:
        st.markdown(
            "**2) Reviewer Agent**\n"
            "Input: Generator JSON\n"
            "Output: `{status, feedback[]}`"
        )
    with col3:
        st.markdown(
            "**3) One Refinement Pass**\n"
            "If review fails, feedback is injected into Generator for one retry."
        )


def main() -> None:
    st.set_page_config(page_title="AI Assessment Agent Pipeline", layout="wide")
    st.title("AI Assessment: Agent-Based, UI-Driven")
    st.caption(f"Model: {MODEL_NAME}")

    api_key = resolve_api_key()

    with st.expander("API Configuration", expanded=False):
        if api_key:
            st.success("Gemini API key loaded from internal configuration.")
        else:
            st.warning(
                "No API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in environment "
                "variables or in .streamlit/secrets.toml, then restart the app."
            )

    render_flow_summary()

    st.markdown("### Run Pipeline")
    with st.form("agent_pipeline_form"):
        grade = st.number_input("Grade", min_value=1, max_value=12, value=4, step=1)
        topic = st.text_input("Topic", value="Types of angles")
        run_clicked = st.form_submit_button("Run Agent Pipeline")

    if not run_clicked:
        return

    if not topic.strip():
        st.error("Topic is required.")
        return

    if not api_key.strip():
        st.error("Gemini API key is required.")
        return

    try:
        with st.status("Running agent pipeline...", expanded=True) as status:
            st.write("Step 1/3: Generator Agent creates draft output.")
            result = run_pipeline(grade=int(grade), topic=topic.strip(), api_key=api_key.strip())

            st.write("Step 2/3: Reviewer Agent evaluates draft output.")
            if result.review_output.status == "fail":
                st.write("Step 3/3: Reviewer returned fail, running one refinement pass.")
            else:
                st.write("Step 3/3: Reviewer returned pass, refinement not needed.")

            status.update(label="Pipeline complete", state="complete", expanded=False)

        st.markdown("### Generator Input")
        st.json(
            {
                "grade": result.generator_input.grade,
                "topic": result.generator_input.topic,
            }
        )

        st.markdown("### Generator Output")
        st.json(result.draft_output.to_dict())

        st.markdown("### Reviewer Output")
        st.json(result.review_output.to_dict())

        if result.refined_output is not None:
            st.markdown("### Refined Output (One Pass)")
            st.json(result.refined_output.to_dict())
        else:
            st.info("Refinement was not triggered because reviewer status was pass.")

    except Exception as exc:  # noqa: BLE001
        st.error(f"Pipeline failed: {exc}")


if __name__ == "__main__":
    main()
