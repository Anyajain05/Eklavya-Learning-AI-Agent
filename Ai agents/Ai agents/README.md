# AI Assessment: Agent-Based, UI-Driven

This project implements a UI-driven two-agent pipeline using Gemini API (`gemini-2.5-flash`):

1. Generator Agent
- Input: `{ "grade": <int>, "topic": <string> }`
- Output:
  - `explanation` (string)
  - `mcqs` (list of question objects)

2. Reviewer Agent
- Input: Generator output JSON
- Output:
  - `status` (`pass` or `fail`)
  - `feedback` (list of strings)

If Reviewer returns `fail`, the pipeline performs one refinement pass by rerunning Generator with reviewer feedback.

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Gemini API key internally (no sidebar input):

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"
```

Or use Streamlit secrets:

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Put your real key in `.streamlit/secrets.toml`

The app automatically reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) from environment variables first, then from Streamlit secrets.

## Run

```bash
python -m streamlit run app.py
```

## Notes

- Model used: `gemini-2.5-flash`
- UI shows all pipeline stages:
  - Generator output
  - Reviewer feedback
  - Refined output (if reviewer fails)
