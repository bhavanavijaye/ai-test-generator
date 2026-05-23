from fastapi import FastAPI, HTTPException # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware # pyright: ignore[reportMissingImports]
from pydantic import BaseModel # pyright: ignore[reportMissingImports]
import requests # pyright: ignore[reportMissingModuleSource]
import re

app = FastAPI(title="AI Test Case Generator", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"   # ← switched from gemma2:2b


class RequirementInput(BaseModel):
    requirements: str
    test_type: str = "all"


class APIInput(BaseModel):
    api_spec: str


class CodeInput(BaseModel):
    requirements: str
    language: str = "python"


def call_ollama(prompt: str) -> str:
    try:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,      # lower = more focused output
                "top_p": 0.9,
                "num_predict": 2048,
            }
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=240)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama not running. Run: ollama run llama3.2:3b"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def parse_tests(raw: str) -> list:
    """
    Robust parser — handles llama3.2 output variations.
    Tries block format first, falls back to line-by-line.
    """
    tests = []

    # Strategy 1: Split on TEST CASE markers
    blocks = re.split(r"(?:^|\n)\s*(?:TEST CASE|Test Case|TC)\s*\d+\s*[:\-]?", raw, flags=re.IGNORECASE)
    
    for block in blocks:
        block = block.strip()
        if len(block) < 15:
            continue

        def grab(label: str, text: str) -> str:
            """Extract a field value, handling multi-line content."""
            pattern = rf"(?:^|\n)\s*{label}\s*[:\-]\s*(.+?)(?=\n\s*(?:Name|Type|Description|Steps|Expected|Priority|Test|TC)\s*[:\-]|\Z)"
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""

        name     = grab("Name", block)
        ttype    = grab("Type", block)
        desc     = grab("Description", block)
        steps_raw= grab("Steps", block)
        expected = grab("Expected", block)
        priority = grab("Priority", block)

        # --- Normalize type ---
        ttype_lower = ttype.lower()
        if "unit" in ttype_lower:
            ttype = "unit"
        elif "integration" in ttype_lower:
            ttype = "integration"
        elif any(k in ttype_lower for k in ["edge","negative","boundary","security","invalid"]):
            ttype = "edge_case"
        elif "happy" in ttype_lower:
            ttype = "happy_path"
        else:
            ttype = "unit"

        # --- Normalize priority ---
        p = priority.lower()
        priority = "high" if "high" in p else ("low" if "low" in p else "medium")

        # --- Parse steps ---
        steps = []
        if steps_raw:
            # split on pipe, newline+number, or newline+dash
            raw_steps = re.split(r"\s*\|\s*|\n\s*[\d]+[\.\)]\s*|\n\s*[-•]\s*", steps_raw)
            steps = [s.strip() for s in raw_steps if s.strip() and len(s.strip()) > 3]

        if name or desc:
            tests.append({
                "name":        name or f"Test {len(tests)+1}",
                "type":        ttype,
                "description": desc or block[:200],
                "steps":       steps or ["Set up preconditions", "Execute action", "Verify outcome"],
                "expected":    expected or "System behaves as specified in requirements",
                "priority":    priority,
            })

    # Strategy 2: fallback — chunk by double newline
    if not tests:
        chunks = [c.strip() for c in re.split(r"\n{2,}", raw) if len(c.strip()) > 40]
        for i, chunk in enumerate(chunks[:8]):
            lines = [l.strip() for l in chunk.splitlines() if l.strip()]
            tests.append({
                "name":        lines[0][:70] if lines else f"Test {i+1}",
                "type":        "unit",
                "description": " ".join(lines[1:3]) if len(lines) > 1 else chunk[:180],
                "steps":       ["Execute the described scenario", "Check the result"],
                "expected":    lines[-1][:150] if lines else "Correct behavior",
                "priority":    "medium",
            })

    return tests


@app.get("/health")
async def health():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return {
                "backend": "ok",
                "ollama": "ok",
                "model": MODEL,
                "model_available": any(MODEL.split(":")[0] in m for m in models),
                "available_models": models
            }
    except Exception:
        pass
    return {"backend": "ok", "ollama": "not running", "model": MODEL}


@app.post("/generate-tests")
async def generate_tests(inp: RequirementInput):
    type_focus = {
        "all":         "unit, integration, and edge case tests",
        "unit":        "unit tests only",
        "integration": "integration tests only",
        "edge":        "edge cases and negative tests only",
    }.get(inp.test_type, "unit, integration, and edge case tests")

    # Truncate very long inputs to prevent model confusion
    requirements = inp.requirements[:2000]

    prompt = f"""You are a QA engineer. Generate 5 test cases for the requirements below.

REQUIREMENTS:
{requirements}

Use EXACTLY this format for each test case — no extra text, no markdown:

TEST CASE 1
Name: [short test name]
Type: unit
Description: [one sentence describing what is tested]
Steps: [step 1] | [step 2] | [step 3]
Expected: [what should happen]
Priority: high

TEST CASE 2
Name: [short test name]
Type: integration
Description: [one sentence]
Steps: [step 1] | [step 2] | [step 3]
Expected: [expected result]
Priority: high

TEST CASE 3
Name: [short test name]
Type: edge_case
Description: [one sentence]
Steps: [step 1] | [step 2]
Expected: [expected error or behavior]
Priority: medium

TEST CASE 4
Name: [short test name]
Type: edge_case
Description: [one sentence]
Steps: [step 1] | [step 2]
Expected: [expected result]
Priority: medium

TEST CASE 5
Name: [short test name]
Type: integration
Description: [one sentence]
Steps: [step 1] | [step 2] | [step 3]
Expected: [expected result]
Priority: low

Focus on: {type_focus}
Write the 5 test cases now:"""

    raw = call_ollama(prompt)
    tests = parse_tests(raw)
    return {"tests": tests, "count": len(tests), "raw": raw}


@app.post("/analyze-api")
async def analyze_api(inp: APIInput):
    spec = inp.api_spec[:2000]

    prompt = f"""You are a QA engineer. Generate 5 API test cases for this endpoint.

API SPECIFICATION:
{spec}

Use EXACTLY this format:

TEST CASE 1
Name: Valid request returns 200
Type: happy_path
Description: Send a valid request and verify successful response
Steps: Build valid request payload | Send POST request | Check response status | Verify response body
Expected: HTTP 200 with success status and activeUntil timestamp
Priority: high

TEST CASE 2
Name: Invalid promo code returns 400
Type: negative
Description: Send request with invalid promo code and verify error response
Steps: Build payload with invalid promo code | Send POST request | Check error response
Expected: HTTP 400 with error INVALID_PROMO_CODE
Priority: high

TEST CASE 3
Name: Declined card returns 402
Type: negative
Description: Use a card that will be declined and verify payment error
Steps: Use a test declined card token | Send POST request | Check error response
Expected: HTTP 402 with error CARD_DECLINED
Priority: high

TEST CASE 4
Name: Missing required fields returns 400
Type: edge_case
Description: Send request without required planId or paymentMethodId
Steps: Build payload with missing fields | Send POST request | Check error response
Expected: HTTP 400 or 422 with validation error
Priority: medium

TEST CASE 5
Name: Invalid UUID for planId
Type: edge_case
Description: Send request with malformed planId that is not a valid UUID
Steps: Build payload with planId set to random string | Send POST request | Check error
Expected: HTTP 400 with validation error
Priority: medium

Now write 5 test cases for the API spec above using this exact format:"""

    raw = call_ollama(prompt)
    tests = parse_tests(raw)
    return {"tests": tests, "count": len(tests), "raw": raw}


@app.post("/generate-code")
async def generate_code(inp: CodeInput):
    requirements = inp.requirements[:1500]
    lang = "pytest (Python)" if inp.language == "python" else "Jest (JavaScript)"
    imports = "import pytest\nfrom unittest.mock import patch, MagicMock" if inp.language == "python" else "const { describe, test, expect, jest } = require('@jest/globals');"

    prompt = f"""Write {lang} test code for the following:

{requirements}

Rules:
- Start with: {imports}
- Write exactly 4 test functions
- Each test has a comment explaining what it checks
- Use realistic mock data (Stripe tokens, UUIDs, etc.)
- Include at least one negative/edge case test
- Write only code — no explanations, no markdown

Write the code now:"""

    raw = call_ollama(prompt)
    # Strip markdown code fences if model adds them
    code = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
    return {"code": code}