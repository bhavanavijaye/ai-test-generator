from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import re

app = FastAPI(title="AI Test Case Generator", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def call_ai(prompt: str) -> str:
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RequirementInput(BaseModel):
    requirements: str
    test_type: str = "all"


class APIInput(BaseModel):
    api_spec: str


class CodeInput(BaseModel):
    requirements: str
    language: str = "python"


def parse_tests(raw: str) -> list:
    tests = []
    blocks = re.split(r"(?:^|\n)\s*(?:TEST CASE|Test Case|TC)\s*\d+\s*[:\-]?", raw, flags=re.IGNORECASE)

    for block in blocks:
        block = block.strip()
        if len(block) < 15:
            continue

        def grab(label: str, text: str) -> str:
            pattern = rf"(?:^|\n)\s*{label}\s*[:\-]\s*(.+?)(?=\n\s*(?:Name|Type|Description|Steps|Expected|Priority|Test|TC)\s*[:\-]|\Z)"
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""

        name      = grab("Name", block)
        ttype     = grab("Type", block)
        desc      = grab("Description", block)
        steps_raw = grab("Steps", block)
        expected  = grab("Expected", block)
        priority  = grab("Priority", block)

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

        p = priority.lower()
        priority = "high" if "high" in p else ("low" if "low" in p else "medium")

        steps = []
        if steps_raw:
            raw_steps = re.split(r"\s*\|\s*|\n\s*[\d]+[\.\)]\s*|\n\s*[-•]\s*", steps_raw)
            steps = [s.strip() for s in raw_steps if s.strip() and len(s.strip()) > 3]

        if name or desc:
            tests.append({
                "name":        name or f"Test {len(tests)+1}",
                "type":        ttype,
                "description": desc or block[:200],
                "steps":       steps or ["Set up preconditions", "Execute action", "Verify outcome"],
                "expected":    expected or "System behaves as specified",
                "priority":    priority,
            })

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


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Test Case Generator is running"}


@app.get("/health")
def health():
    key = os.environ.get("GROQ_API_KEY")
    return {
        "backend": "ok",
        "groq": "configured" if key else "missing GROQ_API_KEY env variable",
        "model": "llama3-8b-8192"
    }


@app.post("/generate-tests")
async def generate_tests(inp: RequirementInput):
    type_focus = {
        "all":         "unit, integration, and edge case tests",
        "unit":        "unit tests only",
        "integration": "integration tests only",
        "edge":        "edge cases and negative tests only",
    }.get(inp.test_type, "unit, integration, and edge case tests")

    requirements = inp.requirements[:2000]

    prompt = f"""You are a QA engineer. Generate 5 test cases for the requirements below.

REQUIREMENTS:
{requirements}

Use EXACTLY this format for each test case:

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

    raw = call_ai(prompt)
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
Steps: Build valid request payload | Send request | Check response status | Verify response body
Expected: HTTP 200 with success response
Priority: high

TEST CASE 2
Name: Missing required fields returns 400
Type: negative
Description: Send request without required fields and verify error
Steps: Build payload with missing fields | Send request | Check error response
Expected: HTTP 400 with validation error
Priority: high

TEST CASE 3
Name: Invalid auth token returns 401
Type: negative
Description: Send request with invalid token and verify auth error
Steps: Set invalid Authorization header | Send request | Check response
Expected: HTTP 401 Unauthorized
Priority: high

TEST CASE 4
Name: Boundary value for numeric field
Type: edge_case
Description: Send request with boundary/extreme numeric value
Steps: Build payload with extreme value | Send request | Check response
Expected: HTTP 400 or handled gracefully
Priority: medium

TEST CASE 5
Name: Duplicate request handling
Type: edge_case
Description: Send the same request twice and check idempotency
Steps: Send valid request | Send same request again | Compare responses
Expected: Both return 200 or second returns 409
Priority: medium

Now write 5 test cases for the API spec above:"""

    raw = call_ai(prompt)
    tests = parse_tests(raw)
    return {"tests": tests, "count": len(tests), "raw": raw}


@app.post("/generate-code")
async def generate_code(inp: CodeInput):
    requirements = inp.requirements[:1500]
    lang = "pytest (Python)" if inp.language == "python" else "Jest (JavaScript)"
    imports = (
        "import pytest\nfrom unittest.mock import patch, MagicMock"
        if inp.language == "python"
        else "const { describe, test, expect, jest } = require('@jest/globals');"
    )

    prompt = f"""Write {lang} test code for the following:

{requirements}

Rules:
- Start with: {imports}
- Write exactly 4 test functions
- Each test has a comment explaining what it checks
- Use realistic mock data
- Include at least one negative/edge case test
- Write only code, no explanations, no markdown fences

Write the code now:"""

    raw = call_ai(prompt)
    code = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
    return {"code": code}
    
