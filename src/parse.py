"""Parse and validate model outputs into quiz questions."""
import json
import re

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _norm(s):
    return s.strip().lower()


def parse_output(text):
    """Parse one model output.

    Returns (result, error):
      result: dict with question, answer, distractors and quality flags,
              or None if the output does not have the required JSON structure
      error:  None if the structure is valid, otherwise a short reason code

    Quality flags (only set on structurally valid outputs):
      strict                 output was ONLY the JSON object, no extra text
      answer_repeated        the correct answer also appears as a distractor
      distractor_duplicates  two or more distractors are identical
    """
    raw = text.strip()
    cleaned = FENCE_RE.sub("", raw).strip()

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None, "no_json"
    blob = cleaned[start : end + 1]

    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "not_object"

    q, a, d = data.get("question"), data.get("answer"), data.get("distractors")
    if not isinstance(q, str) or not q.strip():
        return None, "missing_question"
    if not isinstance(a, str) or not a.strip():
        return None, "missing_answer"
    if not isinstance(d, list) or len(d) != 3 or not all(
        isinstance(x, str) and x.strip() for x in d
    ):
        return None, "bad_distractors"

    d_norm = [_norm(x) for x in d]
    return {
        "question": q.strip(),
        "answer": a.strip(),
        "distractors": [x.strip() for x in d],
        "strict": raw == blob,
        "answer_repeated": _norm(a) in d_norm,
        "distractor_duplicates": len(set(d_norm)) < 3,
    }, None