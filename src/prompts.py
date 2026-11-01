"""Prompt templates shared by the baseline and fine-tuned models.

Both models must see exactly the same instruction so the comparison is fair.
"""
import json

SYSTEM_PROMPT = (
    "You are a teacher who writes clear multiple-choice quiz questions "
    "for students."
)

INSTRUCTION = (
    "Read the passage below and write one multiple-choice question that can be "
    "answered using only information in the passage.\n"
    "Respond with only a JSON object in exactly this format, with no other text:\n"
    '{"question": "...", "answer": "...", "distractors": ["...", "...", "..."]}\n'
    "The answer must be correct according to the passage. The three distractors "
    "must be wrong but plausible.\n\n"
    "Passage:\n"
)


def build_messages(passage):
    """Chat messages for generation (used by both baseline and fine-tuned model)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION + passage.strip()},
    ]


def format_target(question, answer, distractors):
    """The JSON string the model should produce."""
    return json.dumps(
        {"question": question, "answer": answer, "distractors": list(distractors)}
    )


def sciq_to_messages(ex):
    """Full training conversation for one SciQ example: prompt plus correct output."""
    target = format_target(
        ex["question"],
        ex["correct_answer"],
        [ex["distractor1"], ex["distractor2"], ex["distractor3"]],
    )
    return build_messages(ex["support"]) + [
        {"role": "assistant", "content": target}
    ]