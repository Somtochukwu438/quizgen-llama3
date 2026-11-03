"""Quiz generation with validation and regeneration.

Shared by the evaluation notebook and the demo app, so both use exactly the
same tested code path.
"""
import re
from contextlib import nullcontext

from .parse import parse_output
from .prompts import build_messages

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def load_model(adapter_path=None, model_name=MODEL_NAME):
    """Load the 4-bit base model, optionally with the LoRA adapter on top."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto"
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def generate(model, tok, passages, sample=False, batch_size=8, max_new_tokens=200):
    """Generate one raw output per passage.

    sample=False is greedy decoding (used for evaluation); sample=True uses light
    sampling (used for regeneration and for varied demo quizzes).
    """
    import torch

    settings = (dict(do_sample=True, temperature=0.7, top_p=0.9)
                if sample else dict(do_sample=False))
    outs = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i : i + batch_size]
        prompts = [
            tok.apply_chat_template(build_messages(p), tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id, **settings)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def needs_retry(parsed):
    """True if the output is unusable as a quiz question (exact-match checks)."""
    return parsed is None or parsed["answer_repeated"] or parsed["distractor_duplicates"]


def _tokens(s):
    """Lowercased words, with a crude plural strip so "shortage" matches "shortages"."""
    words = re.findall(r"[a-z0-9]+", s.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in words}


def near_duplicate(parsed, threshold=0.8):
    """True if any distractor shares most of its words with the answer.

    Catches near-copies that exact matching misses, e.g. "insomnia, apnea, ..."
    vs "insomnia, sleep apnea, ...", or "shortage" vs "shortages". Legitimately
    similar options are unaffected: "rem sleep" vs "nrem sleep" shares only half
    its words.
    """
    answer = _tokens(parsed["answer"])
    for d in parsed["distractors"]:
        dist = _tokens(d)
        if answer and dist and len(answer & dist) / max(len(answer), len(dist)) >= threshold:
            return True
    return False


def _failing(records, strict, checker, check_threshold):
    """Indices of records that should be regenerated."""
    fail = [i for i, r in enumerate(records)
            if needs_retry(r["parsed"]) or (strict and near_duplicate(r["parsed"]))]
    if checker is not None:
        rest = [i for i in range(len(records)) if i not in fail]
        if rest:
            ok = checker.answerable(
                [records[i]["parsed"]["question"] for i in rest],
                [records[i]["parsed"]["answer"] for i in rest],
                [records[i]["passage"] for i in rest],
                threshold=check_threshold,
            )
            fail += [i for i, good in zip(rest, ok) if not good]
    return sorted(fail)


def generate_quiz(model, tok, passages, use_adapter=True, max_tries=3, fresh=False,
                  strict=False, checker=None, check_threshold=0.67):
    """Generate one question per passage, with validation and up to max_tries attempts.

    fresh=False:  the first attempt is greedy, so results are repeatable
                  (used for evaluation and the model comparison).
    fresh=True:   every attempt is sampled, so each call gives new questions.
    strict=True:  also retry near-duplicate distractors.
    checker:      an AnswerChecker; if given, also retry questions whose answer key
                  the QA model cannot recover from the passage. check_threshold is
                  stricter than the evaluation's 0.5 because a one-word difference
                  (e.g. "rem sleep" vs "nrem sleep") can flip the meaning.
    strict and checker are used by the demo only; the reported evaluation used
    the exact-match checks alone.

    Returns a list of dicts with passage, raw, parsed, error, attempts and flagged
    (True if the final question still failed validation after all attempts).
    use_adapter=False turns the LoRA adapter off, so the model behaves as the baseline.
    """
    off = not use_adapter and hasattr(model, "disable_adapter")
    with (model.disable_adapter() if off else nullcontext()):
        records = []
        for p, out in zip(passages, generate(model, tok, passages, sample=fresh)):
            parsed, err = parse_output(out)
            records.append({"passage": p, "raw": out, "parsed": parsed,
                            "error": err, "attempts": 1})
        for _ in range(max_tries - 1):
            idx = _failing(records, strict, checker, check_threshold)
            if not idx:
                break
            retries = generate(model, tok, [records[i]["passage"] for i in idx], sample=True)
            for i, out in zip(idx, retries):
                parsed, err = parse_output(out)
                records[i].update(raw=out, parsed=parsed, error=err,
                                  attempts=records[i]["attempts"] + 1)
        flagged = set(_failing(records, strict, checker, check_threshold))
        for i, r in enumerate(records):
            r["flagged"] = i in flagged
    return records