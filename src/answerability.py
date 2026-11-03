"""Answerability check: can a QA model recover the answer key from the passage?

Used in the evaluation as a metric, and in the demo as a filter that catches
questions whose answer key is not supported by the passage (e.g. a wrong key).
"""
import collections
import re
import string

QA_MODEL = "deepset/roberta-base-squad2"


def normalize(s):
    """SQuAD-style normalization: lowercase, drop punctuation and articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    """Word-overlap F1 between two answers (the standard SQuAD measure)."""
    p, g = normalize(pred).split(), normalize(gold).split()
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


class AnswerChecker:
    """Extractive QA model that answers questions from a passage, or returns "" if it can't."""

    def __init__(self, model_name=QA_MODEL, device=None):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()

    def answer(self, questions, passages, batch_size=16, max_answer_len=30):
        """Return the passage span that best answers each question, or "" for no answer."""
        import torch

        answers = []
        for i in range(0, len(questions), batch_size):
            qs, ps = questions[i:i + batch_size], passages[i:i + batch_size]
            enc = self.tok(qs, ps, truncation="only_second", max_length=512, padding=True,
                           return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")
            with torch.no_grad():
                out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
            for j in range(len(qs)):
                s, e = out.start_logits[j].cpu(), out.end_logits[j].cpu()
                ctx = torch.tensor([k for k, sid in enumerate(enc.sequence_ids(j)) if sid == 1])
                k = min(20, len(ctx))
                starts = ctx[s[ctx].topk(k).indices].tolist()
                ends = ctx[e[ctx].topk(k).indices].tolist()
                best, span = float("-inf"), None
                for a in starts:
                    for b in ends:
                        if a <= b < a + max_answer_len and s[a] + e[b] > best:
                            best, span = (s[a] + e[b]).item(), (a, b)
                null_score = (s[0] + e[0]).item()
                if span is None or null_score > best:
                    answers.append("")
                else:
                    answers.append(ps[j][offsets[j][span[0]][0]:offsets[j][span[1]][1]])
        return answers

    def answerable(self, questions, answers, passages, threshold=0.5):
        """True for each question whose answer key matches the QA model's answer."""
        preds = self.answer(questions, passages)
        return [token_f1(p, a) >= threshold for p, a in zip(preds, answers)]