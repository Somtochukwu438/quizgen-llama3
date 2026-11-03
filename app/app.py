"""Gradio demo: interactive quizzes from course notes or pasted text."""
import random
import re

import gradio as gr

from src.generate import generate_quiz

MAX_Q = 5
NOTES, OWN_TEXT = "Topic from my notes", "Paste my own text"


def chunk_text(text, target_words=120):
    """Split pasted text into passages of roughly target_words, on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], []
    for s in sentences:
        current.append(s)
        if sum(len(x.split()) for x in current) >= target_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 15]


def format_record(title, rec):
    """Markdown for one generated question, used in the comparison view."""
    if not rec["parsed"]:
        return f"### {title}\nNo valid question after {rec['attempts']} attempt(s) ({rec['error']})."
    p = rec["parsed"]
    return (f"### {title}\n**Q:** {p['question']}\n\n**Answer:** {p['answer']}\n\n"
            f"**Distractors:** {', '.join(p['distractors'])}\n\n_Attempts: {rec['attempts']}_")


def build_app(model, tok, index):
    def get_passages(source, topic, text, n):
        if source == NOTES:
            hits = index.search(topic, k=n) if topic.strip() else []
            return [h["text"] for h in hits], [f"{h['week']} · {h['section']}" for h in hits]
        chunks = chunk_text(text)[:n]
        return chunks, [f"Your text, part {i + 1}" for i in range(len(chunks))]

    def make_quiz(source, topic, text, n):
        passages, labels = get_passages(source, topic, text, int(n))
        records = generate_quiz(model, tok, passages, fresh=True, strict=True) if passages else []
        answers, cols, qs, radios, feedback = [], [], [], [], []
        for i in range(MAX_Q):
            rec = records[i] if i < len(records) else None
            if rec and rec["parsed"]:
                p = rec["parsed"]
                options = [p["answer"]] + p["distractors"]
                random.shuffle(options)
                answers.append(p["answer"])
                cols.append(gr.Column(visible=True))
                qs.append(gr.Markdown(f"**Q{i + 1}. {p['question']}**  \n_Source: {labels[i]}_"))
                radios.append(gr.Radio(choices=options, value=None))
            else:
                answers.append(None)
                cols.append(gr.Column(visible=False))
                qs.append(gr.Markdown(""))
                radios.append(gr.Radio(choices=[], value=None))
            feedback.append(gr.Markdown(""))
        made = sum(a is not None for a in answers)
        if not passages:
            status = "Enter a topic, or paste at least a few sentences of text."
        else:
            status = f"Generated {made} question(s). Pick an answer to check it."
        return [status, answers] + cols + qs + radios + feedback

    def make_feedback(i):
        def check(choice, answers):
            if not answers or i >= len(answers) or answers[i] is None or choice is None:
                return ""
            if choice == answers[i]:
                return "✅ Correct!"
            return f"❌ Not quite. The answer is **{answers[i]}**."
        return check

    def compare(source, topic, text):
        passages, labels = get_passages(source, topic, text, 1)
        if not passages:
            return "Enter a topic, or paste at least a few sentences of text.", "", ""
        base = generate_quiz(model, tok, passages, use_adapter=False)[0]
        tuned = generate_quiz(model, tok, passages, use_adapter=True)[0]
        return (f"**Passage ({labels[0]}):** {passages[0]}",
                format_record("Baseline (no fine-tuning)", base),
                format_record("Fine-tuned (LoRA on SciQ)", tuned))

    with gr.Blocks(title="Quiz Generator: Llama 3.2 + LoRA") as demo:
        gr.Markdown("# Quiz Generator\nLlama 3.2 3B fine-tuned with LoRA on SciQ, "
                    "with retrieval over biological psychology notes.")
        source = gr.Radio([NOTES, OWN_TEXT], value=NOTES, label="Where should questions come from?")
        topic = gr.Textbox(label="Topic", placeholder="e.g. sleep stages, pain, brain imaging")
        text = gr.Textbox(label="Your text", lines=8, visible=False,
                          placeholder="Paste a paragraph or more of notes")
        n = gr.Slider(1, MAX_Q, value=3, step=1, label="Number of questions")
        source.change(lambda s: (gr.Textbox(visible=s == NOTES), gr.Textbox(visible=s == OWN_TEXT)),
                      inputs=source, outputs=[topic, text])
        with gr.Row():
            quiz_btn = gr.Button("Make quiz", variant="primary")
            compare_btn = gr.Button("Compare baseline vs fine-tuned")

        status = gr.Markdown()
        answers = gr.State([])
        cols, qs, radios, feedback = [], [], [], []
        for i in range(MAX_Q):
            with gr.Column(visible=False) as col:
                qs.append(gr.Markdown())
                radios.append(gr.Radio(choices=[], label="Your answer", interactive=True))
                feedback.append(gr.Markdown())
            cols.append(col)
        for i in range(MAX_Q):
            radios[i].change(make_feedback(i), inputs=[radios[i], answers], outputs=feedback[i])
        quiz_btn.click(make_quiz, inputs=[source, topic, text, n],
                       outputs=[status, answers] + cols + qs + radios + feedback)

        with gr.Accordion("Model comparison", open=False):
            passage_md = gr.Markdown()
            with gr.Row():
                base_md, tuned_md = gr.Markdown(), gr.Markdown()
        compare_btn.click(compare, inputs=[source, topic, text],
                          outputs=[passage_md, base_md, tuned_md])
    return demo