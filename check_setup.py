"""Quick setup check that runs without a GPU: loads the notes and tests the parser."""
from src.notes import load_notes
from src.parse import parse_output

print(len(load_notes("data/notes")), "passages (expected 12)")
print(parse_output('{"question": "Q?", "answer": "a", "distractors": ["b", "c", "d"]}'))
print(parse_output("no json here"), "(expected: (None, 'no_json'))")