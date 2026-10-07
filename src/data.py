from .config import EVAL_TEXTS


def load_eval_texts():
    raw = EVAL_TEXTS.read_text(encoding="utf-8")
    return [p.strip() for p in raw.split("\n\n") if p.strip()]
