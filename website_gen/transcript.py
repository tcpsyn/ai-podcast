import re

SPEAKER_RE = re.compile(r"^([A-Z][A-Z0-9 .'-]{0,30}):\s*(.*)$", re.DOTALL)


def parse_transcript(raw: str) -> list[tuple[str, str]]:
    """Split a transcript into (speaker, text) turns.

    Paragraphs are blank-line separated. A paragraph that does not open with a
    SPEAKER: label is attributed to whoever spoke last, which is how the
    transcriber emits long turns that wrap.
    """
    turns: list[tuple[str, str]] = []
    current = None
    for para in re.split(r"\n\s*\n", raw or ""):
        para = para.strip()
        if not para:
            continue
        m = SPEAKER_RE.match(para)
        if m:
            current = m.group(1).strip()
            text = m.group(2).strip()
        else:
            text = para
        if current is None:
            current = "LUKE"
        if text:
            turns.append((current, text))
    return turns
