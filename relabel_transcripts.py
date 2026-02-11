#!/usr/bin/env python3
"""Re-label podcast transcripts with LUKE:/CALLER: speaker labels using LLM."""

import os, re, sys, time, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
TRANSCRIPT_DIR = Path(__file__).parent / "website" / "transcripts"
MODEL = "anthropic/claude-3.5-sonnet"
CHUNK_SIZE = 8000

PROMPT = """Insert speaker labels into this radio show transcript. The show is "Luke at the Roost". The host is LUKE. Callers call in one at a time.

CRITICAL: Output EVERY SINGLE WORD from the input. Do NOT summarize, shorten, paraphrase, or skip ANY text. The output must contain the EXACT SAME words as the input, with ONLY speaker labels and line breaks added.

At each speaker change, insert a blank line and the new speaker's label (e.g., "LUKE:" or "REGGIE:").

Speaker identification:
- LUKE is the host — he introduces callers, asks questions, does sponsor reads, opens and closes the show
- Callers are introduced by name by Luke (e.g., "let's talk to Earl", "next up Brenda")
- Use caller FIRST NAME in caps as the label
- When Luke says "Tell me about..." or asks a question, that's LUKE
- When someone responds with their story/opinion/answer, that's the CALLER

Output format — ONLY the labeled transcript with blank lines between turns. No notes, no commentary."""

CONTEXT_PROMPT = "\n\nCONTEXT: The previous section ended with the speaker {speaker}. Last few words: \"{tail}\""


def chunk_text(text, max_chars=CHUNK_SIZE):
    if len(text) <= max_chars:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_chars:
            # Merge tiny tails into the previous chunk
            if chunks and len(text) < 1000:
                chunks[-1] = chunks[-1] + " " + text
            else:
                chunks.append(text)
            break

        # Find a good break point near max_chars
        pos = text[:max_chars].rfind('. ')
        if pos < max_chars // 2:
            pos = text[:max_chars].rfind('? ')
        if pos < max_chars // 2:
            pos = text[:max_chars].rfind('! ')
        if pos < max_chars // 2:
            pos = max_chars

        chunks.append(text[:pos + 1].strip())
        text = text[pos + 1:].strip()

    return chunks


def label_chunk(text, context=""):
    prompt = PROMPT + "\n\nTRANSCRIPT:\n" + text
    if context:
        prompt += context

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "temperature": 0
        }
    )

    if response.status_code != 200:
        print(f"  API error: {response.status_code} {response.text[:200]}")
        return None

    content = response.json()["choices"][0]["message"]["content"].strip()

    # Remove any markdown code block wrappers
    if content.startswith("```"):
        content = re.sub(r'^```\w*\n?', '', content)
        content = re.sub(r'\n?```$', '', content)

    return content


def get_last_speaker(text):
    lines = text.strip().split('\n')
    for line in reversed(lines):
        match = re.match(r'^([A-Z][A-Z\s\'-]+?):', line.strip())
        if match:
            return match.group(1)
    return "LUKE"


def validate_output(original, labeled):
    """Basic validation that the output looks right."""
    # Check that speaker labels exist (at least 1 for short chunks)
    speaker_lines = re.findall(r'^[A-Z][A-Z\s\'-]+?:', labeled, re.MULTILINE)
    if len(speaker_lines) < 1:
        return False

    # Check that output isn't drastically shorter (allowing for some reformatting)
    orig_words = len(original.split())
    labeled_words = len(labeled.split())
    if labeled_words < orig_words * 0.5:
        print(f"  WARNING: Output is {labeled_words} words vs {orig_words} input words ({labeled_words * 100 // orig_words}%)")
        return False

    return True


def process_transcript(filepath):
    text = filepath.read_text().strip()
    # Strip existing timestamp markers
    text = re.sub(r'\[[\d:]+\]\s*', '', text)
    # Normalize whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    print(f"  {len(text)} chars")

    chunks = chunk_text(text)
    print(f"  {len(chunks)} chunk(s)")

    labeled_parts = []
    context = ""

    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...")
        labeled = label_chunk(chunk, context)

        if labeled is None:
            print(f"  ERROR: API call failed for chunk {i + 1}")
            return None

        if not validate_output(chunk, labeled):
            print(f"  ERROR: Validation failed for chunk {i + 1}")
            return None

        labeled_parts.append(labeled)

        # Build context for next chunk
        last_speaker = get_last_speaker(labeled)
        tail = labeled.strip()[-100:]
        context = CONTEXT_PROMPT.format(speaker=last_speaker, tail=tail)

        if i < len(chunks) - 1:
            time.sleep(0.5)

    # Join parts, ensuring proper spacing between chunks
    result = "\n\n".join(labeled_parts)
    # Normalize: ensure exactly one blank line between speaker turns
    result = re.sub(r'\n{3,}', '\n\n', result)
    # Fix format: put speaker label on same line as text (SPEAKER:\ntext -> SPEAKER: text)
    result = re.sub(r'^([A-Z][A-Z\s\'-]+?):\s*\n(?!\n)', r'\1: ', result, flags=re.MULTILINE)
    return result


def main():
    if not API_KEY:
        print("Error: OPENROUTER_API_KEY not set")
        sys.exit(1)

    files = sys.argv[1:] if len(sys.argv) > 1 else None
    if files:
        transcripts = [TRANSCRIPT_DIR / f for f in files]
    else:
        transcripts = sorted(TRANSCRIPT_DIR.glob("*.txt"))

    for filepath in transcripts:
        if not filepath.exists():
            print(f"Skipping {filepath.name} (not found)")
            continue
        print(f"\nProcessing: {filepath.name}")
        labeled = process_transcript(filepath)
        if labeled is None:
            print(f"  SKIPPED (processing failed)")
            continue
        filepath.write_text(labeled + "\n")
        print(f"  Saved ({len(labeled)} chars)")

    print("\nDone!")


if __name__ == "__main__":
    main()
