#!/usr/bin/env python3
"""Generate social media announcement images for Luke at the Roost.

Usage:
    python make_social_post.py                           # regenerate with defaults
    python make_social_post.py --title "NEW FEATURE"     # custom title
    python make_social_post.py --body body_text.txt      # body from file

Outputs square (1080x1080) and landscape (1200x675) PNGs to social_posts/.
"""

import argparse
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageOps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COVER = os.path.join(SCRIPT_DIR, "website/images/cover.png")
OUT_DIR = os.path.join(SCRIPT_DIR, "social_posts")

# Brand colors
BG = (18, 13, 7)
ACCENT = (232, 121, 29)
WHITE = (255, 255, 255)
MUTED = (175, 165, 150)
LIGHTER = (220, 215, 205)

# macOS system fonts — swap these on Linux/Windows
FONT_BLACK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"


def load_font(path, size):
    return ImageFont.truetype(path, size)


def text_bbox(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1], bb[1]  # width, height, y_offset


def wrap_text(draw, text, x, y, max_w, font, fill, line_gap=10,
              cover_right=None, cover_bottom=None):
    """Word-wrap text onto the image, narrowing lines that overlap the cover.

    line_gap: fixed pixel gap between lines (not a multiplier).
    Returns y just below the last line of text (no trailing gap)."""
    words = text.split()
    lines = []
    cur = ""
    cur_y = y

    for word in words:
        test = f"{cur} {word}".strip()
        eff_w = max_w
        if cover_right and cover_bottom and cur_y < cover_bottom:
            eff_w = cover_right - x - 20

        tw, th, _ = text_bbox(draw, test, font)
        if tw > eff_w and cur:
            lines.append((cur, cur_y))
            _, lh, _ = text_bbox(draw, cur, font)
            cur_y += lh + line_gap
            cur = word
        else:
            cur = test

    if cur:
        lines.append((cur, cur_y))
        _, lh, _ = text_bbox(draw, cur, font)

    for line, ly in lines:
        draw.text((x, ly), line, font=font, fill=fill)

    return cur_y + lh  # return y just past the last line's bottom


def center_text(draw, text, y, canvas_w, font, fill):
    tw, th, _ = text_bbox(draw, text, font)
    draw.text(((canvas_w - tw) // 2, y), text, font=font, fill=fill)
    return y + th


def draw_email_box(draw, email, y, canvas_w, font):
    tw, th, y_off = text_bbox(draw, email, font)
    px, py = 22, 16
    box_w = tw + px * 2
    box_x = (canvas_w - box_w) // 2
    draw.rounded_rectangle(
        [box_x, y, box_x + box_w, y + th + py * 2],
        radius=8, fill=(45, 30, 12), outline=ACCENT, width=2,
    )
    draw.text((box_x + px, y + py - y_off), email, font=font, fill=ACCENT)
    return y + th + py * 2


def draw_accent_bars(draw, w, h, thickness):
    draw.rectangle([0, 0, w, thickness], fill=ACCENT)
    draw.rectangle([0, h - thickness, w, h], fill=ACCENT)


def paste_cover(img, x, y, size, radius):
    cover = Image.open(COVER).resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size, size], radius=radius, fill=255)
    img.paste(cover, (x, y), mask)


def make_square(title, paragraphs, email, filename="email_announcement_square.png"):
    W = 1080
    img = Image.new("RGB", (W, W), BG)
    draw = ImageDraw.Draw(img)
    draw_accent_bars(draw, W, W, 8)

    # Cover image — top right
    cover_size, cover_x, cover_y = 240, W - 290, 35
    paste_cover(img, cover_x, cover_y, cover_size, 20)
    cover_bottom = cover_y + cover_size + 15

    m = 60
    y = 40
    tw_full = W - m * 2

    # Header
    draw.text((m, y), "LUKE AT THE ROOST", font=load_font(FONT_BOLD, 24), fill=ACCENT)
    y += 30
    tag = load_font(FONT_REG, 20)
    draw.text((m, y), "Late-night call-in radio", font=tag, fill=MUTED)
    draw.text((m, y + 26), "powered by AI", font=tag, fill=MUTED)
    y += 75

    # Consistent spacing constants
    LINE_GAP = 12       # between lines within a block
    SECTION_GAP = 32    # between sections (body→CTA, CTA→footer)
    PARA_GAP = 26       # between body paragraphs
    TITLE_GAP = 48      # between title and first body paragraph

    # Title
    y = wrap_text(draw, title, m, y, tw_full, load_font(FONT_BLACK, 72), WHITE,
                  line_gap=LINE_GAP, cover_right=cover_x, cover_bottom=cover_bottom)
    y += TITLE_GAP

    # Body paragraphs
    body_font = load_font(FONT_REG, 32)
    colors = [LIGHTER] + [MUTED] * (len(paragraphs) - 1)
    for i, (para, color) in enumerate(zip(paragraphs, colors)):
        cr = cover_x if y < cover_bottom else None
        cb = cover_bottom if y < cover_bottom else None
        y = wrap_text(draw, para, m, y, tw_full, body_font, color,
                      line_gap=LINE_GAP, cover_right=cr, cover_bottom=cb)
        if i < len(paragraphs) - 1:
            y += PARA_GAP

    y += SECTION_GAP

    # Email CTA
    y = draw_email_box(draw, email, y, W, load_font(FONT_BOLD, 36))
    y += SECTION_GAP

    # Footer
    y = center_text(draw, "New episodes drop daily. Be part of the next one.",
                    y, W, load_font(FONT_REG, 24), MUTED)
    y += PARA_GAP
    info = load_font(FONT_REG, 22)
    center_text(draw, "lukeattheroost.com", y, W, info, ACCENT)
    y += PARA_GAP
    center_text(draw, "Spotify  \u00b7  Apple Podcasts  \u00b7  YouTube  \u00b7  RSS",
                y, W, info, MUTED)

    os.makedirs(OUT_DIR, exist_ok=True)
    img.save(os.path.join(OUT_DIR, filename), quality=95)
    print(f"Square: {filename}")


def make_landscape(title, paragraphs, email, filename="email_announcement_twitter.png"):
    TW, TH = 1200, 675
    img = Image.new("RGB", (TW, TH), BG)
    draw = ImageDraw.Draw(img)
    draw_accent_bars(draw, TW, TH, 6)

    # Cover image — top right
    cover_size, cover_x, cover_y = 180, TW - 220, 22
    paste_cover(img, cover_x, cover_y, cover_size, 16)
    cover_bottom = cover_y + cover_size + 10

    m = 45
    y = 25
    tw_full = TW - m * 2

    # Header
    draw.text((m, y), "LUKE AT THE ROOST", font=load_font(FONT_BOLD, 20), fill=ACCENT)
    y += 24
    draw.text((m, y), "Late-night call-in radio powered by AI",
              font=load_font(FONT_REG, 17), fill=MUTED)
    y += 38

    # Consistent spacing constants
    LINE_GAP = 8        # between lines within a block
    SECTION_GAP = 20    # between sections
    PARA_GAP = 16       # between body paragraphs
    TITLE_GAP = 32      # between title and first body paragraph

    # Title
    y = wrap_text(draw, title, m, y, tw_full, load_font(FONT_BLACK, 50), WHITE,
                  line_gap=LINE_GAP, cover_right=cover_x, cover_bottom=cover_bottom)
    y += TITLE_GAP

    # Body paragraphs
    body_font = load_font(FONT_REG, 23)
    colors = [LIGHTER] + [MUTED] * (len(paragraphs) - 1)
    for i, (para, color) in enumerate(zip(paragraphs, colors)):
        cr = cover_x if y < cover_bottom else None
        cb = cover_bottom if y < cover_bottom else None
        y = wrap_text(draw, para, m, y, tw_full, body_font, color,
                      line_gap=LINE_GAP, cover_right=cr, cover_bottom=cb)
        if i < len(paragraphs) - 1:
            y += PARA_GAP

    y += SECTION_GAP

    # Email CTA
    y = draw_email_box(draw, email, y, TW, load_font(FONT_BOLD, 26))
    y += SECTION_GAP

    # Footer
    y = center_text(draw, "New episodes drop daily. Be part of the next one.",
                    y, TW, load_font(FONT_REG, 19), MUTED)
    y += PARA_GAP
    center_text(draw, "lukeattheroost.com  \u00b7  Spotify  \u00b7  Apple Podcasts  \u00b7  YouTube",
                y, TW, load_font(FONT_REG, 17), (140, 132, 120))

    os.makedirs(OUT_DIR, exist_ok=True)
    img.save(os.path.join(OUT_DIR, filename), quality=95)
    print(f"Landscape: {filename}")


# --- Default content ---

DEFAULT_TITLE = "NOW ACCEPTING LISTENER EMAILS"
DEFAULT_EMAIL = "submissions@lukeattheroost.com"
DEFAULT_PARAGRAPHS = [
    "Got a story? A question? A hot take that\u2019s been eating at you since midnight? A confession you need to get off your chest? Send it to the show.",
    "The best listener emails get read live on air during the next episode \u2014 either by Luke himself on the mic, or by one of his robot friends. Your words, on the show, heard by everyone tuning in.",
    "Can\u2019t call 208-439-LUKE at 2 AM? Don\u2019t want to talk on the phone? Now you\u2019ve got another way to be part of the conversation. Write in anytime \u2014 day or night, long or short, serious or unhinged.",
]


def main():
    parser = argparse.ArgumentParser(description="Generate social media images")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--body", help="Text file with paragraphs (blank-line separated)")
    parser.add_argument("--prefix", default="email_announcement",
                        help="Output filename prefix")
    args = parser.parse_args()

    if args.body:
        with open(args.body) as f:
            paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
    else:
        paragraphs = DEFAULT_PARAGRAPHS

    make_square(args.title, paragraphs, args.email,
                filename=f"{args.prefix}_square.png")
    make_landscape(args.title, paragraphs, args.email,
                   filename=f"{args.prefix}_twitter.png")


if __name__ == "__main__":
    main()
