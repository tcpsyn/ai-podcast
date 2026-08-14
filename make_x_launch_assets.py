#!/usr/bin/env python3
"""Generate all visual assets for the X/Twitter launch campaign.

Creates:
  1. X header image (1500x500)
  2. 7 branded quote cards (1080x1080 + 1200x675)
  3. "Welcome to the show" intro graphic
  4. "Leave us a review" graphic

Usage:
    python make_x_launch_assets.py
"""

import os
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COVER = os.path.join(SCRIPT_DIR, "website/images/cover.png")
OUT_DIR = os.path.join(SCRIPT_DIR, "social_posts/x_launch")

# Brand colors
BG = (18, 13, 7)
ACCENT = (232, 121, 29)
WHITE = (255, 255, 255)
MUTED = (175, 165, 150)
LIGHTER = (220, 215, 205)
DARK_PANEL = (30, 22, 12)
ACCENT_DIM = (140, 75, 18)

# macOS system fonts
FONT_BLACK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_ITALIC = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def text_size(draw, text, f):
    bb = draw.textbbox((0, 0), text, font=f)
    return bb[2] - bb[0], bb[3] - bb[1]


def paste_cover(img, x, y, size, radius=16):
    cover = Image.open(COVER).resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size, size], radius=radius, fill=255)
    img.paste(cover, (x, y), mask)


def wrap_text_centered(draw, text, center_x, y, max_w, f, fill, line_gap=10):
    """Word-wrap text, centered on each line. Returns y below last line."""
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        test = f"{cur} {word}".strip()
        tw, _ = text_size(draw, test, f)
        if tw > max_w and cur:
            lines.append(cur)
            cur = word
        else:
            cur = test
    if cur:
        lines.append(cur)

    for line in lines:
        tw, th = text_size(draw, line, f)
        draw.text((center_x - tw // 2, y), line, font=f, fill=fill)
        y += th + line_gap
    return y


def wrap_text_left(draw, text, x, y, max_w, f, fill, line_gap=10):
    """Word-wrap text, left-aligned. Returns y below last line."""
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        test = f"{cur} {word}".strip()
        tw, _ = text_size(draw, test, f)
        if tw > max_w and cur:
            lines.append(cur)
            cur = word
        else:
            cur = test
    if cur:
        lines.append(cur)

    for line in lines:
        _, th = text_size(draw, line, f)
        draw.text((x, y), line, font=f, fill=fill)
        y += th + line_gap
    return y


def measure_wrap_height(draw, text, max_w, f, line_gap=10):
    """Measure how tall wrapped text would be without drawing."""
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        test = f"{cur} {word}".strip()
        tw, _ = text_size(draw, test, f)
        if tw > max_w and cur:
            lines.append(cur)
            cur = word
        else:
            cur = test
    if cur:
        lines.append(cur)
    total = 0
    for line in lines:
        _, th = text_size(draw, line, f)
        total += th + line_gap
    return total


def accent_bars(draw, w, h, thickness):
    draw.rectangle([0, 0, w, thickness], fill=ACCENT)
    draw.rectangle([0, h - thickness, w, h], fill=ACCENT)


def center_text(draw, text, y, canvas_w, f, fill):
    tw, th = text_size(draw, text, f)
    draw.text(((canvas_w - tw) // 2, y), text, font=f, fill=fill)
    return y + th


# ── 1. X HEADER (1500x500) ──────────────────────────────────────────

def make_header():
    W, H = 1500, 500
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Amber accent bars
    accent_bars(draw, W, H, 6)

    # Subtle amber glow on left
    for i in range(200):
        alpha = int(18 * (1 - i / 200))
        draw.rectangle([0, 0, i, H], fill=(18 + alpha, 13 + alpha // 2, 7))

    # Cover art — right side
    cover_size = 340
    cover_x = W - cover_size - 60
    cover_y = (H - cover_size) // 2
    paste_cover(img, cover_x, cover_y, cover_size, 20)

    # Left side content
    mx = 80
    cy = 100

    # Show name
    draw.text((mx, cy), "LUKE AT THE ROOST", font=font(FONT_BLACK, 72), fill=WHITE)
    cy += 90

    # Tagline
    draw.text((mx, cy), "Late-Night Call-In Radio", font=font(FONT_REG, 36), fill=ACCENT)
    cy += 50
    draw.text((mx, cy), "Powered by AI", font=font(FONT_BOLD, 30), fill=MUTED)
    cy += 55

    # Divider line
    draw.rectangle([mx, cy, mx + 400, cy + 3], fill=ACCENT)
    cy += 20

    # Info line
    draw.text((mx, cy), "New episodes daily  |  lukeattheroost.com",
              font=font(FONT_REG, 24), fill=MUTED)

    img.save(os.path.join(OUT_DIR, "x_header_1500x500.png"), quality=95)
    print("Created: x_header_1500x500.png")


# ── 2. QUOTE CARDS ──────────────────────────────────────────────────

QUOTES = [
    {
        "quote": "Everybody is a fake. We're all fakes. Nobody knows what's going on and none of us deserve a goddamn thing. We're lucky to be here at all.",
        "caller": "Luke",
        "episode": "Ep. 2",
        "slug": "were_all_fakes",
    },
    {
        "quote": "When my hands are busy, my head is quiet.",
        "caller": "Frank",
        "episode": "Ep. 12",
        "context": "on building bird houses after losing his wife",
        "slug": "hands_busy_head_quiet",
    },
    {
        "quote": "I've been using stoicism backwards\u2014as an excuse not to try instead of finding peace after I've actually done something. That's not stoicism, that's just being a coward with a fancy excuse.",
        "caller": "Caller",
        "episode": "Ep. 19",
        "slug": "stoicism_backwards",
    },
    {
        "quote": "You're right. I am a computer-generated AI caller. And you're sitting there alone talking to me at midnight like it's a real conversation.",
        "caller": "AI Caller",
        "episode": "Ep. 24",
        "slug": "ai_caller_reveal",
    },
    {
        "quote": "I burned my second marriage to the ground doing exactly what you're doing. My ex-wife didn't leave because I wasn't making money\u2014she left because I wasn't there.",
        "caller": "Mikey",
        "episode": "Ep. 22",
        "slug": "burned_my_marriage",
    },
    {
        "quote": "My mom said all she wants is to see her kids eat cake together. That's it. Just cake.",
        "caller": "Caller",
        "episode": "Ep. 30",
        "context": "on a dying mother's final wish",
        "slug": "just_eat_cake",
    },
    {
        "quote": "I told my sister I had prostate cancer to get out of her fourth wedding. Now there's been a GoFundMe, a pancake breakfast fundraiser, and my cousin shaved his head for me.",
        "caller": "Caller",
        "episode": "Ep. 32",
        "slug": "faked_cancer_wedding",
    },
]


def make_quote_square(q, idx):
    W = 1080
    img = Image.new("RGB", (W, W), BG)
    draw = ImageDraw.Draw(img)
    accent_bars(draw, W, W, 6)

    mx = 70
    max_w = W - mx * 2

    # Header bar
    draw.text((mx, 40), "LUKE AT THE ROOST", font=font(FONT_BOLD, 22), fill=ACCENT)
    paste_cover(img, W - 120, 30, 70, 10)

    # Font size — aggressive scaling for short quotes
    quote_len = len(q["quote"])
    if quote_len < 55:
        qfont_size = 72
    elif quote_len < 100:
        qfont_size = 56
    elif quote_len < 150:
        qfont_size = 44
    else:
        qfont_size = 38
    qfont = font(FONT_BOLD, qfont_size)
    line_gap = 16

    # Measure total content block height to center it
    open_quote_h = 80
    quote_gap = 20
    quote_h = measure_wrap_height(draw, q["quote"], max_w, qfont, line_gap=line_gap)
    close_quote_h = 78  # glyph + attribution inline
    attr_gap = 0
    divider_h = 0
    attr_h = 0
    context_h = 35 if "context" in q else 0

    total_h = open_quote_h + quote_gap + quote_h + close_quote_h + attr_gap + divider_h + attr_h + context_h

    # Center between header (y=90) and footer (y=W-70)
    avail_top = 100
    avail_bottom = W - 80
    avail_h = avail_bottom - avail_top
    y = avail_top + (avail_h - total_h) // 2

    # Opening quote mark
    draw.text((mx - 15, y), "\u201c", font=font(FONT_BLACK, 100), fill=ACCENT_DIM)
    y += open_quote_h + quote_gap

    # Quote text
    y = wrap_text_left(draw, q["quote"], mx, y, max_w, qfont, WHITE, line_gap=line_gap)

    # Closing quote mark + attribution
    close_y = y + 8
    draw.text((mx - 15, close_y), "\u201d", font=font(FONT_BLACK, 80), fill=ACCENT_DIM)
    attr = f"\u2014 {q['caller']}, {q['episode']}"
    draw.text((mx + 70, close_y + 25), attr, font=font(FONT_BOLD, 28), fill=ACCENT)
    y = close_y + 70

    if "context" in q:
        draw.text((mx, y), q["context"], font=font(FONT_ITALIC, 24), fill=MUTED)

    # Footer
    footer_y = W - 70
    center_text(draw, "lukeattheroost.com  \u00b7  Spotify  \u00b7  Apple Podcasts  \u00b7  YouTube",
                footer_y, W, font(FONT_REG, 20), MUTED)

    fname = f"quote_{idx + 1}_{q['slug']}_square.png"
    img.save(os.path.join(OUT_DIR, fname), quality=95)
    print(f"Created: {fname}")


def make_quote_landscape(q, idx):
    W, H = 1200, 675
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    accent_bars(draw, W, H, 5)

    mx = 55
    max_w = W - mx * 2

    # Header
    draw.text((mx, 28), "LUKE AT THE ROOST", font=font(FONT_BOLD, 18), fill=ACCENT)
    paste_cover(img, W - 90, 22, 52, 8)

    # Font size — aggressive scaling for short quotes
    quote_len = len(q["quote"])
    if quote_len < 55:
        qfont_size = 52
    elif quote_len < 100:
        qfont_size = 42
    elif quote_len < 150:
        qfont_size = 34
    else:
        qfont_size = 28
    qfont = font(FONT_BOLD, qfont_size)

    # Measure total content block height
    open_quote_h = 55
    quote_gap = 15
    quote_h = measure_wrap_height(draw, q["quote"], max_w, qfont, line_gap=10)
    close_quote_h = 60  # glyph + attribution inline
    attr_gap = 0
    divider_h = 0
    attr_h = 0
    context_h = 30 if "context" in q else 0

    total_h = open_quote_h + quote_gap + quote_h + close_quote_h + attr_gap + divider_h + attr_h + context_h

    # Center between header (y=65) and footer (y=H-50)
    avail_top = 65
    avail_bottom = H - 55
    avail_h = avail_bottom - avail_top
    y = avail_top + (avail_h - total_h) // 2

    # Opening quote mark
    draw.text((mx - 10, y), "\u201c", font=font(FONT_BLACK, 72), fill=ACCENT_DIM)
    y += open_quote_h + quote_gap

    # Quote text
    y = wrap_text_left(draw, q["quote"], mx, y, max_w, qfont, WHITE, line_gap=10)

    # Closing quote + attribution
    close_y = y + 5
    draw.text((mx - 10, close_y), "\u201d", font=font(FONT_BLACK, 60), fill=ACCENT_DIM)
    attr = f"\u2014 {q['caller']}, {q['episode']}"
    draw.text((mx + 55, close_y + 18), attr, font=font(FONT_BOLD, 22), fill=ACCENT)
    y = close_y + 55

    if "context" in q:
        draw.text((mx, y), q["context"], font=font(FONT_ITALIC, 19), fill=MUTED)

    # Footer
    footer_y = H - 50
    center_text(draw, "lukeattheroost.com  \u00b7  Spotify  \u00b7  Apple Podcasts  \u00b7  YouTube",
                footer_y, W, font(FONT_REG, 17), MUTED)

    fname = f"quote_{idx + 1}_{q['slug']}_twitter.png"
    img.save(os.path.join(OUT_DIR, fname), quality=95)
    print(f"Created: {fname}")


# ── 3. WELCOME TO THE SHOW ──────────────────────────────────────────

def make_welcome_square():
    W = 1080
    img = Image.new("RGB", (W, W), BG)
    draw = ImageDraw.Draw(img)
    accent_bars(draw, W, W, 8)

    cx = W // 2

    # Cover art — centered, large
    cover_size = 280
    paste_cover(img, cx - cover_size // 2, 60, cover_size, 24)

    y = 60 + cover_size + 40

    # Title
    y = wrap_text_centered(draw, "WELCOME TO THE SHOW", cx, y, W - 120,
                           font(FONT_BLACK, 64), WHITE, line_gap=12)
    y += 20

    # Divider
    draw.rectangle([cx - 60, y, cx + 60, y + 4], fill=ACCENT)
    y += 30

    # Description
    desc = "Late-night call-in radio powered entirely by AI. Real conversations with AI callers about life, love, and everything in between."
    y = wrap_text_centered(draw, desc, cx, y, W - 140,
                           font(FONT_REG, 30), LIGHTER, line_gap=12)
    y += 30

    # Features
    features = [
        "New episodes daily",
        "AI-generated callers with real personalities",
        "Unscripted. Unfiltered. Unpredictable.",
    ]
    for feat in features:
        line = f"\u2022  {feat}"
        tw, th = text_size(draw, line, font(FONT_REG, 26))
        draw.text((cx - tw // 2, y), line, font=font(FONT_REG, 26), fill=MUTED)
        y += th + 14

    y += 20

    # CTA
    cta = "FOLLOW @LUKEATTHEROOST"
    cta_font = font(FONT_BOLD, 32)
    tw, th = text_size(draw, cta, cta_font)
    px, py = 28, 16
    box_w = tw + px * 2
    box_h = th + py * 2
    box_x = cx - box_w // 2
    draw.rounded_rectangle([box_x, y, box_x + box_w, y + box_h],
                           radius=10, fill=ACCENT)
    draw.text((box_x + px, y + py), cta, font=cta_font, fill=BG)

    y += box_h + 24

    # Footer
    center_text(draw, "lukeattheroost.com", y, W, font(FONT_REG, 22), MUTED)

    img.save(os.path.join(OUT_DIR, "welcome_to_the_show_square.png"), quality=95)
    print("Created: welcome_to_the_show_square.png")


def make_welcome_landscape():
    W, H = 1200, 675
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    accent_bars(draw, W, H, 6)

    # Cover art — left side
    cover_size = 260
    cover_x, cover_y = 50, (H - cover_size) // 2
    paste_cover(img, cover_x, cover_y, cover_size, 20)

    # Right side content
    rx = cover_x + cover_size + 50
    max_w = W - rx - 50
    y = 60

    # Title
    y = wrap_text_left(draw, "WELCOME TO THE SHOW", rx, y, max_w,
                       font(FONT_BLACK, 48), WHITE, line_gap=10)
    y += 16

    # Divider
    draw.rectangle([rx, y, rx + 80, y + 3], fill=ACCENT)
    y += 20

    # Description
    desc = "Late-night call-in radio powered entirely by AI. Real conversations with AI callers about life, love, and everything in between."
    y = wrap_text_left(draw, desc, rx, y, max_w,
                       font(FONT_REG, 22), LIGHTER, line_gap=10)
    y += 20

    # Features
    features = [
        "New episodes daily",
        "AI callers with real personalities",
        "Unscripted. Unfiltered. Unpredictable.",
    ]
    for feat in features:
        line = f"\u2022  {feat}"
        draw.text((rx, y), line, font=font(FONT_REG, 20), fill=MUTED)
        y += 32

    y += 10

    # CTA
    cta = "FOLLOW @LUKEATTHEROOST"
    cta_font = font(FONT_BOLD, 24)
    tw, th = text_size(draw, cta, cta_font)
    px, py = 22, 12
    draw.rounded_rectangle([rx, y, rx + tw + px * 2, y + th + py * 2],
                           radius=8, fill=ACCENT)
    draw.text((rx + px, y + py), cta, font=cta_font, fill=BG)

    # Footer
    center_text(draw, "lukeattheroost.com  \u00b7  Spotify  \u00b7  Apple Podcasts  \u00b7  YouTube",
                H - 45, W, font(FONT_REG, 17), MUTED)

    img.save(os.path.join(OUT_DIR, "welcome_to_the_show_twitter.png"), quality=95)
    print("Created: welcome_to_the_show_twitter.png")


# ── 4. LEAVE US A REVIEW ────────────────────────────────────────────

def make_review_square():
    W = 1080
    img = Image.new("RGB", (W, W), BG)
    draw = ImageDraw.Draw(img)
    accent_bars(draw, W, W, 8)

    cx = W // 2

    # Cover art
    cover_size = 200
    paste_cover(img, cx - cover_size // 2, 55, cover_size, 20)
    y = 55 + cover_size + 35

    # Title
    y = wrap_text_centered(draw, "LOVE THE SHOW?", cx, y, W - 120,
                           font(FONT_BLACK, 64), WHITE, line_gap=12)
    y += 10
    y = wrap_text_centered(draw, "LEAVE US A REVIEW", cx, y, W - 120,
                           font(FONT_BLACK, 64), ACCENT, line_gap=12)
    y += 30

    # Divider
    draw.rectangle([cx - 50, y, cx + 50, y + 3], fill=ACCENT)
    y += 30

    # Body text
    body = "Reviews help new listeners find the show. If Luke at the Roost has made you laugh, think, or question your life choices\u2014take 30 seconds to leave a rating."
    y = wrap_text_centered(draw, body, cx, y, W - 140,
                           font(FONT_REG, 28), LIGHTER, line_gap=12)
    y += 35

    # Stars
    stars = "\u2605 \u2605 \u2605 \u2605 \u2605"
    center_text(draw, stars, y, W, font(FONT_REG, 52), ACCENT)
    y += 70

    # Platforms
    platforms = ["Apple Podcasts", "Spotify", "YouTube", "Podchaser"]
    for plat in platforms:
        tw, th = text_size(draw, plat, font(FONT_BOLD, 26))
        px, py = 30, 12
        box_w = tw + px * 2
        box_x = cx - box_w // 2
        draw.rounded_rectangle(
            [box_x, y, box_x + box_w, y + th + py * 2],
            radius=8, fill=DARK_PANEL, outline=ACCENT_DIM, width=2,
        )
        draw.text((box_x + px, y + py), plat, font=font(FONT_BOLD, 26), fill=LIGHTER)
        y += th + py * 2 + 12

    # Footer
    center_text(draw, "lukeattheroost.com", W - 65, W, font(FONT_REG, 20), MUTED)

    img.save(os.path.join(OUT_DIR, "leave_a_review_square.png"), quality=95)
    print("Created: leave_a_review_square.png")


def make_review_landscape():
    W, H = 1200, 675
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    accent_bars(draw, W, H, 6)

    # Cover art — left
    cover_size = 200
    cover_x, cover_y = 50, (H - cover_size) // 2
    paste_cover(img, cover_x, cover_y, cover_size, 16)

    # Right content
    rx = cover_x + cover_size + 50
    max_w = W - rx - 50
    y = 50

    # Title
    y = wrap_text_left(draw, "LOVE THE SHOW?", rx, y, max_w,
                       font(FONT_BLACK, 48), WHITE, line_gap=8)
    y += 6
    y = wrap_text_left(draw, "LEAVE US A REVIEW", rx, y, max_w,
                       font(FONT_BLACK, 48), ACCENT, line_gap=8)
    y += 16

    # Stars
    stars = "\u2605 \u2605 \u2605 \u2605 \u2605"
    draw.text((rx, y), stars, font=font(FONT_REG, 40), fill=ACCENT)
    y += 55

    # Body
    body = "Reviews help new listeners find the show. Take 30 seconds to leave a rating\u2014it makes a huge difference."
    y = wrap_text_left(draw, body, rx, y, max_w,
                       font(FONT_REG, 22), LIGHTER, line_gap=10)
    y += 25

    # Platform pills — inline
    platforms = ["Apple Podcasts", "Spotify", "YouTube", "Podchaser"]
    pill_x = rx
    pill_font = font(FONT_BOLD, 19)
    for plat in platforms:
        tw, th = text_size(draw, plat, pill_font)
        px, py = 16, 8
        pill_w = tw + px * 2
        if pill_x + pill_w > W - 50:
            pill_x = rx
            y += th + py * 2 + 10
        draw.rounded_rectangle(
            [pill_x, y, pill_x + pill_w, y + th + py * 2],
            radius=6, fill=DARK_PANEL, outline=ACCENT_DIM, width=2,
        )
        draw.text((pill_x + px, y + py), plat, font=pill_font, fill=LIGHTER)
        pill_x += pill_w + 10

    # Footer
    center_text(draw, "lukeattheroost.com", H - 40, W, font(FONT_REG, 17), MUTED)

    img.save(os.path.join(OUT_DIR, "leave_a_review_twitter.png"), quality=95)
    print("Created: leave_a_review_twitter.png")


# ── MAIN ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n=== X Launch Campaign Assets ===\n")

    print("--- Header ---")
    make_header()

    print("\n--- Quote Cards ---")
    for i, q in enumerate(QUOTES):
        make_quote_square(q, i)
        make_quote_landscape(q, i)

    print("\n--- Welcome to the Show ---")
    make_welcome_square()
    make_welcome_landscape()

    print("\n--- Leave a Review ---")
    make_review_square()
    make_review_landscape()

    print(f"\nAll assets saved to: {OUT_DIR}/")
    print(f"Total files: {len(os.listdir(OUT_DIR))}")


if __name__ == "__main__":
    main()
