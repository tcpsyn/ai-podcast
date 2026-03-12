# Clips Page & Landing Page Redesign

## Clips Page (`/clips`)

Responsive grid gallery of podcast clips with click-to-play YouTube embeds.

**Grid:** 3 columns desktop, 2 tablet, 1 mobile. Cards use 9:16 vertical aspect ratio.

**Card pre-click:** Dark bg-light card with clip title (bold), episode label, centered orange play button, description text below. Matches site aesthetic.

**Card playing:** Click swaps card for YouTube Shorts iframe (`youtube-nocookie.com`, autoplay). Fills same 9:16 space.

**Data:** Static `website/data/clips.json` aggregated from per-episode `clips-metadata.json` files. Each entry: title, description, episode_number, optional `youtube_id`. Cards without youtube_id show no play button.

**Featured row:** Top 3 hand-picked clips displayed larger, followed by full grid below.

**Nav:** "Clips" added to hero secondary links and footer nav.

## Landing Page Improvements

**About section** (between hero and episodes): Centered text block. Show description + AI teaser line ("Part human callers, part AI-generated characters, fully unhinged advice") + "See how it works" link. No card background.

**Clips highlight** (between about and episodes): Horizontal row of 3 featured clips, same card style as clips page. "Best Clips" header with "See all clips" link.

**Final section order:** Banner → Hero → About → Featured Clips → Episodes → Testimonials → Footer

## How It Works — Reaper Video

New "Post-Production Automation" section with native `<video>` tag (mp4 on CDN). Shows Reaper automating silence removal, ad ducking, loudness normalization. Wrapped in hiw-hero-card style container.
