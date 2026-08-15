# The Wellspring Callers — Design

**Date:** 2026-08-15
**Status:** Approved, not yet implemented

## Problem

Silas is the show's strongest recurring character, but he is the only window into
The Wellspring — and he controls the framing entirely. He describes a warm
commune of seekers. Nobody contradicts him. The bit is funny but static, and the
cult has no weight because there are no stakes the audience can see.

## Goal

Add callers from inside The Wellspring who offer perspectives Silas does not
control — including frightened ones — so the commune becomes a place with real
consequences rather than a running gag. Keep Silas playable as comedy.

## Structure

An **anchor** character with a progressing arc, plus a **pool** of rotating
members for fresh angles.

- The anchor carries continuity and escalating dread.
- The pool supplies variety and prevents the counter-perspective from
  flattening into one note.
- Silas remains the leader's version of events.

## The anchor: Doyle

```yaml
name: Doyle
voice: Grant        # plain, tight — deliberate contrast with Silas's Sebastian
age: 47
group: wellspring
register: dry, understated, factual
explicitness: low
```

Two years in. Joined eight months after his divorce, when a member sold him soap
at a farmers market in Las Cruces and asked him a question nobody had asked him
in a year. He runs the soap kettles — twelve-hour days, physical, unglamorous.

The soap room is the engine of the character: **it is where the money is, and
nobody spiritual pays attention to it.** He sees shipping manifests, deposit
slips, whose names are on which accounts. He is not a seeker. He is the guy who
fixes the pump.

**His fear is specific.** He is not afraid of being hurt. He is afraid of being
right. He has a truck, a CDL, and nowhere to be — he could leave tomorrow, and
has been telling himself that for eleven months. What frightens him is that he
keeps not leaving and can no longer tell whether that is his choice.

**Calling mechanic.** No signal at the Terlingua property. Doyle calls from town
on supply runs — a gas station lot in Study Butte, engine running, twenty
minutes before someone notices the truck has been gone too long. This caps call
length naturally, explains why he cannot talk long, and gives the host a ticking
clock. An abrupt hangup is in character, not a failure.

**Register.** He describes appalling things in the flat voice of a man reading a
parts list. The comedy and the dread come from the same place: he does not
editorialize.

## The pool

Six members, each an ordinary regular file, drawn one at a time.

| Member | Voice | Age | Register | Explicitness |
|---|---|---|---|---|
| Renata — true believer | Celeste | 34 | Radiantly happy, evangelical, funny | high |
| Alvin — bookkeeper | Conrad | 61 | Dry, precise, funny-then-cold | low |
| Junie — grew up there | Pippa | 19 | Guileless, no frame of reference | low |
| Ford — fourteen days in | Levi | 28 | Love-bombed, evangelical, too fast | medium |
| Marguerite — wants out | Marlene | 58 | Exhausted, practical, sad | medium |
| Cyrus — inner circle | Damon | 41 | Charming, then not | medium |

**Junie is a fixed constraint.** Someone who grew up inside the commune is the
show's darkest available note, and she works precisely because nothing explicit
ever attaches to her. Her horror is that she believes all of it is normal. Her
`explicitness: low` is not a default — it does not change.

## Tone

Mostly unsettling-but-comic, with the capacity to go genuinely dark. The
Wellspring should carry real weight without every call being heavy. Register is
assigned per character rather than globally, so the true believer plays funny in
the same season Doyle plays frightening.

**Adult content.** The show is adult and the Wellspring's sexuality is already
load-bearing in canon. Characters may talk frankly about sex and kink. This is
controlled per character via `explicitness: low | medium | high` rather than a
global switch, so explicitness varies by who called and the show does not
flatten into wall-to-wall smut.

**The one hard separation: consensual weirdness and coercion stay in different
lanes.** An openly horny commune with elaborate rituals is the comedy. Silas
pressuring a member into participating is the horror — that is where the weight
comes from. Blurring them casually makes the show mean rather than dark and
makes Silas irredeemable, which costs the character. Existing canon already
handles this correctly: his 2026-03-15 reckoning treats the coercion as real
harm.

## Selection mechanics

Current behavior (`backend/main.py`, `_build_backgrounds`): every active regular
draws independently at `min(1.0, 0.4 + 0.3 × shows_missed)`, then the list is
truncated to 3.

Adding eight Wellspring files to that would let the commune eat every regular
slot. So:

1. Partition regulars by the new `group` frontmatter field.
2. Ungrouped regulars behave exactly as today.
3. Within `group: wellspring`, members draw on their existing per-member
   probability, then the group is **capped**:
   - **1** Wellspring caller per episode normally
   - **2** on a 15% roll — a "both sides tonight" episode
   - On a two-night, prefer Silas plus one non-Silas member
4. Group cap applies before the global 3-regular truncation.

## Data flow

```
~/code/dotfiles/regulars/*.md   (+ silas/silas.md)
    ↓ regulars_v2.load_all_active_regulars()
    ↓   Regular gains: group, register, explicitness
    ↓ _build_backgrounds()  — group partition + cap
    ↓ regulars_included
    ↓ caller_gen.build_batch_prompt()  — renders register + explicitness
    ↓ Sonnet 4.6 batch
    ↓ session.caller_backgrounds
    ↓ get_caller_prompt()  — live dialog
```

Touch points:

- `regulars_v2.py:29` `load_regular()` — parse four new frontmatter fields
  (`group`, `group_lead`, `register`, `explicitness`), defaulting so existing
  files keep working
- `regulars_v2.py:19` `Regular` dataclass — four new fields
- `main.py` `_build_backgrounds()` — group partition and cap
- `caller_gen.py:132-138` — render `register` and `explicitness` into the batch
  prompt (currently renders only name, lore, arc_state)
- Eight new markdown files; one frontmatter line added to `silas.md`

## Failure mode to guard

Model refusals do not look like errors. `llm.py:385-395` handles exceptions and
empty completions by returning `None` with a printed message, but a content
refusal typically arrives as a **200 with refusal text**, which flows through as
ordinary dialog and breaks character on air — or as an empty completion that
reads as dead air.

Add explicit detection so a refusal is visibly distinct from a caller who simply
had nothing to say, surfaced in the host's log rather than only stdout. Without
this, tuning `explicitness: high` upward has no feedback signal.

## Testing

- `load_regular()` parses the new fields; files lacking them still load
- Group cap holds: 8 Wellspring regulars never yield more than 2 in a lineup
- Two-night rate approximates 15% over many seeded draws
- On a two-night, Silas is preferred plus exactly one non-Silas member
- Ungrouped regulars are unaffected by group logic
- Batch prompt contains `register` and `explicitness` for each included regular
- Junie's `explicitness` is `low` (guards against a careless edit)

## Out of scope

- Arc progression automation — arc_state stays hand-edited, as today
- Promotion of pool members to anchors
- Any change to how walk-in callers are generated
