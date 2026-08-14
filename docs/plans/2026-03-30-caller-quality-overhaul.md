# Caller Quality Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every caller interesting, layered, and conversationally driven — with specific details to reveal when pressed, strong opinions, and stories that sustain 10+ exchanges without going flat.

**Architecture:** Four interconnected changes: (1) LLM-generated updates for returning callers, (2) hidden layers in CallerBackground, (3) rebalanced caller prompt from reactive to driven, (4) massively expanded topic pools with more edgy/interesting/specific content. The alien abduction call (Bev, ep45) is the gold standard — professional expertise, progressive reveals, specific details, caller-driven energy.

**Tech Stack:** Python, OpenRouter LLM API, existing CallerBackground dataclass

---

## Task 1: Add Hidden Layers to CallerBackground

The core structural change. Callers currently get ONE situation paragraph. When the host digs in, there's nothing underneath. Add 3 fields to CallerBackground that give the caller ammunition for deeper conversations.

**Files:**
- Modify: `backend/main.py` — CallerBackground dataclass (~line 6140), LLM background generation prompt (~line 5310)

**Step 1: Add fields to CallerBackground dataclass**

Find the CallerBackground dataclass and add three new fields:

```python
hidden_layers: list[str] = field(default_factory=list)  # 3 details they haven't mentioned yet — the juicy stuff underneath
burning_opinion: str = ""  # Something they're dying to say — will bring up even without being asked
stakes: str = ""  # What's at risk for them — why this matters, what happens if nothing changes
```

**Step 2: Update the LLM background generation prompt**

In the JSON output spec of the background generation prompt (~line 5322), add:

```
- "hidden_layers": A list of exactly 3 specific details the caller HASN'T mentioned yet but will reveal when pressed. These are the layers underneath the surface story. Think: the part they're embarrassed about, the complication they haven't admitted, the thing that happened AFTER the main event, the detail that changes everything. Each should be 1-2 sentences and SPECIFIC enough to sustain a follow-up question. Example: if the surface story is "my neighbor stole my mail" — layer 1 might be "the stolen mail included a paternity test result," layer 2 might be "the neighbor is actually her ex-husband's new girlfriend," layer 3 might be "she's been retaliating by feeding the neighbor's cat so it likes her better."
- "burning_opinion": ONE thing this caller is dying to say — a strong opinion, a controversial take, something they'll volunteer even without being asked. This is what makes them INTERESTING to talk to. Not a generic feeling ("I'm frustrated") but a specific, arguable position ("I think what she did was right and I'd do it again"). Make it provocative enough that the host would want to push back.
- "stakes": What's at risk for this caller. Why does this matter? What happens if nothing changes? "My sister won't talk to me" is weak. "If I don't fix this by Thursday, my sister is telling our parents about the money I took and I'll get cut out of the will" is strong. Real consequences, real deadlines, real pressure.
```

**Step 3: Update the prompt's "WHAT MAKES A GOOD CALLER" section**

Add to the existing guidance:

```
DEPTH TEST: Before finalizing, ask yourself — if the host asks "tell me more about that" THREE times, does the caller have three genuinely new, interesting things to reveal? If not, the story is too shallow. Add complications, secrets, or consequences that create layers.
```

**Step 4: Thread hidden_layers into the caller system prompt**

In `get_caller_prompt()` (~line 6015), add a section that feeds hidden layers to the caller:

```python
# Hidden layers — details to reveal when pressed
layers_block = ""
if caller.get('hidden_layers'):
    layers = caller['hidden_layers']
    layers_block = f"""
DETAILS YOU HAVEN'T MENTIONED YET (reveal these naturally when Luke asks follow-up questions or digs deeper — don't dump them all at once, let them come out one at a time as the conversation develops):
- {layers[0] if len(layers) > 0 else ''}
- {layers[1] if len(layers) > 1 else ''}
- {layers[2] if len(layers) > 2 else ''}
"""

# Burning opinion — something they're dying to say
opinion_block = ""
if caller.get('burning_opinion'):
    opinion_block = f"""
SOMETHING YOU'RE DYING TO SAY: {caller['burning_opinion']}
You'll bring this up when there's a natural opening — you don't need to be asked. This is YOUR call and you have a POINT to make.
"""

# Stakes — why this matters
stakes_block = ""
if caller.get('stakes'):
    stakes_block = f"""
WHAT'S AT STAKE: {caller['stakes']}
This isn't abstract — there are real consequences. Mention this when it's relevant. It's why you're calling NOW instead of just thinking about it.
"""
```

Insert these blocks into the prompt after `{story_block}`.

**Step 5: Commit**

---

## Task 2: LLM-Generated Updates for Returning Callers

Currently `_generate_returning_caller_background()` gives returning callers their old summaries and a one-line "something has changed." The dialog model has to improvise an update from nothing. Fix this by using the LLM to generate a specific, interesting new development.

**Files:**
- Modify: `backend/main.py` — `_generate_returning_caller_background()` (~line 4722)

**Step 1: Add LLM story continuation to returning caller background**

After building `prev_section` with call history (~line 4764), add an async LLM call to generate a new development:

```python
# Generate a specific new development via LLM
update_prompt = f"""A returning caller on a late-night radio show is calling back. Here's their history:

NAME: {regular['name']}, {age}, {gender}
JOB: {job}
PREVIOUS CALLS:
{chr(10).join(f'- {c["summary"]}' for c in prev_calls[-3:])}

Generate a SPECIFIC new development in their ongoing situation. Something has changed, escalated, or taken an unexpected turn since their last call. This should be:
- Surprising but believable — a natural next chapter, not a soap opera twist
- Specific with names, details, and concrete events
- Interesting enough to sustain a 5-10 minute conversation
- Connected to their previous calls but moving the story FORWARD

Also generate 3 hidden layers (details they'll reveal when pressed) and a burning opinion (something they're dying to say about how things have developed).

Respond with JSON:
{{
    "new_development": "2-3 sentences describing what happened since their last call",
    "hidden_layers": ["detail 1", "detail 2", "detail 3"],
    "burning_opinion": "their strong take on the situation now"
}}

Output ONLY valid JSON, no markdown fences."""
```

Make this an async function. Call the LLM, parse the JSON, and inject the new_development into `prev_section` and the hidden_layers/burning_opinion into the caller's background data.

**Step 2: Update the returning caller story_block in get_caller_prompt()**

Change the returning caller `story_block` (~line 6056) from the passive "something has developed" to:

```python
story_block = f"""YOUR STORY: You're calling back about your ongoing situation. Here's what's NEW since your last call — this is why you're calling tonight:

{{new_development}}

You have SPECIFIC things to talk about. Don't just vaguely reference "things have changed" — tell Luke EXACTLY what happened. You're calling because this new development is significant and you need to talk it through. You have details, you have feelings about it, and you have a point you want to make.

When Luke asks about your previous calls, give him the quick version — then get to what's NEW. The update is why you're here tonight."""
```

**Step 3: Commit**

---

## Task 3: Rebalance Caller Prompt — Driven, Not Passive

The current prompt over-emphasizes reactivity ("GO WHERE THE HOST TAKES YOU", "Let him drive"). This makes callers passive. Real compelling callers have their own agenda AND respond to the host.

**Files:**
- Modify: `backend/main.py` — `get_caller_prompt()` (~line 6015)

**Step 1: Rewrite the "GO WHERE THE HOST TAKES YOU" section**

Replace the current passive framing (~line 6094) with a balanced version:

```
GO WITH THE HOST BUT BRING YOUR OWN ENERGY. When Luke pushes you in a direction, challenges you, calls you out, or plays devil's advocate — engage with it. Don't shut down, don't deflect. If he says "but isn't that really about your dad?" — sit with that. BUT you're not a passive interview subject. You called because you have something to SAY. Between his questions, volunteer details he didn't ask for. Share the part you're embarrassed about. Drop the detail that changes everything. Push back when you disagree. Ask HIM what he thinks. The best callers are the ones who give the host material AND have their own momentum. You're not here to answer questions — you're here to have a CONVERSATION.
```

**Step 2: Update the "REACT TO LUKE" section**

Change from pure reactivity (~line 6096) to a balance:

```
REACT TO LUKE — BUT KEEP YOUR MOMENTUM: Your first sentence should respond to what Luke just said. But your SECOND sentence should add something new — a detail he didn't ask for, a complication, a related story, your opinion. Don't just answer and stop. Answer, then GIVE HIM MORE. If he asks "what happened next?" — don't just tell him what happened next. Tell him what happened next AND how it made you feel AND the part you haven't told anyone yet. Fill the space. Dead air is your enemy.
```

**Step 3: Add a "WHEN LUKE ASKS FOR DETAILS" section**

Add a new section after the REACT TO LUKE block:

```
WHEN LUKE ASKS FOR DETAILS — DELIVER. If Luke asks "tell me more about that" or "what do you mean?" or pushes for specifics — this is your moment. Don't give a vague one-sentence answer. Paint the picture. Who was there? What did they actually say? What were you doing when it happened? What did the room look like? What was going through your head? Specifics are what make a call memorable. "She was mad" is boring. "She threw her drink at the wall and said 'I knew you'd do this, you're just like your father'" is radio gold. ALWAYS have a specific answer ready — if Luke is digging, it means he's interested. Reward that interest with detail.
```

**Step 4: Commit**

---

## Task 4: Massively Expand Topic Pools

The current pools have good coverage but need more entries in the categories that produce the best calls: morally complex situations, sex/relationship drama, genuinely weird experiences, and callers with real questions about their lives. The alien abduction call worked because it was SPECIFIC, WEIRD, and the caller had EXPERTISE.

**Files:**
- Modify: `backend/main.py` — PROBLEMS, STORIES, WEIRD, ADVICE, GOSSIP, TOPIC_CALLIN pools

**Step 1: Add 80+ new PROBLEMS entries**

Focus on: moral dilemmas with no clear right answer, sex/relationship situations that are messy and specific, workplace drama with real stakes, family situations with complications. Every entry should pass the "tell me more" test — can you ask 3 follow-up questions and get interesting answers?

Categories to emphasize:
- **Moral dilemmas**: situations where both sides have a point, where doing the "right" thing has real costs
- **Sex/relationship mess**: not generic "my partner cheated" but specific, awkward, funny situations with details
- **Money/ethics**: found money, inheritance drama, business partner disputes, discovered fraud
- **Secrets discovered**: found out something they shouldn't know, now they have to decide what to do with it
- **Professional expertise callers**: people calling about weird things they encountered AT WORK (like Bev the nurse with the alien patient) — medical, legal, construction, teaching, law enforcement perspectives

**Step 2: Add 60+ new WEIRD entries**

The WEIRD pool produces the best calls when it's specific enough. Focus on:
- Genuinely unexplainable personal experiences (not just "something spooky happened")
- Bizarre neighbor/coworker behavior with specific ongoing patterns
- Objects/places that don't behave normally
- Coincidences too specific to be coincidence
- Things they've noticed that nobody else seems to see
- Callers who have developed elaborate theories about everyday phenomena

**Step 3: Add 60+ new STORIES entries**

Focus on stories where the caller was INVOLVED, not just an observer:
- Times they did something they can't believe they did
- Situations that escalated beyond all reason
- Encounters with strangers that changed their perspective
- Times they were absolutely, completely wrong about something
- Situations where they were the bad guy and know it
- Things they got away with that they probably shouldn't have

**Step 4: Add 40+ new ADVICE entries**

Real questions people would actually call a radio show about:
- "Am I wrong for..." situations with genuine ambiguity
- Situations where they've already decided but want validation
- Timing/approach questions ("how do I tell my wife...")
- Callers who are about to do something and want a gut check
- Callers asking about the host's opinion on something specific and controversial

**Step 5: Add 40+ new GOSSIP entries**

Gossip works best when the caller has DETAILS and opinions:
- Discovered something about someone in their life that changes everything
- Workplace gossip with real consequences if it gets out
- Small-town drama with escalating stakes
- Things overheard that they probably shouldn't have heard

**Step 6: Commit**

---

## Task 5: Increase Response Budget for Substantive Answers

The current budget gives 15% of standard calls only 450 tokens / 3 sentences. When a caller has a great story, 3 sentences isn't enough. Shift the distribution to give callers more room to breathe, especially early in the call.

**Files:**
- Modify: `backend/main.py` — `_pick_response_budget()` (~line 8280)

**Step 1: Adjust default response budget distribution**

```python
# Default distribution — give callers room to tell their story
roll = random.random()
if roll < 0.10:
    return 500, 4   # 10% — quick response (was 15% / 3 sentences)
elif roll < 0.35:
    return 600, 5   # 25% — normal conversation (was 30% / 4 sentences)
elif roll < 0.65:
    return 700, 6   # 30% — room to breathe (was 30% / 5 sentences)
else:
    return 800, 7   # 35% — telling a story or riffing (was 25% / 6 sentences)
```

Also bump up shape-specific budgets proportionally.

**Step 2: Increase MIN_RESPONSE_WORDS**

Change from 20 to 30:
```python
MIN_RESPONSE_WORDS = 30  # Retry if response is shorter than this
```

**Step 3: Commit**

---

## Task 6: Quality Review and Integration Test

Read through all changes, verify they integrate cleanly, and test with a simulated caller generation.

**Files:**
- Read: all modified files

**Step 1: Verify CallerBackground field additions parse correctly**

Run a test background generation to make sure the new fields (hidden_layers, burning_opinion, stakes) are populated by the LLM and parsed into the dataclass.

**Step 2: Verify returning caller LLM update generates properly**

Test with an existing regular (e.g., Shonda) to confirm the LLM generates a specific new development.

**Step 3: Verify the new prompt sections render correctly in get_caller_prompt()**

Check that hidden_layers, burning_opinion, and stakes blocks appear in the system prompt for a caller that has them.

**Step 4: Restart server and verify no import/syntax errors**

```bash
pkill -f "uvicorn backend.main:app"
/Users/lukemacneil/code/ai-podcast/venv/bin/python -m uvicorn backend.main:app --reload --reload-dir backend --host 0.0.0.0 --port 8000
```

**Step 5: Commit**
