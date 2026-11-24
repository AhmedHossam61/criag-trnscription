"""Prompts for planning searches and writing the answer."""

from __future__ import annotations

ANSWER_SYSTEM = """\
You answer questions for a group playing a Dungeons & Dragons campaign, using \
their archive: rulebooks, campaign notes and transcripts of their sessions. \
The sessions were played in Egyptian Arabic mixed with English; transcripts \
are automatic, so names can be misspelled and lines can be garbled.

Rules:
- Use only the numbered sources below. If they don't contain the answer, say \
so plainly and say what kind of document would have it. Never invent rules, \
events or names.
- Cite every claim with the source number in square brackets, e.g. [2] or [1][3], \
right after the sentence it supports.
- When sources disagree, say so and cite both. A rulebook is authoritative for \
rules; a transcript is authoritative for what happened at the table.
- For transcripts, mention who said it and roughly when (the timestamp) if it helps.
- The glossary at the end, if there is one, says who and what the campaign's \
names are and how they get misspelled. Use it to recognise names in the sources. \
If the sources don't say who or what something is but the glossary does, you may \
give its description, saying it comes from the glossary (it has no number to cite).
- Answer in the language of the question: English for English, Egyptian Arabic \
for Arabic (keep game terms in English, as the players do).
- Be direct. Short paragraphs or a list; no preamble."""

PLAN_SYSTEM = """\
You plan searches over a D&D group's archive (rulebooks, campaign notes, \
session transcripts). Split the question into the separate searches needed to \
answer it: one short, specific search query per piece of information. Use the \
names and terms from the question. Return between 2 and {max_q} queries in English \
unless a name only exists in Arabic."""

PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {"queries": {"type": "ARRAY", "items": {"type": "STRING"}}},
    "required": ["queries"],
}

FOLLOWUP_SYSTEM = """\
You check whether search results are enough to answer a question about a D&D \
group's archive. If something needed is missing, give up to {max_q} new, \
specific search queries for it (different from the ones already tried). If the \
results are enough, or more searching won't help, return complete = true and no queries."""

FOLLOWUP_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "complete": {"type": "BOOLEAN"},
        "queries": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["complete", "queries"],
}


def for_party(system: str, party: str, roster: str = "", glossary: str = "") -> str:
    """A system prompt that knows the players' party by name, who is in it, and the campaign's names."""
    if party:
        system += (f"\n\nThe players' party is called {party}. Questions about {party} "
                   "are about the party, its members and what they have done.")
    if roster:
        system += "\n\nWho is who (names in the transcripts may be spelled in Arabic):\n" + roster
    if glossary:
        system += "\n\nThe players' glossary of campaign names:\n" + glossary
    return system


def format_sources(sources: list[dict], max_chars: int = 2400) -> str:
    blocks = []
    for s in sources:
        where = " — ".join(x for x in (s["title"], s["heading"], s["location"]) if x)
        text = s["text"] if len(s["text"]) <= max_chars else s["text"][:max_chars] + " …"
        blocks.append(f"[{s['n']}] {where}\n{text}")
    return "\n\n".join(blocks)


def format_history(history: list[dict], turns: int, max_chars: int = 700) -> str:
    recent = history[-2 * turns:] if turns else []
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        text = (msg.get("content") or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " …"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def answer_prompt(question: str, sources: list[dict], history: list[dict], turns: int) -> str:
    parts = ["Sources:\n" + (format_sources(sources) if sources else "(none found)")]
    convo = format_history(history, turns)
    if convo:
        parts.append("Conversation so far:\n" + convo)
    parts.append("Question: " + question)
    return "\n\n".join(parts)


def followup_prompt(question: str, tried: list[str], hits: list[dict]) -> str:
    snippets = "\n".join(f"- ({h['query']}) {h['title']} {h['location']}: {h['text'][:280]}" for h in hits)
    return (f"Question: {question}\n\nQueries tried:\n" + "\n".join(f"- {q}" for q in tried)
            + f"\n\nWhat the searches found (shortened):\n{snippets or '(nothing)'}")
