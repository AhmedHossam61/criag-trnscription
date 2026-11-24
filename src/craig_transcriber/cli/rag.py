"""
`scribe rag` -- the archive from the terminal.

Usage:
    scribe rag ingest                 # index new or changed files in data/documents + transcripts
    scribe rag ingest --rebuild       # re-embed everything
    scribe rag status                 # what is indexed, which models
    scribe rag ask "What does Ritual Adept do?"
    scribe rag ask --mode multi "Compare the cult and the guardians"
    scribe rag search "celestial shards"   # retrieval only, no LLM call
    scribe rag read data/documents/characters/Kael.pdf   # the text it gets from a file
    scribe rag classify "..."         # one-step or multi-step, and how sure
    scribe rag router-eval            # how well the router does on its examples
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from ..rag.config import DEFAULT_CONFIG, load_config


def _engine(args, *, need_llm: bool = True):
    from ..rag.engine import RagEngine
    config = load_config(args.config)
    kwargs = {}
    if args.no_rerank:
        kwargs["use_reranker"] = False
    if not need_llm:
        from ..llm.base import ProviderChain, LLMProvider
        kwargs["llm"] = ProviderChain([LLMProvider()])
    return RagEngine(config, **kwargs)


def print_sources(sources: list[dict]) -> None:
    for s in sources:
        where = " · ".join(x for x in (s["location"], s["heading"]) if x)
        score = f"rerank {s['rerank']:.2f}" if s["rerank"] is not None else f"score {s['score']:.3f}"
        print(f"  [{s['n']}] {s['title']}{' · ' + where if where else ''}   ({score})")
        snippet = " ".join(s["text"].split())[:220]
        print(textwrap.indent(textwrap.fill(snippet, 96), "      "))


def cmd_ingest(args) -> None:
    engine = _engine(args, need_llm=False)
    print(f"Indexing with {engine.embedder.name} ...")
    report = engine.reindex(rebuild=args.rebuild)
    print(f"\nDone: {report.summary()}")
    for doc, err in report.failed.items():
        print(f"  ! {doc}: {err}")


def cmd_status(args) -> None:
    engine = _engine(args)
    s = engine.status()
    print(f"Documents   {s['documents']}  ({', '.join(f'{v} {k}' for k, v in s['kinds'].items()) or 'none'})")
    print(f"Passages    {s['passages']}")
    print(f"Embedder    {s['embedding_model']}" + ("   ! index built with " + s['index_model'] +
                                                     " -- run scribe rag ingest" if s["index_stale"] else ""))
    print(f"Reranker    {s['reranker'] or 'off'}")
    print(f"Router      {s['router']}")
    print(f"LLM         {s['llm']}")
    for doc in engine.documents():
        print(f"  {doc['kind']:<10} {doc['passages']:>5} passages  {doc['path']}")


def cmd_ask(args) -> None:
    engine = _engine(args)
    question = " ".join(args.question)
    answered = False
    for event in engine.ask(question, mode=args.mode):
        kind = event["type"]
        if kind == "route":
            print(f"Route: {event['mode']}-step (P(multi) = {event['probability']:.2f}, by {event['by']})")
        elif kind == "plan":
            print(f"Search {event['hop']} ({event['planner']}): " + " | ".join(event["queries"]))
        elif kind == "sources":
            print("\nSources:")
            print_sources(event["sources"])
            print()
        elif kind == "token":
            print(event["text"], end="", flush=True)
            answered = True
        elif kind == "done":
            t = event["timing"]
            print(f"\n\n({event['model']}; search {t['search_s']}s, answer {t['answer_s']}s)")
        elif kind == "error":
            print(("\n" if answered else "") + f"error: {event['message']}", file=sys.stderr)


def cmd_search(args) -> None:
    engine = _engine(args, need_llm=False)
    hits = engine.retriever().search(" ".join(args.query), args.k)
    print_sources([h.as_source(n) for n, h in enumerate(hits, start=1)])


def cmd_read(args) -> None:
    from ..rag.documents import DOCUMENT_TYPES, SourceFile, read_sections
    for path in args.files:
        kind = "transcript" if path.name == "transcript.json" else DOCUMENT_TYPES.get(path.suffix.lower())
        if not kind or not path.is_file():
            print(f"{path}: not a file the archive reads ({', '.join(sorted(DOCUMENT_TYPES))})")
            continue
        sections = read_sections(SourceFile(str(path), path, kind, path.stem))
        words = sum(len(s.text.split()) for s in sections)
        print(f"=== {path}  ({len(sections)} sections, {words} words)")
        for s in sections:
            print(f"\n--- {' · '.join(x for x in (s.heading, s.location) if x) or '(section)'}")
            print(s.text.strip() or "(no text: a scanned page needs OCR)")
        print()


def cmd_classify(args) -> None:
    engine = _engine(args, need_llm=False)
    route = engine.classify(" ".join(args.question))
    print(f"{route.mode}-step   P(multi) = {route.probability:.2f}   (by {route.by})")


def cmd_router_eval(args) -> None:
    engine = _engine(args, need_llm=False)
    result = engine.router.evaluate()
    print(f"{result['examples']} examples, {result['multi_share']:.0%} multi-step")
    print(f"Classifier (5-fold cross-validation): {result['classifier_accuracy']:.0%} correct")
    print(f"Keyword rules alone:                  {result['heuristic_accuracy']:.0%} correct")
    if result["mistakes"]:
        print("\nClassifier mistakes (question, true label, P(multi)):")
        for q, label, p in result["mistakes"]:
            print(f"  {p:.2f}  {label:<6}  {q}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scribe rag", description="The question-answering archive.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("Usage:")[1])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="settings file (default config/rag.toml)")
    parser.add_argument("--no-rerank", action="store_true", help="skip the reranker for this run")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index new and changed files")
    p.add_argument("--rebuild", action="store_true", help="re-embed every file")
    p.set_defaults(func=cmd_ingest)

    sub.add_parser("status", help="what is indexed").set_defaults(func=cmd_status)

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question", nargs="+")
    p.add_argument("--mode", choices=["auto", "single", "multi"], default="auto")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("search", help="show what retrieval finds, without asking the LLM")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", type=int, default=8)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("read", help="print the text the archive gets from a file (no models)")
    p.add_argument("files", nargs="+", type=Path)
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("classify", help="one-step or multi-step?")
    p.add_argument("question", nargs="+")
    p.set_defaults(func=cmd_classify)

    sub.add_parser("router-eval", help="cross-validate the router on its examples").set_defaults(
        func=cmd_router_eval)

    args = parser.parse_args(argv)
    args.func(args)
