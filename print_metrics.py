"""
Results Table Viewer
====================
Reads results/raw_results.json and prints three formatted tables:

  Table 1 — Per-question results (one row per question, all 3 approaches)
  Table 2 — Overall accuracy / latency / tokens summary
  Table 3 — Per question-type breakdown

Usage:
    python view_results.py                        # reads results/raw_results.json
    python view_results.py path/to/raw_results.json

Optional flags:
    --no-answers      hide the answer columns (cleaner for quick scanning)
    --type <type>     filter to a specific question_type
    --failed          show only questions where at least one approach was wrong
"""

import json
import sys
import statistics
from pathlib import Path


# ── ANSI colours (disabled automatically on Windows if not supported) ──────────

def _supports_colour() -> bool:
    import os
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.name != "nt"


USE_COLOUR = _supports_colour()

GREEN  = "\033[92m" if USE_COLOUR else ""
RED    = "\033[91m" if USE_COLOUR else ""
YELLOW = "\033[93m" if USE_COLOUR else ""
BOLD   = "\033[1m"  if USE_COLOUR else ""
DIM    = "\033[2m"  if USE_COLOUR else ""
RESET  = "\033[0m"  if USE_COLOUR else ""


def tick(val: bool) -> str:
    return f"{GREEN}✓{RESET}" if val else f"{RED}✗{RESET}"


def colourise_delta(delta: float) -> str:
    if delta > 0:
        return f"{GREEN}+{delta:.1f}%{RESET}"
    elif delta < 0:
        return f"{RED}{delta:.1f}%{RESET}"
    return f"{DIM}0.0%{RESET}"


def best_marker(val: float, best: float) -> str:
    """Append a star if this value equals the best."""
    s = f"{val:.1f}%"
    if val == best:
        return f"{BOLD}{GREEN}{s} ★{RESET}"
    return s


# ── Helpers ───────────────────────────────────────────────────────────────────

def truncate(text: str, width: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def divider(char: str = "─", width: int = 120) -> str:
    return char * width


def print_header(title: str, width: int = 120) -> None:
    print()
    print(divider("═", width))
    print(f"{BOLD}{title}{RESET}".center(width + len(BOLD) + len(RESET)))
    print(divider("═", width))


# ── Table 1 — Per-question detail ─────────────────────────────────────────────

def print_per_question_table(
    results: list[dict],
    show_answers: bool = True,
) -> None:
    print_header("TABLE 1 — Per-Question Results")

    ANS_W   = 38   # answer column width
    Q_W     = 42   # question column width
    TYPE_W  = 22   # question_type column width
    ID_W    = 10   # question_id column width

    if show_answers:
        header = (
            f"{'#':<4} {'ID':<{ID_W}} {'Type':<{TYPE_W}} {'Qi':>2}  "
            f"{'Question':<{Q_W}}  "
            f"{'Qdr':>4} {'Gph':>4} {'Hyb':>4}  "
            f"{'Qdrant Answer':<{ANS_W}}  "
            f"{'Graphiti Answer':<{ANS_W}}  "
            f"{'Hybrid Answer':<{ANS_W}}  "
            f"{'Ground Truth':<{ANS_W}}"
        )
    else:
        header = (
            f"{'#':<4} {'ID':<{ID_W}} {'Type':<{TYPE_W}} {'Qi':>2}  "
            f"{'Question':<{Q_W}}  "
            f"{'Qdr':>4} {'Gph':>4} {'Hyb':>4}  "
            f"{'Qlat':>6} {'Glat':>6} {'Hlat':>6}"
        )

    print(header)
    print(divider())

    prev_qid = None
    for i, r in enumerate(results, 1):
        qid   = r.get("question_id", "")[:ID_W]
        qtype = r.get("question_type", "")[:TYPE_W]
        q_idx = r.get("question_index", 0)
        q     = truncate(r.get("question", ""), Q_W)

        qdr = tick(r.get("qdrant_correct",  False))
        gph = tick(r.get("graph_correct",   False))
        hyb = tick(r.get("hybrid_correct",  False))

        # Separator between different question_ids
        if prev_qid and prev_qid != r.get("question_id"):
            print(divider("·"))
        prev_qid = r.get("question_id")

        if show_answers:
            qa  = truncate(r.get("qdrant_answer",  ""), ANS_W)
            ga  = truncate(r.get("graph_answer",   ""), ANS_W)
            ha  = truncate(r.get("hybrid_answer",  ""), ANS_W)
            gt  = truncate(r.get("ground_truth",   ""), ANS_W)
            print(
                f"{i:<4} {qid:<{ID_W}} {qtype:<{TYPE_W}} {q_idx:>2}  "
                f"{q:<{Q_W}}  "
                f"{qdr:>4} {gph:>4} {hyb:>4}  "
                f"{qa:<{ANS_W}}  "
                f"{ga:<{ANS_W}}  "
                f"{ha:<{ANS_W}}  "
                f"{gt:<{ANS_W}}"
            )
        else:
            ql = r.get("qdrant_latency_s", 0.0)
            gl = r.get("graph_latency_s",  0.0)
            hl = r.get("hybrid_latency_s", 0.0)
            print(
                f"{i:<4} {qid:<{ID_W}} {qtype:<{TYPE_W}} {q_idx:>2}  "
                f"{q:<{Q_W}}  "
                f"{qdr:>4} {gph:>4} {hyb:>4}  "
                f"{ql:>5.1f}s {gl:>5.1f}s {hl:>5.1f}s"
            )

    print(divider())
    total = len(results)
    qdr_n = sum(1 for r in results if r.get("qdrant_correct"))
    gph_n = sum(1 for r in results if r.get("graph_correct"))
    hyb_n = sum(1 for r in results if r.get("hybrid_correct"))
    print(
        f"{BOLD}TOTALS{RESET}  {total} questions   "
        f"Qdrant: {qdr_n}/{total} ({100*qdr_n/total:.1f}%)   "
        f"Graphiti: {gph_n}/{total} ({100*gph_n/total:.1f}%)   "
        f"Hybrid: {hyb_n}/{total} ({100*hyb_n/total:.1f}%)"
    )


# ── Table 2 — Overall summary ─────────────────────────────────────────────────

def print_overall_table(results: list[dict]) -> None:
    print_header("TABLE 2 — Overall Summary")

    APPROACHES = [
        ("qdrant",  "Qdrant (Semantic)"),
        ("graph",   "Graphiti (KGraph)"),
        ("hybrid",  "Hybrid (Parallel)"),
    ]

    rows = []
    for prefix, label in APPROACHES:
        valid = [r for r in results if r.get(f"{prefix}_correct") is not None]
        if not valid:
            continue
        n         = len(valid)
        n_correct = sum(1 for r in valid if r[f"{prefix}_correct"])
        latencies = sorted(r[f"{prefix}_latency_s"] for r in valid)
        tokens    = [r[f"{prefix}_prompt_tokens"] for r in valid]

        q1 = latencies[int(n * 0.25)]
        q3 = latencies[min(int(n * 0.75), n - 1)]

        rows.append({
            "label":        label,
            "n":            n,
            "correct":      n_correct,
            "score_pct":    100 * n_correct / n,
            "mean_lat":     statistics.mean(latencies),
            "median_lat":   statistics.median(latencies),
            "iqr_lat":      q3 - q1,
            "avg_tokens":   statistics.mean(tokens),
        })

    best_score = max(r["score_pct"] for r in rows)

    col = [22, 8, 10, 10, 10, 10, 10, 12]
    hdr = (
        f"{'Approach':<{col[0]}} {'N':>{col[1]}} "
        f"{'Correct':>{col[2]}} {'Score':>{col[3]}} "
        f"{'MeanLat':>{col[4]}} {'MedLat':>{col[5]}} "
        f"{'IQR(s)':>{col[6]}} {'AvgTokens':>{col[7]}}"
    )
    print(hdr)
    print(divider())

    for r in rows:
        score_str = best_marker(r["score_pct"], best_score)
        print(
            f"{r['label']:<{col[0]}} {r['n']:>{col[1]}} "
            f"{r['correct']:>{col[2]}} {score_str:>{col[3] + 20}} "
            f"{r['mean_lat']:>{col[4]}.2f}s {r['median_lat']:>{col[5]}.2f}s "
            f"{r['iqr_lat']:>{col[6]}.3f} {r['avg_tokens']:>{col[7]}.0f}"
        )

    print(divider())

    # Delta rows
    if len(rows) == 3:
        qdrant_score  = rows[0]["score_pct"]
        graph_score   = rows[1]["score_pct"]
        hybrid_score  = rows[2]["score_pct"]
        print(
            f"  Hybrid vs Qdrant:   {colourise_delta(hybrid_score - qdrant_score)}   "
            f"Hybrid vs Graphiti: {colourise_delta(hybrid_score - graph_score)}"
        )


# ── Table 3 — Per question-type breakdown ─────────────────────────────────────

def print_type_table(results: list[dict]) -> None:
    print_header("TABLE 3 — Per Question-Type Breakdown")

    q_types = sorted({r["question_type"] for r in results})

    col = [28, 6, 10, 12, 10, 14, 16, 12]
    hdr = (
        f"{'Question Type':<{col[0]}} {'N':>{col[1]}} "
        f"{'Qdrant%':>{col[2]}} {'Graphiti%':>{col[3]}} "
        f"{'Hybrid%':>{col[4]}} {'Hyb-Qdr':>{col[5]}} "
        f"{'Hyb-Gph':>{col[6]}} {'Best':>{col[7]}}"
    )
    print(hdr)
    print(divider())

    for qt in q_types:
        subset = [r for r in results if r["question_type"] == qt]
        n = len(subset)

        def pct(key):
            return 100 * sum(1 for r in subset if r.get(key)) / n

        qd = pct("qdrant_correct")
        gr = pct("graph_correct")
        hy = pct("hybrid_correct")
        best = max(qd, gr, hy)

        best_label = (
            "Hybrid"   if hy == best
            else "Graphiti" if gr == best
            else "Qdrant"
        )

        print(
            f"{qt:<{col[0]}} {n:>{col[1]}} "
            f"{best_marker(qd, best):>{col[2] + 20}} "
            f"{best_marker(gr, best):>{col[3] + 20}} "
            f"{best_marker(hy, best):>{col[4] + 20}} "
            f"{colourise_delta(hy - qd):>{col[5] + 20}} "
            f"{colourise_delta(hy - gr):>{col[6] + 20}} "
            f"{BOLD}{best_label}{RESET}"
        )

    print(divider())


# ── Failed-only filter ────────────────────────────────────────────────────────

def filter_failed(results: list[dict]) -> list[dict]:
    """Keep only questions where at least one approach got it wrong."""
    return [
        r for r in results
        if not (
            r.get("qdrant_correct")
            and r.get("graph_correct")
            and r.get("hybrid_correct")
        )
    ]


# ── Disagreement analysis ─────────────────────────────────────────────────────

def print_disagreement_analysis(results: list[dict]) -> None:
    """
    Show cases where approaches disagree — useful for understanding
    where each method adds value over the others.
    """
    print_header("BONUS — Disagreement Analysis")

    patterns = {
        "All correct      (Q✓ G✓ H✓)": lambda r: r.get("qdrant_correct") and r.get("graph_correct") and r.get("hybrid_correct"),
        "All wrong        (Q✗ G✗ H✗)": lambda r: not r.get("qdrant_correct") and not r.get("graph_correct") and not r.get("hybrid_correct"),
        "Hybrid only      (Q✗ G✗ H✓)": lambda r: not r.get("qdrant_correct") and not r.get("graph_correct") and r.get("hybrid_correct"),
        "Qdrant only      (Q✓ G✗ H✗)": lambda r: r.get("qdrant_correct") and not r.get("graph_correct") and not r.get("hybrid_correct"),
        "Graphiti only    (Q✗ G✓ H✗)": lambda r: not r.get("qdrant_correct") and r.get("graph_correct") and not r.get("hybrid_correct"),
        "Hybrid+Qdrant    (Q✓ G✗ H✓)": lambda r: r.get("qdrant_correct") and not r.get("graph_correct") and r.get("hybrid_correct"),
        "Hybrid+Graphiti  (Q✗ G✓ H✓)": lambda r: not r.get("qdrant_correct") and r.get("graph_correct") and r.get("hybrid_correct"),
        "Qdrant+Graphiti  (Q✓ G✓ H✗)": lambda r: r.get("qdrant_correct") and r.get("graph_correct") and not r.get("hybrid_correct"),
    }

    n = len(results)
    col_w = 36
    print(f"{'Pattern':<{col_w}} {'Count':>6} {'%':>7}")
    print(divider(width=55))
    for label, fn in patterns.items():
        count = sum(1 for r in results if fn(r))
        pct   = 100 * count / n if n else 0
        bar   = "█" * int(pct / 2)
        print(f"{label:<{col_w}} {count:>6}  {pct:>5.1f}%  {DIM}{bar}{RESET}")
    print(divider(width=55))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    # Parse flags
    show_answers  = "--no-answers" not in args
    failed_only   = "--failed"     in args
    type_filter   = None
    json_path     = Path("results/raw_results.json")

    clean_args = []
    i = 0
    while i < len(args):
        if args[i] == "--type" and i + 1 < len(args):
            type_filter = args[i + 1]
            i += 2
        elif args[i] in ("--no-answers", "--failed"):
            i += 1
        else:
            clean_args.append(args[i])
            i += 1

    if clean_args:
        json_path = Path(clean_args[0])

    if not json_path.exists():
        print(f"[error] File not found: {json_path}", file=sys.stderr)
        print("  Usage: python view_results.py [path/to/raw_results.json] [--no-answers] [--failed] [--type <type>]")
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        results: list[dict] = json.load(f)

    if not results:
        print("[error] raw_results.json is empty.")
        sys.exit(1)

    # Apply filters
    if type_filter:
        results = [r for r in results if r.get("question_type") == type_filter]
        if not results:
            print(f"[error] No results found for question_type='{type_filter}'")
            sys.exit(1)
        print(f"\n{YELLOW}Filtered to question_type='{type_filter}' — {len(results)} questions{RESET}")

    if failed_only:
        before = len(results)
        results = filter_failed(results)
        print(f"\n{YELLOW}--failed filter: {len(results)}/{before} questions had at least one wrong answer{RESET}")

    # Print all tables
    print(f"\n{BOLD}Results viewer — {json_path}{RESET}")
    print(f"{DIM}{len(results)} question(s) loaded{RESET}")

    print_per_question_table(results, show_answers=show_answers)
    print_overall_table(results)
    print_type_table(results)
    print_disagreement_analysis(results)

    print(f"\n{DIM}Tip: run with --no-answers for a compact view, --failed to focus on errors, --type <name> to filter by question type{RESET}\n")


if __name__ == "__main__":
    main()