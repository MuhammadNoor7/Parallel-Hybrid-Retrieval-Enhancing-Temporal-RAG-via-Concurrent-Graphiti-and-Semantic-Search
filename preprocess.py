from __future__ import annotations

import heapq
import json
from pathlib import Path

IN_PATH = Path("data/longmemeval_s_cleaned.json")
OUT_PATH = Path("data/longmemeval_s_cleaned_top5_shortest.json")
TOP_K = 5
CHUNK_SIZE = 1024 * 1024


def main() -> None:
    # Keep TOP_K smallest spans using a max-heap implemented as a min-heap over negative span.
    # Tuple layout: (-span, -end_offset, index, start_line, end_line, start_offset, end_offset)
    heap: list[tuple[int, int, int, int, int, int, int]] = []

    in_string = False
    escape = False
    seen_top_array = False
    in_obj = False
    brace_depth = 0

    obj_start_line: int | None = None
    obj_start_offset: int | None = None

    line_num = 1
    offset = 0
    obj_index = -1

    with IN_PATH.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break

            for b in chunk:
                if b == 10:  # \n
                    line_num += 1

                if in_string:
                    if escape:
                        escape = False
                    elif b == 92:  # \\
                        escape = True
                    elif b == 34:  # "
                        in_string = False
                else:
                    if b == 34:  # "
                        in_string = True
                    else:
                        if not seen_top_array:
                            if b == 91:  # [
                                seen_top_array = True
                        else:
                            if in_obj:
                                if b == 123:  # {
                                    brace_depth += 1
                                elif b == 125:  # }
                                    brace_depth -= 1
                                    if brace_depth == 0:
                                        obj_end_line = line_num
                                        obj_end_offset = offset + 1
                                        obj_index += 1

                                        assert obj_start_line is not None
                                        assert obj_start_offset is not None

                                        span = obj_end_line - obj_start_line + 1
                                        item = (
                                            -span,
                                            -obj_end_offset,
                                            obj_index,
                                            obj_start_line,
                                            obj_end_line,
                                            obj_start_offset,
                                            obj_end_offset,
                                        )

                                        if len(heap) < TOP_K:
                                            heapq.heappush(heap, item)
                                        else:
                                            # heap[0] is the *largest* span among kept items.
                                            if item > heap[0]:
                                                heapq.heapreplace(heap, item)

                                        in_obj = False
                            else:
                                if b == 123:  # {
                                    in_obj = True
                                    brace_depth = 1
                                    obj_start_line = line_num
                                    obj_start_offset = offset
                                elif b == 93:  # ]
                                    # End of top-level array
                                    seen_top_array = False

                offset += 1

    best: list[tuple[int, int, int, int, int, int]] = []
    for neg_span, _neg_end_offset, idx, start_line, end_line, start_off, end_off in heap:
        best.append((-neg_span, idx, start_line, end_line, start_off, end_off))

    best.sort(key=lambda t: (t[0], t[1]))

    rows: list[dict] = []
    with IN_PATH.open("rb") as f:
        for span, idx, start_line, end_line, start_off, end_off in best:
            f.seek(start_off)
            obj_bytes = f.read(end_off - start_off)
            obj = json.loads(obj_bytes)
            rows.append(obj)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_PATH} with {len(rows)} rows")
    for rank, (span, idx, start_line, end_line, _start_off, _end_off) in enumerate(best, start=1):
        print(f"{rank}. index={idx} span_lines={span} file_lines={start_line}-{end_line}")


if __name__ == "__main__":
    main()
