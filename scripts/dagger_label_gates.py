"""DAgger Step 2b: label gate timesteps from MP4 previews.

For each *.preview.mp4 whose sibling *.pkl[.xz] does not yet contain a gate,
prints the preview path and prompts in the terminal for the gate frame. You
open the MP4 yourself in whichever player you prefer (VS Code, mpv, browser,
etc.), read the frame number off the burned-in counter, and type it in.

How the gate is stored
----------------------
The gate is **appended** to the failure pkl as an additional xz stream
containing a small pickle ``{"gate_idx": N, "labeled_at": ...}``. The xz
container natively supports concatenated streams, so the file remains a
valid .pkl.xz that ``lzma.open`` reads transparently — but writing the
label is O(1) (a few hundred bytes), with no rewrite of the multi-MB main
pickle.

Already-labeled detection
-------------------------
Uses ``pkl.mtime > preview.mtime`` as the heuristic. The renderer (Step 2a)
writes the preview after the failure pkl is created; appending a gate
updates the pkl's mtime. Don't re-render previews with --overwrite on a
labeled directory or this heuristic breaks (you'd see false-negatives and
have to --relabel).

Prereq: run dagger_render_failures.py first so the *.preview.mp4 files exist.

Usage:
    python scripts/dagger_label_gates.py dataset/raw/sim/one_leg/dagger_iter0/low/failure
"""

from __future__ import annotations

import argparse
import datetime
import lzma
import pickle
from pathlib import Path


def _pkl_for_preview(preview_path: Path):
    """Return the matching .pkl.xz (preferred) or .pkl next to a preview, or None."""
    name = preview_path.name
    if name.endswith(".preview.mp4"):
        stem = name[: -len(".preview.mp4")]
    else:
        stem = preview_path.stem
    for cand in (preview_path.parent / f"{stem}.pkl.xz", preview_path.parent / f"{stem}.pkl"):
        if cand.exists():
            return cand
    return None


def _is_labeled(pkl_path: Path, preview_path: Path) -> bool:
    """pkl mtime > preview mtime => a gate has been appended."""
    return pkl_path.stat().st_mtime > preview_path.stat().st_mtime


def _append_gate(pkl_path: Path, gate_idx: int) -> None:
    """Append a tiny xz-compressed gate record. O(1) — no rewrite of the main pickle."""
    patch = {"gate_idx": int(gate_idx), "labeled_at": datetime.datetime.now().isoformat()}
    encoded = lzma.compress(pickle.dumps(patch))
    with open(pkl_path, "ab") as f:
        f.write(encoded)


def _truncate_to(pkl_path: Path, target_size: int) -> None:
    """Cut the file back to ``target_size`` bytes — used to undo an appended gate stream."""
    with open(pkl_path, "r+b") as f:
        f.truncate(target_size)


def _prompt_gate(preview_name: str, can_undo: bool):
    """Returns ('label', gate_idx) | ('skip', None) | ('quit', None) | ('undo', None)."""
    extra = " / u=undo last" if can_undo else ""
    while True:
        resp = input(f"  gate frame for {preview_name}  [int / s=skip / q=quit{extra}]: ").strip().lower()
        if resp in ("q", "quit"):
            return ("quit", None)
        if resp in ("s", "skip", ""):
            return ("skip", None)
        if can_undo and resp in ("u", "b", "back", "undo"):
            return ("undo", None)
        try:
            n = int(resp)
            if n < 0:
                raise ValueError("must be >= 0")
            return ("label", n)
        except ValueError as e:
            print(f"    invalid ({e}); try again")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("failure_dir", help="Path to dagger_iterN/{randomness}/failure/")
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Re-prompt and append a new gate even if the pkl already appears labeled.",
    )
    args = parser.parse_args()

    failure_dir = Path(args.failure_dir).expanduser().resolve()
    if not failure_dir.exists():
        print(f"ERROR: {failure_dir} does not exist.")
        return

    previews = sorted(failure_dir.glob("*.preview.mp4"))
    if not previews:
        print(f"No previews in {failure_dir}. Run dagger_render_failures.py first.")
        return

    pending = []
    n_orphan = 0
    for p in previews:
        pkl = _pkl_for_preview(p)
        if pkl is None:
            n_orphan += 1
            continue
        if not args.relabel and _is_labeled(pkl, p):
            continue
        pending.append((p, pkl))

    n_already = len(previews) - len(pending) - n_orphan
    summary = f"{len(previews)} previews total — {n_already} already labeled, {len(pending)} to do"
    if n_orphan:
        summary += f", {n_orphan} orphan (no matching pkl)"
    print(summary)

    # undo_stack: list of (pending_index, pkl_path, size_before_append) for each
    # successful label, so 'u' at any prompt can pop the most recent and rewind.
    undo_stack: list = []
    n_new = 0
    i = 0
    while i < len(pending):
        preview, pkl = pending[i]
        print(f"\n[{i + 1}/{len(pending)}] {preview}")

        action, frame = _prompt_gate(preview.name, can_undo=bool(undo_stack))

        if action == "quit":
            print("  quitting")
            break

        if action == "skip":
            print("  skipped")
            i += 1
            continue

        if action == "undo":
            prev_i, prev_pkl, prev_size = undo_stack.pop()
            _truncate_to(prev_pkl, prev_size)
            print(f"  undid gate on {prev_pkl.name} (truncated back to {prev_size} bytes); re-prompting that episode")
            n_new -= 1
            i = prev_i
            continue

        size_before = pkl.stat().st_size
        _append_gate(pkl, frame)
        undo_stack.append((i, pkl, size_before))
        print(f"  -> gate={frame} appended to {pkl.name}")
        n_new += 1
        i += 1

    print(f"\nDone. Appended {n_new} gate(s).")


if __name__ == "__main__":
    main()
