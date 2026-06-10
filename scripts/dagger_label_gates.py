"""DAgger Step 2b: label gate timesteps from MP4 previews.

For each *.preview.mp4 whose sibling *.pkl[.xz] does not yet contain a gate,
prints the preview path and prompts in the terminal for the gate frame. You
open the MP4 yourself in whichever player you prefer (VS Code, mpv, browser,
etc.), read the frame number off the burned-in counter, and type it in.

How the gate is stored
----------------------
The gate is **appended** to the failure pkl as an additional xz stream
containing a small pickle ``{"gate_idx": N, "labeled_at": ..., "overrides": ...}``.
The xz container natively supports concatenated streams, so the file remains
a valid .pkl.xz that ``lzma.open`` reads transparently — but writing the
label is O(1) (a few hundred bytes), with no rewrite of the multi-MB main
pickle.

Per-gate overrides
------------------
After the gate frame is entered, two optional override prompts capture:

  - force_pre_assemble_done: list of part indices to set to True on restore.
    Use this when the policy actually placed an earlier part correctly but
    the FSM never observed the simultaneity window that flips its
    pre_assemble_done flag (so without the override the expert would
    re-assemble a part that's already in place).
  - force_state: {part_idx: state_name} to override _last_state on restore.
    Necessary for the Non-Markovian expert (where compute_state's sequential
    transition check reads _last_state) and useful when the snapshotted FSM
    state is in a stuck configuration. By design dagger_collect_corrections.py
    does NOT reset NM cycle counters / align_offset / other per-part caches:
    the chosen override state is presumed not-yet-reached, so its counters
    should already be at initial defaults from the snapshot. Pick a different
    override state if that assumption fails.

State names are validated against the relevant part class's ALL_STATES
constant (Leg.ALL_STATES / TableTop.ALL_STATES).

Furniture support
-----------------
The part-name → part-class dispatch (for state-name validation) is currently
HARDCODED for one_leg / square_table: "leg" in name → Leg.ALL_STATES,
"top" or "body" in name → TableTop.ALL_STATES. Other furniture (cabinet,
lamp, round_table) will hit an explicit error during validation — extend
``_part_class_for`` when adding support.

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

import torch  # noqa: F401 — required to unpickle the torch tensors in failure pkls


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


def _append_gate(pkl_path: Path, gate_idx: int, overrides: dict | None = None) -> None:
    """Append a tiny xz-compressed gate record. O(1) — no rewrite of the main pickle."""
    patch = {"gate_idx": int(gate_idx), "labeled_at": datetime.datetime.now().isoformat()}
    if overrides:
        patch["overrides"] = overrides
    encoded = lzma.compress(pickle.dumps(patch))
    with open(pkl_path, "ab") as f:
        f.write(encoded)


def _truncate_to(pkl_path: Path, target_size: int) -> None:
    """Cut the file back to ``target_size`` bytes — used to undo an appended gate stream."""
    with open(pkl_path, "r+b") as f:
        f.truncate(target_size)


def _load_snapshot_at_gate(pkl_path: Path, gate_idx: int):
    """Load the failure pkl's first stream and return (parts, assembled_set) at gate_idx.

    Returns (None, None) if gate_idx is out of range or the pkl is malformed.
    """
    with lzma.open(pkl_path, "rb") as f:
        data = pickle.load(f)  # first stream = main data; gate streams come after
    snapshots = data.get("snapshots", [])
    if not (0 <= gate_idx < len(snapshots)):
        print(f"    invalid gate_idx={gate_idx}: snapshot count is {len(snapshots)}")
        return None, None
    phys, parts = snapshots[gate_idx]
    if not parts:
        return None, None
    assembled_sets = phys.get("assembled_sets", [set()])
    return parts[0], set(assembled_sets[0])  # single-env failure pkls


def _furniture_from_path(failure_dir: Path) -> str:
    """Derive the furniture name from the failure dir's path.

    Layout: <root>/raw/sim/<furniture>/<demo_source>/<randomness>/failure
    """
    return failure_dir.parents[2].name


def _load_should_be_assembled(furniture_name: str):
    """Return the furniture's should_be_assembled pair list (or None if unknown).

    Mirrors `furniture_sim_env.py:1621` — the assembly order, not the part-index order,
    determines which part the FSM works on next.
    """
    name_to_cls = {
        "one_leg": ("one_leg", "OneLeg"),
        "square_table": ("square_table", "SquareTable"),
        "lamp": ("lamp", "Lamp"),
        "round_table": ("round_table", "RoundTable"),
        "desk": ("desk", "Desk"),
        "cabinet": ("cabinet", "Cabinet"),
        "chair": ("chair", "Chair"),
        "stool": ("stool", "Stool"),
        "drawer": ("drawer", "Drawer"),
    }
    if furniture_name not in name_to_cls:
        print(f"  warning: no should_be_assembled mapping for furniture={furniture_name!r}; active-part marking disabled")
        return None
    mod_name, cls_name = name_to_cls[furniture_name]
    mod = __import__(f"furniture_bench.furniture.{mod_name}", fromlist=[cls_name])
    return getattr(mod, cls_name)().should_be_assembled


def _part_class_for(part_name: str):
    """Return the Part subclass for a given snapshot part name.

    HARDCODED for one_leg / square_table — extend when adding other furniture.
    Lazy-imports furniture_bench (heavy: triggers IsaacGym) only when this
    function is actually called.
    """
    from furniture_bench.furniture.parts.leg import Leg
    from furniture_bench.furniture.parts.table_top import TableTop

    name = part_name.lower()
    if "leg" in name:
        return Leg
    if "top" in name or "body" in name:
        return TableTop
    raise ValueError(
        f"Unknown part name {part_name!r} — _part_class_for dispatch only knows "
        f"one_leg / square_table parts. Extend dagger_label_gates.py:_part_class_for "
        f"to add other furniture types."
    )


def _find_active_part_idx(parts: list, assembled_set: set, should_be_assembled):
    """Return the active part index, mirroring env logic at furniture_sim_env.py:1621-1648.

    Walks `should_be_assembled` in order, skipping pairs already in
    `assembled_set`; for the next unassembled pair (i, j), the active part is
    parts[i] if it still needs pre-assembly, else parts[j]. Returns None when
    all pairs are assembled or `should_be_assembled` is unavailable.
    """
    if not should_be_assembled:
        return None
    for i, j in should_be_assembled:
        if (i, j) in assembled_set:
            continue
        if parts[i].get("pre_assemble_done") is False:
            return i
        return j
    return None


def _display_snapshot_state(parts: list, active_idx) -> None:
    """Print a per-part summary of the snapshotted FSM state at the gate.

    Marks the active part (the one the FSM is currently working on) with
    `<- ACTIVE`. Active-part determination follows the env's logic — see
    `_find_active_part_idx`.
    """
    for i, p in enumerate(parts):
        name = p.get("name", "<unknown>")
        pad = p.get("pre_assemble_done", "?")
        ls = p.get("_last_state", "?")
        marker = "  <- ACTIVE" if i == active_idx else ""
        print(f"    parts[{i}]={name:30s}  pre_assemble_done={str(pad):5s}  _last_state={ls!r}{marker}")


def _prompt_yes_no(question: str, default: bool = True) -> bool:
    """Y/n prompt with retry on invalid input. Empty input → default."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        resp = input(f"  {question} {suffix}: ").strip().lower()
        if not resp:
            return default
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("    invalid; please answer y or n")


def _prompt_pre_assemble_done_overrides(n_parts: int) -> list:
    """Parse comma-separated part indices; re-prompt on invalid input. Empty → []."""
    while True:
        resp = input("  Force pre_assemble_done=True for parts (comma-sep idx, Enter=none): ").strip()
        if not resp:
            return []
        try:
            indices = [int(tok.strip()) for tok in resp.split(",") if tok.strip()]
            for i in indices:
                if not 0 <= i < n_parts:
                    raise ValueError(f"part index {i} out of range [0, {n_parts - 1}]")
            return sorted(set(indices))
        except ValueError as e:
            print(f"    invalid ({e}); try again")


def _prompt_state_overrides(parts: list, active_idx) -> dict:
    """Override the ACTIVE part's _last_state with a state string. Empty → {}.

    Only the active part's _last_state matters to the next FSM tick (the FSM
    advances sequentially through assembly pairs and only reads _last_state
    for the one currently being worked on), so we auto-target it instead of
    asking for a part index.
    """
    if active_idx is None:
        print("  (no active part; skipping _last_state override)")
        return {}
    part_name = parts[active_idx].get("name", "<unknown>")
    current_state = parts[active_idx].get("_last_state", "?")
    part_cls = _part_class_for(part_name)
    while True:
        resp = input(
            f"  Override active part [{active_idx}] {part_name!r} _last_state "
            f"(current={current_state!r}, Enter=keep): "
        ).strip()
        if not resp:
            return {}
        if resp not in part_cls.ALL_STATES:
            print(f"    invalid: {resp!r} not in {part_cls.__name__}.ALL_STATES = {list(part_cls.ALL_STATES)}")
            continue
        return {active_idx: resp}


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
            # gate_idx must be >= 1: dagger_collect_corrections.py needs at least
            # one preceding obs (o_{n-1}) to form the correct (o_{n-1}, o_n) policy
            # history for the gate sample. Gate=0 would have no preceding obs.
            if n < 1:
                raise ValueError("must be >= 1 (gate=0 has no preceding obs for history context)")
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

    furniture_name = _furniture_from_path(failure_dir)
    should_be_assembled = _load_should_be_assembled(furniture_name)
    print(f"furniture={furniture_name}  should_be_assembled={should_be_assembled}")

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

        # Load snapshot at this gate so we can display the FSM state and let
        # the user confirm or override. parts is the per-part list at the gate.
        parts, assembled_set = _load_snapshot_at_gate(pkl, frame)
        if parts is None:
            print("  could not load snapshot; saving gate without overrides")
            overrides_dict = None
        else:
            active_idx = _find_active_part_idx(parts, assembled_set, should_be_assembled)
            print(f"  Snapshot FSM state at gate={frame}  (assembled_set={assembled_set or '{}'}):")
            _display_snapshot_state(parts, active_idx)
            if _prompt_yes_no("Is this state correct?", default=True):
                overrides_dict = None
            else:
                force_pad = _prompt_pre_assemble_done_overrides(n_parts=len(parts))
                force_state = _prompt_state_overrides(parts, active_idx)
                overrides_dict = {}
                if force_pad:
                    overrides_dict["force_pre_assemble_done"] = force_pad
                if force_state:
                    overrides_dict["force_state"] = force_state
                if not overrides_dict:
                    overrides_dict = None

        size_before = pkl.stat().st_size
        _append_gate(pkl, frame, overrides_dict)
        undo_stack.append((i, pkl, size_before))
        if overrides_dict:
            print(f"  -> gate={frame} with overrides={overrides_dict} appended to {pkl.name}")
        else:
            print(f"  -> gate={frame} appended to {pkl.name}")
        n_new += 1
        i += 1

    print(f"\nDone. Appended {n_new} gate(s).")


if __name__ == "__main__":
    main()
