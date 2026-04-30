"""count_cluster_within_rmsd.py

For each row in a CSV of (PDB ID, PyMOL-style residue selection) pairs, count
how many members of the protein's RCSB sequence-identity cluster (40 % by
default) deposited on or before a configurable cutoff are within
--rmsd-threshold of the reference's altloc A and altloc B over the selected
residues. Cluster members are stripped to highest-occupancy altloc (the AF3
training-prep convention). Kabsch alignment uses the full chain backbone of
sequence-aligned residues; the RMSD is computed on heavy atoms of the
selected residues only, matched by atom name.

Output: original CSV plus columns
  - n_within_thresh_altloc_A
  - n_within_thresh_altloc_B
  - n_cluster_evaluated  (denominator: cluster members actually scored)
  - n_selected_resids_missing_in_reference  (CSV resids not present in RCSB ref)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB.Polypeptide import index_to_one, three_to_index

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_RT_structures_from_cluster import PDBClusterAnalyzer  # noqa: E402

BACKBONE_ATOMS = ("N", "CA", "C", "O")

SELECTION_PART = re.compile(
    r"^\s*chain\s+(\S+)\s+and\s+resi\s+(-?\d+)\s*-\s*(-?\d+)\s*$",
    re.IGNORECASE,
)


# ----------------------------- selection parsing ----------------------------


def parse_selection(
    s: str,
) -> Tuple[str, List[Tuple[str, Tuple[int, ...]]]]:
    """Parse 'chain A and resi 1-1;chain A and resi 318-320' into a chain ID
    and an ordered list of `(canonical_text, resid_tuple)` parts.

    The canonical text uses the form ``"chain X and resi LO-HI"`` so cache
    keys are stable across whitespace variations in the input CSV.
    """
    chains: Set[str] = set()
    parts: List[Tuple[str, Tuple[int, ...]]] = []
    for raw in s.split(";"):
        if not raw.strip():
            continue
        m = SELECTION_PART.match(raw)
        if not m:
            raise ValueError(f"Cannot parse selection part: {raw!r}")
        chain, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        if hi < lo:
            raise ValueError(f"resi {lo}-{hi} has end < start in part: {raw!r}")
        chains.add(chain)
        canonical = f"chain {chain} and resi {lo}-{hi}"
        parts.append((canonical, tuple(range(lo, hi + 1))))
    if len(chains) != 1:
        raise ValueError(f"Selection spans multiple chains {chains}: {s!r}")
    return chains.pop(), parts


# ----------------------------- cluster lookup -------------------------------


def find_cluster_for_entity(
    clusters: Sequence[Set[str]], entity_id: str
) -> Optional[Set[str]]:
    for cluster in clusters:
        if entity_id in cluster:
            return cluster
    return None


# ----------------------------- structure loading ----------------------------


def _chain_polymer_atoms(struct: "struc.AtomArray", chain_id: str) -> "struc.AtomArray":
    """Restrict to ATOM records (hetero=False) on a chain, drop hydrogens."""
    mask = (struct.chain_id == chain_id) & (~struct.hetero) & (struct.element != "H")
    return struct[mask]


def _chain_sequence(chain_struct: "struc.AtomArray") -> Tuple[str, List[int]]:
    """Build one-letter sequence and resid list from a single-chain polymer AtomArray."""
    starts = struc.get_residue_starts(chain_struct)
    seq_chars = []
    resids: List[int] = []
    for i in starts:
        rname = str(chain_struct.res_name[i])
        try:
            one = index_to_one(three_to_index(rname))
        except Exception:
            one = "X"
        seq_chars.append(one)
        resids.append(int(chain_struct.res_id[i]))
    return "".join(seq_chars), resids


def _altloc_letters_per_residue(
    chain_atoms: "struc.AtomArray",
) -> Dict[int, Set[str]]:
    """Per-residue set of altloc IDs present (including shared markers)."""
    out: Dict[int, Set[str]] = {}
    starts = struc.get_residue_starts(chain_atoms, add_exclusive_stop=True)
    altlocs = chain_atoms.altloc_id
    res_ids = chain_atoms.res_id
    for s, e in zip(starts[:-1], starts[1:]):
        out[int(res_ids[s])] = {str(x) for x in altlocs[s:e]}
    return out


def load_reference(
    cif_path: str, chain_id: str
) -> Tuple[
    "struc.AtomArray",
    "struc.AtomArray",
    str,
    str,
    List[int],
    Dict[int, Set[str]],
]:
    """Returns (struct_altloc_A, struct_altloc_B, entity_id, sequence, resids,
    altloc_letters_per_resid).

    Both altloc structures contain shared atoms (altloc '.' or ' ') plus their
    respective altloc atoms. Hydrogens and HETATM ligands are dropped. The
    altloc_letters map gives the set of altloc IDs present per residue (used
    by the caller to decide whether the residue carries both A and B
    alternatives).
    """
    cif = pdbx.CIFFile.read(cif_path)
    full = pdbx.get_structure(
        cif, model=1, altloc="all", extra_fields=["label_entity_id"]
    )
    chain_atoms = _chain_polymer_atoms(full, chain_id)
    if len(chain_atoms) == 0:
        raise ValueError(f"No polymer atoms on chain {chain_id} in {cif_path}")

    entity_ids = sorted(set(str(e) for e in chain_atoms.label_entity_id))
    if len(entity_ids) > 1:
        # Pick the entity with the most atoms (the polymer).
        counts = {e: int((chain_atoms.label_entity_id == e).sum()) for e in entity_ids}
        entity_id = max(counts, key=counts.get)
    else:
        entity_id = entity_ids[0]

    # Shared atoms use '.' in RCSB mmCIF; ' ' (space) in some non-RCSB writers.
    shared = (chain_atoms.altloc_id == ".") | (chain_atoms.altloc_id == " ")
    mask_a = shared | (chain_atoms.altloc_id == "A")
    mask_b = shared | (chain_atoms.altloc_id == "B")
    struct_a = chain_atoms[mask_a]
    struct_b = chain_atoms[mask_b]

    # Sequence comes from the full chain (altloc='all'); get_residue_starts collapses
    # duplicate altloc atoms into one start per residue.
    seq, resids = _chain_sequence(chain_atoms)
    altloc_letters = _altloc_letters_per_residue(chain_atoms)
    return struct_a, struct_b, entity_id, seq, resids, altloc_letters


def load_member(
    cif_path: str,
) -> "struc.AtomArray":
    """Load a cluster-member structure stripped to highest-occupancy altloc, no H, no HETATM."""
    cif = pdbx.CIFFile.read(cif_path)
    full = pdbx.get_structure(
        cif, model=1, altloc="occupancy", extra_fields=["label_entity_id"]
    )
    return full[(~full.hetero) & (full.element != "H")]


def find_member_chain(
    member_polymer: "struc.AtomArray", target_entity_id: str, ref_seq: str
) -> Optional[str]:
    """Return chain_id of the member chain matching the target entity.

    Prefer label_entity_id match; fall back to best pairwise-alignment score
    against the reference sequence if no chain has the target entity (e.g. when
    entity numbering changed between the cluster file and the current mmCIF).
    """
    entity_match = member_polymer.label_entity_id == target_entity_id
    if entity_match.any():
        chain_ids = sorted(set(str(c) for c in member_polymer.chain_id[entity_match]))
        return chain_ids[0]

    aligner = PairwiseAligner()
    aligner.mode = "global"
    best_score = -float("inf")
    best_chain: Optional[str] = None
    for cid in sorted(set(str(c) for c in member_polymer.chain_id)):
        chain_struct = member_polymer[member_polymer.chain_id == cid]
        if len(chain_struct) == 0:
            continue
        seq, _ = _chain_sequence(chain_struct)
        if len(seq) < 10:
            continue
        try:
            score = float(aligner.score(ref_seq, seq))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_chain = cid
    return best_chain


# ------------------------- residue mapping & RMSD ---------------------------


def align_residues(
    ref_seq: str,
    mob_seq: str,
    ref_resids: Sequence[int],
    mob_resids: Sequence[int],
) -> Dict[int, int]:
    """Return ref_resid -> mob_resid for non-gap aligned positions."""
    if not ref_seq or not mob_seq:
        return {}
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    alignments = aligner.align(ref_seq, mob_seq)
    if not alignments:
        return {}
    aln = alignments[0]
    ref_blocks, mob_blocks = aln.aligned
    res_map: Dict[int, int] = {}
    for (rs, re_), (ms, me_) in zip(ref_blocks, mob_blocks):
        for offset in range(re_ - rs):
            res_map[ref_resids[rs + offset]] = mob_resids[ms + offset]
    return res_map


def _residue_atoms(chain_struct: "struc.AtomArray", res_id: int) -> "struc.AtomArray":
    return chain_struct[chain_struct.res_id == res_id]


def _stack_matched_atoms(
    ref_chain: "struc.AtomArray",
    mob_chain: "struc.AtomArray",
    ref_resid: int,
    mob_resid: int,
    atom_name_filter: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ref_coords, mob_coords) matched by atom name within one residue pair.

    If atom_name_filter is None, match every atom name common to both residues
    (e.g. "all heavy atoms"). Else restrict to names in the filter (e.g. backbone).
    Returns (k, 3), (k, 3) arrays.
    """
    ref_res = _residue_atoms(ref_chain, ref_resid)
    mob_res = _residue_atoms(mob_chain, mob_resid)
    if len(ref_res) == 0 or len(mob_res) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    ref_names = list(ref_res.atom_name)
    mob_names = list(mob_res.atom_name)
    common = set(ref_names) & set(mob_names)
    if atom_name_filter is not None:
        common = common & set(atom_name_filter)
    if not common:
        return np.empty((0, 3)), np.empty((0, 3))

    ref_coords = []
    mob_coords = []
    for name in common:
        # Take the first match in each residue. After altloc stripping there
        # should be at most one atom per name; if not, [0] is a safe pick.
        ri = ref_names.index(name)
        mi = mob_names.index(name)
        ref_coords.append(ref_res.coord[ri])
        mob_coords.append(mob_res.coord[mi])
    return np.asarray(ref_coords), np.asarray(mob_coords)


def compute_rmsd_to_reference(
    ref_chain: "struc.AtomArray",
    mob_chain: "struc.AtomArray",
    res_map: Dict[int, int],
    parts: Sequence[Tuple[str, Sequence[int]]],
    min_backbone_atoms: int = 10,
    min_selected_atoms: int = 3,
) -> Dict[str, float]:
    """Kabsch-align on full-chain backbone of aligned residues, then compute
    one RMSD per selection part using the same global transform.

    Returns a dict ``{part_text: rmsd}``. Parts with fewer than
    ``min_selected_atoms`` matched atoms are absent. Returns ``{}`` if the
    backbone alignment itself can't be computed.
    """
    bb_ref: List[np.ndarray] = []
    bb_mob: List[np.ndarray] = []
    for ref_rid, mob_rid in res_map.items():
        r, m = _stack_matched_atoms(
            ref_chain, mob_chain, ref_rid, mob_rid, atom_name_filter=BACKBONE_ATOMS
        )
        if len(r) > 0:
            bb_ref.append(r)
            bb_mob.append(m)
    if not bb_ref:
        return {}
    bb_ref_arr = np.concatenate(bb_ref, axis=0)
    bb_mob_arr = np.concatenate(bb_mob, axis=0)
    if len(bb_ref_arr) < min_backbone_atoms:
        return {}

    _, transform = struc.superimpose(bb_ref_arr, bb_mob_arr)

    out: Dict[str, float] = {}
    for part_text, part_resids in parts:
        sel_ref: List[np.ndarray] = []
        sel_mob: List[np.ndarray] = []
        for rid in part_resids:
            if rid not in res_map:
                continue
            r, m = _stack_matched_atoms(ref_chain, mob_chain, rid, res_map[rid])
            if len(r) > 0:
                sel_ref.append(r)
                sel_mob.append(m)
        if not sel_ref:
            continue
        sel_ref_arr = np.concatenate(sel_ref, axis=0)
        sel_mob_arr = np.concatenate(sel_mob, axis=0)
        if len(sel_ref_arr) < min_selected_atoms:
            continue
        sel_mob_fitted = transform.apply(sel_mob_arr)
        diff = sel_ref_arr - sel_mob_fitted
        out[part_text] = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return out


# --------------------------- per-protein orchestration ----------------------


def evaluate_protein(
    protein_id: str,
    selection_str: str,
    clusters: Sequence[Set[str]],
    analyzer: PDBClusterAnalyzer,
    rmsd_threshold: float,
    max_workers: Optional[int],
    verbose: bool = False,
) -> Tuple[
    List[Tuple[str, Tuple[int, ...]]],
    Dict[str, Dict[str, int]],
    Set[str],
]:
    """Returns ``(parts, part_stats, skipped_parts)``.

    ``parts`` is the parsed selection in input order: list of
    ``(canonical_text, resids)``. ``part_stats`` maps each part_text to
    ``{n_within_A, n_within_B, n_evaluated, n_missing}``.
    ``skipped_parts`` is the set of part_text values whose residues lack
    altloc A or B coverage in the reference (so a strict A-vs-B comparison
    isn't meaningful); these are dropped from the output entirely.

    Failures (download / parse / no cluster) return zeros for every part so
    the caller can still emit one row per part. Diagnostic detail goes to
    stdout under ``--verbose``.
    """
    chain_id, parts = parse_selection(selection_str)

    def _zero_stats(part_missing: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        return {
            pt: {
                "n_within_A": 0,
                "n_within_B": 0,
                "n_evaluated": 0,
                "n_missing": part_missing.get(pt, 0),
            }
            for pt, _ in parts
        }

    ref_path = analyzer._download_mmcif_cached(protein_id)
    if ref_path is None:
        print(f"  [{protein_id}] reference mmCIF download failed", file=sys.stderr)
        return (parts, _zero_stats({pt: 0 for pt, _ in parts}), set())

    try:
        ref_a, ref_b, entity_id, ref_seq, ref_resids, altloc_letters = (
            load_reference(ref_path, chain_id)
        )
    except Exception as e:
        print(f"  [{protein_id}] reference parse failed: {e}", file=sys.stderr)
        return (parts, _zero_stats({pt: 0 for pt, _ in parts}), set())

    # Per-part missing-residue counts and present-residue subsets for compute.
    ref_resid_set = set(ref_resids)
    part_missing_count: Dict[str, int] = {}
    parts_for_compute: List[Tuple[str, Tuple[int, ...]]] = []
    skipped_parts: Set[str] = set()
    for part_text, part_resids in parts:
        present = tuple(r for r in part_resids if r in ref_resid_set)
        part_missing_count[part_text] = len(part_resids) - len(present)
        if verbose and part_missing_count[part_text]:
            missing = [r for r in part_resids if r not in ref_resid_set]
            print(
                f"  [{protein_id}] {part_text}: "
                f"{part_missing_count[part_text]} resid(s) missing in reference: "
                f"{missing}"
            )
        # Strict A/B check: every present residue must carry both an
        # A-side atom (shared or altloc 'A') and a B-side atom (shared or
        # altloc 'B'). If any residue fails, drop the part with a warning.
        bad: List[Tuple[int, Set[str]]] = []
        for rid in present:
            letters = altloc_letters.get(rid, set())
            has_a = bool(letters & {".", " ", "A"})
            has_b = bool(letters & {".", " ", "B"})
            if not (has_a and has_b):
                bad.append((rid, letters))
        if bad:
            details = ", ".join(
                f"{rid}({','.join(sorted(letters)) or '-'})" for rid, letters in bad
            )
            print(
                f"  [{protein_id}] {part_text}: dropped — residue(s) "
                f"{details} lack altloc A or B coverage",
                file=sys.stderr,
            )
            skipped_parts.add(part_text)
            continue
        parts_for_compute.append((part_text, present))

    cluster_key = f"{protein_id}_{entity_id}"
    cluster = find_cluster_for_entity(clusters, cluster_key)
    if cluster is None:
        print(
            f"  [{protein_id}] no cluster contains entity {cluster_key}",
            file=sys.stderr,
        )
        return (parts, _zero_stats(part_missing_count), skipped_parts)

    if verbose:
        print(f"  [{protein_id}] cluster {cluster_key} has {len(cluster)} entities")

    # Filter by deposit_date <= cutoff via existing analyzer (also applies the
    # baseline X-ray + has_released filters and drops non-PDB tokens).
    filtered_entities = analyzer.filter(cluster)
    if verbose:
        print(
            f"  [{protein_id}] {len(filtered_entities)} entities pass "
            f"deposit_date <= {analyzer.cutoff_date}"
        )

    part_stats = _zero_stats(part_missing_count)
    counter_lock = threading.Lock()

    def process_member(
        entity: str,
    ) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
        try:
            member_pdb_id, member_entity = entity.split("_", 1)
        except ValueError:
            return None
        if member_pdb_id == protein_id:
            return None  # skip self
        cif = analyzer._download_mmcif_cached(member_pdb_id)
        if cif is None:
            return None
        try:
            member_polymer = load_member(cif)
        except Exception:
            return None
        chain = find_member_chain(member_polymer, member_entity, ref_seq)
        if chain is None:
            return None
        member_chain = member_polymer[member_polymer.chain_id == chain]
        if len(member_chain) == 0:
            return None
        try:
            mob_seq, mob_resids = _chain_sequence(member_chain)
        except Exception:
            return None
        res_map = align_residues(ref_seq, mob_seq, ref_resids, mob_resids)
        if not res_map:
            return None
        try:
            rmsd_a_dict = compute_rmsd_to_reference(
                ref_a, member_chain, res_map, parts_for_compute
            )
            rmsd_b_dict = compute_rmsd_to_reference(
                ref_b, member_chain, res_map, parts_for_compute
            )
        except Exception:
            return None
        if not rmsd_a_dict and not rmsd_b_dict:
            return None
        return rmsd_a_dict, rmsd_b_dict

    workers = max_workers or min(16, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_member, e): e for e in filtered_entities}
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            rmsd_a_dict, rmsd_b_dict = res
            with counter_lock:
                for part_text, _ in parts_for_compute:
                    ra = rmsd_a_dict.get(part_text)
                    rb = rmsd_b_dict.get(part_text)
                    if ra is None or rb is None:
                        continue
                    s = part_stats[part_text]
                    s["n_evaluated"] += 1
                    if ra < rmsd_threshold:
                        s["n_within_A"] += 1
                    if rb < rmsd_threshold:
                        s["n_within_B"] += 1
            if verbose:
                ent = futures[fut]
                mean_a = (
                    sum(rmsd_a_dict.values()) / len(rmsd_a_dict)
                    if rmsd_a_dict
                    else float("nan")
                )
                mean_b = (
                    sum(rmsd_b_dict.values()) / len(rmsd_b_dict)
                    if rmsd_b_dict
                    else float("nan")
                )
                print(
                    f"    [{protein_id}] {ent}: rmsd_A_mean={mean_a:.3f}  "
                    f"rmsd_B_mean={mean_b:.3f}  parts_evaluated={len(rmsd_a_dict)}"
                )

    if verbose:
        for part_text, _ in parts_for_compute:
            s = part_stats[part_text]
            print(
                f"  [{protein_id}] {part_text} within {rmsd_threshold} A: "
                f"A={s['n_within_A']}/{s['n_evaluated']}  "
                f"B={s['n_within_B']}/{s['n_evaluated']}"
            )

    return (parts, part_stats, skipped_parts)


# ----------------------------------- CLI ------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument(
        "--cutoff-date",
        required=True,
        help="ISO date YYYY-MM-DD; cluster members deposited after this are excluded",
    )
    p.add_argument(
        "--seq-identity",
        type=int,
        default=40,
        help="RCSB sequence-identity cluster threshold percent (default: 40)",
    )
    p.add_argument(
        "--rmsd-threshold",
        type=float,
        default=2.0,
        help="RMSD threshold in angstroms (default: 2.0)",
    )
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument(
        "--max-rate",
        type=float,
        default=None,
        help="RCSB requests per second (across all threads)",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    analyzer = PDBClusterAnalyzer(
        target_cluster=0,
        cutoff_date=args.cutoff_date,
        temperature=None,
        resolution=None,
        seq_identity=args.seq_identity / 100.0
        if args.seq_identity > 1
        else args.seq_identity,
        max_workers=args.max_workers,
        max_rate=args.max_rate,
    )

    cluster_cache = os.path.join(
        analyzer.cache_dir, f"clusters-by-entity-{args.seq_identity}.txt"
    )
    print(f"Fetching {args.seq_identity}% cluster file (cache: {cluster_cache}) ...")
    t0 = time.time()
    clusters = analyzer.fetch_clusters(keep_entity_id=True, cache_path=cluster_cache)
    print(f"  {len(clusters)} clusters loaded in {time.time() - t0:.1f}s")

    with open(args.input_csv, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        in_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    out_fieldnames = in_fieldnames + [
        "selection_part_index",
        "selection_part",
        "rmsd_threshold",
        "seq_identity",
        "n_within_thresh_altloc_A",
        "n_within_thresh_altloc_B",
        "n_cluster_evaluated",
        "n_selected_resids_missing_in_reference",
    ]

    def write_out(buf: List[Dict[str, str]]):
        tmp = args.output_csv + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=out_fieldnames)
            w.writeheader()
            w.writerows(buf)
        os.replace(tmp, args.output_csv)

    # Treat --output-csv as a cache, keyed at the per-selection-part level.
    # Cache key = (protein, selection_part_text, rmsd_threshold, seq_identity).
    # Cutoff is implicit in the filename (per-cutoff CSV convention) and not
    # validated against cached rows.
    output_rows: List[Dict[str, str]] = []
    key_to_index: Dict[Tuple[str, str, float, int], int] = {}

    if (
        os.path.exists(args.output_csv)
        and os.path.getsize(args.output_csv) > 0
    ):
        with open(args.output_csv, "r", encoding="utf-8") as fh:
            cache_reader = csv.DictReader(fh)
            cached_fieldnames = list(cache_reader.fieldnames or [])
            required = (
                "selection_part_index",
                "selection_part",
                "rmsd_threshold",
                "seq_identity",
            )
            missing = [c for c in required if c not in cached_fieldnames]
            if missing:
                sys.exit(
                    f"Output CSV {args.output_csv} is missing column(s) "
                    f"{missing}; it was written by an older version of this "
                    "script. Move it aside or delete it, then re-run."
                )
            for cached_row in cache_reader:
                try:
                    cached_thresh = float(cached_row["rmsd_threshold"])
                    cached_identity = int(cached_row["seq_identity"])
                except (TypeError, ValueError):
                    output_rows.append(cached_row)
                    continue
                key = (
                    cached_row.get("protein", "").strip(),
                    cached_row.get("selection_part", "").strip(),
                    cached_thresh,
                    cached_identity,
                )
                output_rows.append(cached_row)
                key_to_index[key] = len(output_rows) - 1
        print(
            f"Resuming from {args.output_csv} with {len(output_rows)} cached "
            f"row(s). Verify the cutoff in the filename matches "
            f"--cutoff-date {args.cutoff_date}."
        )

    for i, row in enumerate(rows, 1):
        protein_id = row.get("protein", "").strip()
        selection = row.get("selection", "").strip()
        if not protein_id or not selection:
            continue

        try:
            chain_id, parts = parse_selection(selection)
        except ValueError as e:
            print(f"  [{protein_id}] selection parse failed: {e}", file=sys.stderr)
            continue

        # Identify which parts already have a cache hit so we can avoid the
        # per-protein compute entirely if every part is cached.
        uncached_part_keys: List[Tuple[str, str, float, int]] = []
        for part_text, _ in parts:
            key = (
                protein_id,
                part_text,
                args.rmsd_threshold,
                args.seq_identity,
            )
            if key not in key_to_index:
                uncached_part_keys.append(key)

        if not uncached_part_keys:
            print(
                f"\n=== [{i}/{len(rows)}] {protein_id} === all "
                f"{len(parts)} part(s) cached, skipping"
            )
            continue

        cached_count = len(parts) - len(uncached_part_keys)
        print(
            f"\n=== [{i}/{len(rows)}] {protein_id} === "
            f"{cached_count} part(s) cached, "
            f"{len(uncached_part_keys)} to compute"
        )

        try:
            _, part_stats, skipped_parts = evaluate_protein(
                protein_id,
                selection,
                clusters,
                analyzer,
                args.rmsd_threshold,
                args.max_workers,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"  [{protein_id}] unhandled error: {e}", file=sys.stderr)
            part_stats = {
                pt: {
                    "n_within_A": 0,
                    "n_within_B": 0,
                    "n_evaluated": 0,
                    "n_missing": 0,
                }
                for pt, _ in parts
            }
            skipped_parts = set()

        if skipped_parts:
            print(
                f"  [{protein_id}] {len(skipped_parts)} of {len(parts)} part(s) "
                f"dropped (altloc letters not A/B)",
                file=sys.stderr,
            )

        for idx, (part_text, _) in enumerate(parts):
            key = (
                protein_id,
                part_text,
                args.rmsd_threshold,
                args.seq_identity,
            )
            if key in key_to_index:
                continue  # leave the cached row untouched
            if part_text in skipped_parts:
                continue  # altloc-incomplete part: drop with warning, no row
            stats = part_stats.get(
                part_text,
                {
                    "n_within_A": 0,
                    "n_within_B": 0,
                    "n_evaluated": 0,
                    "n_missing": 0,
                },
            )
            new_row = dict(row)
            new_row.update(
                {
                    "selection_part_index": idx,
                    "selection_part": part_text,
                    "rmsd_threshold": args.rmsd_threshold,
                    "seq_identity": args.seq_identity,
                    "n_within_thresh_altloc_A": stats["n_within_A"],
                    "n_within_thresh_altloc_B": stats["n_within_B"],
                    "n_cluster_evaluated": stats["n_evaluated"],
                    "n_selected_resids_missing_in_reference": stats["n_missing"],
                }
            )
            output_rows.append(new_row)
            key_to_index[key] = len(output_rows) - 1

        write_out(output_rows)


if __name__ == "__main__":
    main()
