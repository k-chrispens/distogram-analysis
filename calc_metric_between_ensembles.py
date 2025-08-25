#!/usr/bin/env python3
"""
Compare two ensembles via CB–CB distograms using distributional metrics per position

This script builds CB–CB distance matrices for structures in two groups (A and B),
aligns them to a common reference sequence window, stacks the aligned distograms for
each group, and computes per-position distances between the distributions of A[:, i, j]
and B[:, i, j] for i<j only (strictly upper-triangular cells).

Metrics:
    - Wasserstein-1 distance (scipy.stats.wasserstein_distance) on raw distances
    - Jensen–Shannon distance on histograms per pair using adaptive binning (Sturges' rule
        clamped to [--min-bins, --bins]). Only the JS distance (output of jensenshannon())
        is saved for the pairwise matrix.

Outputs (under --out):
    distance_matrices/         : per-structure distograms
    index_files/               : per-structure index mapping to reference window
        between_ensembles/
            wasserstein_matrix.npy/.csv/.png
            jensenshannon_matrix.npy/.csv/.png           (JS distance)
            per_residue_jsd_distance.npy/.csv            (from per-residue distance dists)
            metrics_upper_triangle.npz
    analysis_summary.txt
"""

import argparse
import glob
import os
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBParser, Superimposer, is_aa
from Bio.PDB.Polypeptide import index_to_one, three_to_index
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def detect_file_format(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".cif", ".mmcif"):
        return "mmcif"
    if ext == ".pdb":
        return "pdb"
    return "pdb"


def get_cb_like_coord(res, include_altlocs: bool = False):
    if res.get_resname() == "GLY":
        target_atom = res["CA"] if "CA" in res else None
    else:
        target_atom = res["CB"] if "CB" in res else (res["CA"] if "CA" in res else None)
    if target_atom is None:
        return (
            np.array([0.0, 0.0, 0.0])
            if not include_altlocs
            else [np.array([0.0, 0.0, 0.0])]
        )
    if not include_altlocs:
        return target_atom.coord
    if not target_atom.is_disordered():
        return [target_atom.coord]
    coords = []
    for altloc_id, altloc_atom in getattr(target_atom, "child_dict", {}).items():
        coords.append(altloc_atom.coord)
    return coords if coords else [target_atom.coord]


def _superimposer_align(ref_structure, mobile_structure) -> tuple:
    try:
        ref_atoms = []
        mobile_atoms = []
        for model in ref_structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=True) and "CA" in residue:
                        ref_atoms.append(residue["CA"])
        for model in mobile_structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=True) and "CA" in residue:
                        mobile_atoms.append(residue["CA"])
        if not ref_atoms or not mobile_atoms:
            return float("inf"), False
        min_len = min(len(ref_atoms), len(mobile_atoms))
        superimposer = Superimposer()
        superimposer.set_atoms(ref_atoms[:min_len], mobile_atoms[:min_len])
        superimposer.apply(mobile_structure.get_atoms())
        return superimposer.rms, True
    except Exception:
        return float("inf"), False


def setup_output_directories(base_dir: str) -> dict:
    dirs = {
        "base": base_dir,
        "matrices": os.path.join(base_dir, "distance_matrices"),
        "indices": os.path.join(base_dir, "index_files"),
        "between": os.path.join(base_dir, "between_ensembles"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _adaptive_bin_count(n_a: int, n_b: int, min_bins: int, max_bins: int) -> int:
    """Adaptive number of histogram bins based on smaller sample size.

    Uses Sturges' rule on the smaller group size, clamped to [min_bins, max_bins].
    """
    n = max(1, min(int(n_a), int(n_b)))
    # Sturges' rule: ceil(log2(n) + 1)
    bins = int(np.ceil(np.log2(n) + 1))
    return int(max(min_bins, min(max_bins, bins)))


def worker(task):
    """Module-level worker for ProcessPoolExecutor (must be picklable).

    Expects a dict with keys:
    - name, file, mapping, restrict, global_len, chains, align, ref_path,
      include_altlocs, group, output_dirs
    """
    name: str = task["name"]
    file_path: str = task["file"]
    mapping: List[int] = task["mapping"]
    restrict: List[int] = task["restrict"]
    global_len: int = task["global_len"]
    chains: Optional[str] = task["chains"]
    align: bool = task["align"]
    ref_path: Optional[str] = task["ref_path"]
    group_label: str = task["group"]
    include_altlocs: bool = task.get("include_altlocs", False)
    output_dirs: Dict[str, str] = task["output_dirs"]
    try:
        fmt = detect_file_format(file_path)
        parser = MMCIFParser(QUIET=True) if fmt == "mmcif" else PDBParser(QUIET=True)
        struct = parser.get_structure("mob", file_path)
        # optional alignment
        if align and ref_path and os.path.isfile(ref_path):
            try:
                rfmt = detect_file_format(ref_path)
                rparser = (
                    MMCIFParser(QUIET=True)
                    if rfmt == "mmcif"
                    else PDBParser(QUIET=True)
                )
                ref_struct = rparser.get_structure("ref", ref_path)
                _superimposer_align(ref_struct, struct)
            except Exception:
                pass  # best-effort
        want_local = set(chains.split(",")) if chains else None
        residues: List = []
        for model in struct:
            for chain in model:
                if want_local and chain.id not in want_local:
                    continue
                for res in chain:
                    if is_aa(res, standard=True) and "CA" in res:
                        residues.append(res)
        out_name = name.replace(":", "_")
        if not include_altlocs:
            coords = np.full((global_len, 3), np.nan, dtype=np.float32)
            if residues:
                for gidx, r_idx in enumerate(mapping):
                    if r_idx != -1 and r_idx < len(residues):
                        coords[gidx] = get_cb_like_coord(
                            residues[r_idx], include_altlocs=False
                        )
            coords_ref = coords[restrict]
            diff = coords_ref[:, None, :] - coords_ref[None, :, :]
            D = np.sqrt(np.nansum(diff**2, axis=-1)).astype(np.float32)
            missing = np.isnan(coords_ref).any(axis=1)
            if missing.any():
                D[np.repeat(missing[:, None], len(restrict), axis=1)] = np.nan
                D[np.repeat(missing[None, :], len(restrict), axis=0)] = np.nan
            U = np.triu(D, 1)
            D = U + U.T
            np.fill_diagonal(D, 0.0)
            np.save(
                os.path.join(output_dirs["matrices"], f"{out_name}_cbcb_dist.npy"), D
            )
            # Save index file for mapping (once per structure)
            with open(
                os.path.join(output_dirs["indices"], f"{out_name}_index.tsv"), "w"
            ) as fh:
                fh.write("ref_idx\tres_idx\n")
                if residues:
                    for i_ref, gidx in enumerate(restrict):
                        r_idx = mapping[gidx]
                        if r_idx != -1 and r_idx < len(residues):
                            fh.write(f"{i_ref}\t{r_idx}\n")
            return group_label, name, D
        else:
            # Collect altloc coordinate sets per aligned global index
            coord_lists = [None] * global_len
            if residues:
                for gidx, r_idx in enumerate(mapping):
                    if r_idx != -1 and r_idx < len(residues):
                        coord_lists[gidx] = get_cb_like_coord(
                            residues[r_idx], include_altlocs=True
                        )
                    else:
                        coord_lists[gidx] = []
            # Restrict to reference indices
            coord_lists_ref = [coord_lists[g] for g in restrict]
            max_alt = 1
            if coord_lists_ref:
                max_alt = max(
                    (len(lst) if lst is not None else 0) for lst in coord_lists_ref
                )
                if max_alt <= 0:
                    max_alt = 1
            results = []
            for alt_idx in range(max_alt):
                coords_ref = np.full((len(restrict), 3), np.nan, dtype=np.float32)
                for i_ref, lst in enumerate(coord_lists_ref):
                    if lst and len(lst) > 0:
                        c = lst[min(alt_idx, len(lst) - 1)]
                        coords_ref[i_ref] = c
                diff = coords_ref[:, None, :] - coords_ref[None, :, :]
                D = np.sqrt(np.nansum(diff**2, axis=-1)).astype(np.float32)
                missing = np.isnan(coords_ref).any(axis=1)
                if missing.any():
                    D[np.repeat(missing[:, None], len(restrict), axis=1)] = np.nan
                    D[np.repeat(missing[None, :], len(restrict), axis=0)] = np.nan
                U = np.triu(D, 1)
                D = U + U.T
                np.fill_diagonal(D, 0.0)
                # Save each altloc-specific matrix
                np.save(
                    os.path.join(
                        output_dirs["matrices"],
                        f"{out_name}_cbcb_dist_altloc{alt_idx}.npy",
                    ),
                    D,
                )
                results.append(D)
            # Save index once per structure
            with open(
                os.path.join(output_dirs["indices"], f"{out_name}_index.tsv"), "w"
            ) as fh:
                fh.write("ref_idx\tres_idx\n")
                if residues:
                    for i_ref, gidx in enumerate(restrict):
                        r_idx = mapping[gidx]
                        if r_idx != -1 and r_idx < len(residues):
                            fh.write(f"{i_ref}\t{r_idx}\n")
            return group_label, name, results
    except Exception as e:
        return group_label, name, f"ERROR: {e}"


def main():
    ap = argparse.ArgumentParser(
        description="Compare two ensembles via CB–CB distograms using Wasserstein-1 and Jensen–Shannon distances per position (i<j)"
    )
    ap.add_argument(
        "--group-a", required=True, help="Folder of PDB/mmCIF files for Group A"
    )
    ap.add_argument(
        "--group-b", required=True, help="Folder of PDB/mmCIF files for Group B"
    )
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--chains", help="Comma-separated chain IDs to include (optional)")
    ap.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes (default: CPU count)",
    )
    ap.add_argument(
        "--align-structures",
        action="store_true",
        help="Optionally align structures to a reference using Superimposer before computing distances (does not change internal distances, for parity only)",
    )
    ap.add_argument(
        "--include-altlocs",
        action="store_true",
        help="Include altlocs by generating multiple per-structure distograms (one per altloc index); these are treated as additional samples in each group",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=64,
        help="Max number of bins for JS histogram (used as upper bound for adaptive binning)",
    )
    ap.add_argument(
        "--min-bins",
        type=int,
        default=5,
        help="Min number of bins for JS histogram (lower bound for adaptive binning)",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum samples required per group for a cell to compute metrics",
    )
    ap.add_argument(
        "--reference-structure",
        help="Optional reference structure file to define sequence window",
    )
    ap.add_argument(
        "--reference-sequence",
        help="Optional reference one-letter sequence (used if no reference structure provided)",
    )
    args = ap.parse_args()

    output_dirs = setup_output_directories(args.out)

    def find_structures(folder: str) -> List[str]:
        files = []
        files.extend(glob.glob(os.path.join(folder, "*.pdb")))
        files.extend(glob.glob(os.path.join(folder, "*.cif")))
        files.extend(glob.glob(os.path.join(folder, "*.mmcif")))
        return sorted(files)

    group_a_files = find_structures(args.group_a)
    group_b_files = find_structures(args.group_b)
    if not group_a_files or not group_b_files:
        raise SystemExit("Both --group-a and --group-b must contain PDB/mmCIF files.")

    print(
        f"Group A structures: {len(group_a_files)} | Group B structures: {len(group_b_files)}"
    )

    want = set(args.chains.split(",")) if args.chains else None

    # Parse structures for both groups
    struct_data: Dict[str, Dict] = {}
    group_of: Dict[str, str] = {}

    def parse_folder(files: List[str], group_label: str):
        for structure_file in files:
            fmt = detect_file_format(structure_file)
            base = os.path.splitext(os.path.basename(structure_file))[0]
            try:
                parser = (
                    MMCIFParser(QUIET=True) if fmt == "mmcif" else PDBParser(QUIET=True)
                )
                struct = parser.get_structure("s", structure_file)
            except Exception as e:
                print(f"  [FAIL] {base} ({group_label}): parse error {e}")
                continue
            residues: List = []
            for model in struct:
                for chain in model:
                    if want and chain.id not in want:
                        continue
                    for res in chain:
                        if is_aa(res, standard=True) and "CA" in res:
                            residues.append(res)
            if not residues:
                print(f"  [SKIP] {base} ({group_label}): no residues after filtering")
                continue
            seq_chars = []
            for r in residues:
                try:
                    seq_chars.append(index_to_one(three_to_index(r.get_resname())))
                except KeyError:
                    seq_chars.append("X")
            sequence = "".join(seq_chars)
            key = f"{group_label}:{base}"
            struct_data[key] = {
                "file": structure_file,
                "format": fmt,
                "residues": residues,
                "sequence": sequence,
            }
            group_of[key] = group_label

    parse_folder(group_a_files, "A")
    parse_folder(group_b_files, "B")

    if not struct_data:
        raise SystemExit("No valid structures to process after parsing.")

    # Reference sequence
    if args.reference_structure:
        ref_path = args.reference_structure
        ref_base = os.path.splitext(os.path.basename(ref_path))[0]
        key = f"REF:{ref_base}"
        if key not in struct_data:
            fmt = detect_file_format(ref_path)
            try:
                parser = (
                    MMCIFParser(QUIET=True) if fmt == "mmcif" else PDBParser(QUIET=True)
                )
                ref_struct = parser.get_structure("ref", ref_path)
            except Exception as e:
                raise SystemExit(f"Failed to parse reference structure: {e}")
            residues = []
            for model in ref_struct:
                for chain in model:
                    for res in chain:
                        if is_aa(res, standard=True) and "CA" in res:
                            residues.append(res)
            if not residues:
                raise SystemExit("Reference structure has no residues.")
            seq_chars = []
            for r in residues:
                try:
                    seq_chars.append(index_to_one(three_to_index(r.get_resname())))
                except KeyError:
                    seq_chars.append("X")
            struct_data[key] = {
                "file": ref_path,
                "format": fmt,
                "residues": residues,
                "sequence": "".join(seq_chars),
            }
            group_of[key] = "REF"
        ref_name = key
    elif args.reference_sequence:
        ref_name = "REF:sequence"
        struct_data[ref_name] = {
            "file": None,
            "format": "seq",
            "residues": [],
            "sequence": args.reference_sequence.strip(),
        }
        group_of[ref_name] = "REF"
    else:
        ref_name = max(struct_data.items(), key=lambda kv: len(kv[1]["sequence"]))[0]

    ref_seq = struct_data[ref_name]["sequence"]
    ref_len = len(ref_seq)
    print(f"Reference: {ref_name} (length {ref_len})")

    # Pairwise alignment to reference to derive mapping layout
    insertion_after = [0] * (ref_len + 1)
    alignments: Dict[str, Tuple[str, str]] = {}
    aligner = PairwiseAligner()

    for name, data in struct_data.items():
        seq = data["sequence"]
        if name == ref_name:
            alignments[name] = (ref_seq, seq)
            continue
        aln = aligner.align(ref_seq, seq)[0]
        blocksA = aln.aligned[0]
        blocksB = aln.aligned[1]
        aligned_ref_chars = []
        aligned_seq_chars = []
        ref_pos_raw = 0
        seq_pos_raw = 0
        for (a_start, a_end), (b_start, b_end) in zip(blocksA, blocksB):
            while seq_pos_raw < b_start:
                aligned_ref_chars.append("-")
                aligned_seq_chars.append(seq[seq_pos_raw])
                seq_pos_raw += 1
            while ref_pos_raw < a_start:
                aligned_ref_chars.append(ref_seq[ref_pos_raw])
                aligned_seq_chars.append("-")
                ref_pos_raw += 1
            for i2 in range(a_end - a_start):
                aligned_ref_chars.append(ref_seq[a_start + i2])
                aligned_seq_chars.append(seq[b_start + i2])
            ref_pos_raw = a_end
            seq_pos_raw = b_end
        while seq_pos_raw < len(seq):
            aligned_ref_chars.append("-")
            aligned_seq_chars.append(seq[seq_pos_raw])
            seq_pos_raw += 1
        while ref_pos_raw < len(ref_seq):
            aligned_ref_chars.append(ref_seq[ref_pos_raw])
            aligned_seq_chars.append("-")
            ref_pos_raw += 1
        aligned_ref = "".join(aligned_ref_chars)
        aligned_seq = "".join(aligned_seq_chars)
        alignments[name] = (aligned_ref, aligned_seq)
        # record insertion sizes
        ref_pos = -1
        i = 0
        L_aln = len(aligned_ref)
        while i < L_aln:
            if aligned_ref[i] == "-" and aligned_seq[i] != "-":
                bl = 0
                while i < L_aln and aligned_ref[i] == "-" and aligned_seq[i] != "-":
                    bl += 1
                    i += 1
                block_index = ref_pos + 1
                if bl > insertion_after[block_index]:
                    insertion_after[block_index] = bl
            else:
                if aligned_ref[i] != "-":
                    ref_pos += 1
                i += 1

    # Global index layout; restrict to reference positions
    block_start_index: List[int] = []
    ref_global_index: List[int] = []
    idx = 0
    for k in range(ref_len + 1):
        block_start_index.append(idx)
        idx += insertion_after[k]
        if k < ref_len:
            ref_global_index.append(idx)
            idx += 1
    global_length = idx
    restrict_to_ref = ref_global_index
    L_ref = len(restrict_to_ref)
    print(f"Global aligned length: {global_length} | Reference window length: {L_ref}")

    def build_mapping(aligned_ref: str, aligned_seq: str) -> List[int]:
        mapping = [-1] * global_length
        ref_pos = -1
        ins_offsets_used = [0] * (ref_len + 1)
        seq_res_idx = 0
        i = 0
        L_aln = len(aligned_ref)
        while i < L_aln:
            if aligned_ref[i] == "-" and aligned_seq[i] != "-":
                block_index = ref_pos + 1
                offset = ins_offsets_used[block_index]
                gidx = block_start_index[block_index] + offset
                if gidx < global_length:
                    mapping[gidx] = seq_res_idx
                ins_offsets_used[block_index] += 1
                seq_res_idx += 1
                i += 1
            else:
                if aligned_ref[i] != "-":
                    ref_pos += 1
                    if aligned_seq[i] != "-":
                        gidx = ref_global_index[ref_pos]
                        mapping[gidx] = seq_res_idx
                        seq_res_idx += 1
                i += 1
        return mapping

    # Build per-structure matrices and split to groups (parallel)
    mats_A: List[np.ndarray] = []
    mats_B: List[np.ndarray] = []

    # choose a reference structure file for optional rigid-body alignment
    ref_structure_path: Optional[str] = None
    if args.align_structures:
        # Prefer a provided reference structure; else first Group A file
        if args.reference_structure and os.path.isfile(args.reference_structure):
            ref_structure_path = args.reference_structure
        else:
            ref_candidates = group_a_files if group_a_files else (group_b_files or [])
            ref_structure_path = ref_candidates[0] if ref_candidates else None

    tasks = []
    for name, data in struct_data.items():
        if group_of.get(name) not in ("A", "B"):
            continue
        aligned_ref, aligned_seq = alignments[name]
        mapping = build_mapping(aligned_ref, aligned_seq)
        tasks.append(
            {
                "name": name,
                "file": data["file"],
                "mapping": mapping,
                "restrict": restrict_to_ref,
                "global_len": global_length,
                "chains": args.chains,
                "align": args.align_structures,
                "ref_path": ref_structure_path,
                "include_altlocs": args.include_altlocs,
                "group": group_of[name],
                "output_dirs": output_dirs,
            }
        )

    with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            group_label, name, result = fut.result()
            if isinstance(result, str) and result.startswith("ERROR:"):
                print(f"  [FAIL] {name}: {result}")
                continue
            if group_label == "A":
                if isinstance(result, list):
                    mats_A.extend(result)
                else:
                    mats_A.append(result)
            else:
                if isinstance(result, list):
                    mats_B.extend(result)
                else:
                    mats_B.append(result)

    if not mats_A or not mats_B:
        raise SystemExit("Failed to accumulate matrices for one or both groups.")

    A = np.stack(mats_A).astype(np.float32)
    B = np.stack(mats_B).astype(np.float32)
    print(f"Stacks built: A {A.shape}, B {B.shape}")

    # Metrics per upper triangle
    iu, ju = np.triu_indices(L_ref, k=1)
    M = len(iu)
    w_tri = np.full(M, np.nan, dtype=np.float32)
    js_dist_tri = np.full(M, np.nan, dtype=np.float32)

    # Early check: ensure there is at least some finite data overall
    if not np.isfinite(A).any() or not np.isfinite(B).any():
        raise SystemExit("No finite distance values to compute metrics.")
    for k, (i, j) in enumerate(zip(iu, ju)):
        a = A[:, i, j]
        b = B[:, i, j]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < args.min_samples or b.size < args.min_samples:
            continue
        # Wasserstein-1 on raw distances
        try:
            w_val = float(wasserstein_distance(a, b))
        except Exception:
            w_val = np.nan

        # Adaptive-binning JSD on histograms with shared per-pair edges
        n_bins = _adaptive_bin_count(a.size, b.size, args.min_bins, args.bins)
        vmin = float(min(a.min() if a.size else np.inf, b.min() if b.size else np.inf))
        vmax = float(
            max(a.max() if a.size else -np.inf, b.max() if b.size else -np.inf)
        )
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        edges = np.linspace(vmin, vmax, n_bins + 1, dtype=np.float64)
        ha, _ = np.histogram(a, bins=edges)
        hb, _ = np.histogram(b, bins=edges)
        pa = ha.astype(np.float64)
        pb = hb.astype(np.float64)
        sa = pa.sum()
        sb = pb.sum()
        if sa <= 0 or sb <= 0:
            js_d = np.nan
        else:
            pa /= sa
            pb /= sb
            try:
                js_d = float(jensenshannon(pa, pb, base=2.0))
            except Exception:
                js_d = np.nan
        w_tri[k] = w_val
        js_dist_tri[k] = js_d

    W = np.full((L_ref, L_ref), np.nan, dtype=np.float32)
    JS_DIST = np.full((L_ref, L_ref), np.nan, dtype=np.float32)
    W[iu, ju] = w_tri
    JS_DIST[iu, ju] = js_dist_tri
    W[ju, iu] = w_tri
    JS_DIST[ju, iu] = js_dist_tri
    np.fill_diagonal(W, 0.0)
    np.fill_diagonal(JS_DIST, 0.0)

    os.makedirs(output_dirs["between"], exist_ok=True)
    np.save(os.path.join(output_dirs["between"], "wasserstein_matrix.npy"), W)
    # Save the single JS distance matrix
    np.save(os.path.join(output_dirs["between"], "jensenshannon_matrix.npy"), JS_DIST)
    np.savez_compressed(
        os.path.join(output_dirs["between"], "metrics_upper_triangle.npz"),
        iu=iu,
        ju=ju,
        wasserstein=w_tri,
        jensenshannon=js_dist_tri,
    )
    np.savetxt(
        os.path.join(output_dirs["between"], "wasserstein_matrix.csv"),
        W,
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(output_dirs["between"], "jensenshannon_matrix.csv"),
        JS_DIST,
        delimiter=",",
        fmt="%.6f",
    )

    # Heatmaps (ensure symmetry already enforced)
    cmap = plt.cm.magma.copy()
    cmap.set_bad("black")

    def save_heatmap(mat: np.ndarray, title: str, out_png: str, set_vmax: Optional[float] = None):
        finite_vals = mat[np.isfinite(mat)]
        vmax = np.percentile(finite_vals, 95) if finite_vals.size else 1.0
        vmax = set_vmax if set_vmax else vmax
        plt.figure(figsize=(8, 6), dpi=150)
        plt.imshow(mat, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
        plt.colorbar(label=title)
        plt.xlabel("Reference index")
        plt.ylabel("Reference index")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

    save_heatmap(
        W,
        "Wasserstein-1 (A vs B)",
        os.path.join(output_dirs["between"], "wasserstein_matrix.png"),
    )
    save_heatmap(
        JS_DIST,
        "Jensen–Shannon distance (A vs B)",
        os.path.join(output_dirs["between"], "jensenshannon_matrix.png"),
        set_vmax = 1.0,
    )

    # Summary
    summary_file = os.path.join(output_dirs["base"], "analysis_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Between-ensemble Distogram Metrics Summary\n")
        f.write("=" * 44 + "\n\n")
        f.write(f"Group A: {args.group_a} (n={A.shape[0]})\n")
        f.write(f"Group B: {args.group_b} (n={B.shape[0]})\n")
        f.write(f"Chains: {args.chains if args.chains else 'all'}\n")
        f.write(f"Reference: {ref_name} length {ref_len}\n")
        f.write(f"Reference window length: {L_ref}\n")
        f.write(
            f"JS adaptive bins: min={args.min_bins}, max={args.bins} (per-pair Sturges rule)\n"
        )
        f.write(f"Min samples per group per cell: {args.min_samples}\n")
        valid_w = np.isfinite(W[iu, ju]).sum()
        valid_jsd = np.isfinite(JS_DIST[iu, ju]).sum()
        f.write(f"Valid W1 pairs: {valid_w}/{len(iu)}\n")
        f.write(f"Valid JS distance pairs: {valid_jsd}/{len(iu)}\n")
    print(f"Done. Metrics saved to {output_dirs['between']}")


if __name__ == "__main__":
    main()
