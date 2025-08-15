#!/usr/bin/env python3
"""
CB–CB distance map (from PDB and mmCIF files)

Usage:
  python calc_cb_distogram.py --structure-folder your_structures --out out
  # optional chain filter (comma-separated IDs):
  python calc_cb_distogram.py --structure-folder your_structures --chains A,B --out out
  # include altlocs in variance calculation:
  python calc_cb_distogram.py --structure-folder your_structures --out out --include-altlocs

Outputs:
  out_cbcb_dist.npy   : LxL distances (Å)
  out_cbcb_dist.csv   : CSV distances (Å)
  out_cbcb_heatmap.png: heatmap
  out_index.tsv       : mapping of row/col -> chain, resi, resname
  out_variance_decomposition_heatmap.png: variance heatmap
"""

import argparse
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, is_aa, Superimposer
from Bio.PDB.cealign import CEAligner
from Bio.PDB.Polypeptide import three_to_index, index_to_one
from Bio.Align import PairwiseAligner
import matplotlib.pyplot as plt
import glob
import os
from typing import List, Dict, Tuple


def get_cb_like_coord(res, include_altlocs=False):
    """CB for most residues; CA for Gly or if CB missing.

    Args:
        res: BioPython residue object
        include_altlocs: If True, return all altloc coordinates for disordered atoms

    Returns:
        If include_altlocs=False: single coordinate (3,) array
        If include_altlocs=True: list of coordinates, one for each altloc
    """
    if res.get_resname() == "GLY":
        target_atom = res["CA"] if "CA" in res else None
    else:
        target_atom = res["CB"] if "CB" in res else res["CA"] if "CA" in res else None

    if target_atom is None:
        if include_altlocs:
            return [np.array([0.0, 0.0, 0.0])]
        return np.array([0.0, 0.0, 0.0])

    if not include_altlocs:
        return target_atom.coord

    # When include_altlocs is True, always return a list
    if not target_atom.is_disordered():
        return [target_atom.coord]

    # Handle disordered atoms - collect all altloc coordinates
    coords = []
    if hasattr(target_atom, "child_dict"):
        # This is a DisorderedAtom - get all altlocs
        for altloc_id in target_atom.child_dict.keys():
            altloc_atom = target_atom.child_dict[altloc_id]
            coords.append(altloc_atom.coord)
    else:
        # Fallback - just use the current coordinate
        coords.append(target_atom.coord)

    return coords


def compute_cbcb(residues, include_altlocs=False):
    """Compute CB-CB distance matrix (optimized).

    Args:
        residues: List of BioPython residue objects
        include_altlocs: If True, include altloc variance in calculation

    Returns:
        Distance matrix or list of distance matrices (if altlocs included)
    """
    if not include_altlocs:
        coords = np.array(
            [get_cb_like_coord(r, False) for r in residues], dtype=np.float32
        )  # Use float32 to save memory
        # Vectorized distance calculation
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))  # (L,L)

    # Handle altlocs - create multiple matrices (optimized)
    all_coord_sets = [get_cb_like_coord(r, True) for r in residues]

    # Find maximum number of altlocs across all residues
    max_altlocs = max(len(coord_set) for coord_set in all_coord_sets)

    # Pre-allocate arrays for better performance
    n_residues = len(residues)
    matrices = []

    for altloc_idx in range(max_altlocs):
        # Pre-allocate coordinate array
        coords = np.zeros((n_residues, 3), dtype=np.float32)

        for i, coord_set in enumerate(all_coord_sets):
            # Use the first coordinate if this altloc doesn't exist for this residue
            coord_idx = min(altloc_idx, len(coord_set) - 1)
            coords[i] = coord_set[coord_idx]

        # Vectorized distance calculation
        diff = coords[:, None, :] - coords[None, :, :]
        matrix = np.sqrt(np.sum(diff**2, axis=-1))
        matrices.append(matrix)

    return matrices


def plot_heatmap(matrix, title, out_png, vmin=0.0, vmax=None):
    plt.figure(figsize=(6.5, 5.5), dpi=150)
    im = plt.imshow(matrix, origin="lower", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label("Å")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def detect_file_format(filepath: str) -> str:
    """Detect file format based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".cif", ".mmcif"]:
        return "mmcif"
    elif ext == ".pdb":
        return "pdb"
    else:
        # Try to guess from content or default to PDB
        return "pdb"


def process_single_structure(
    structure_file: str,
    output_dirs: dict,
    chains: str = None,
    include_altlocs: bool = False,
) -> dict:
    """Process a single structure file and return results.

    Args:
        structure_file: Path to structure file
        output_dirs: Dictionary of output directories
        chains: Comma-separated chain IDs (optional)
        include_altlocs: Whether to include altlocs

    Returns:
        Dictionary with matrices and metadata
    """
    file_format = detect_file_format(structure_file)
    file_base = os.path.splitext(os.path.basename(structure_file))[0]

    try:
        # Initialize parsers (fresh for each process)
        if file_format == "mmcif":
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        struct = parser.get_structure("s", structure_file)
    except Exception as e:
        return {"error": f"Error parsing {structure_file}: {e}"}

    want = set(chains.split(",")) if chains else None
    residues = []
    for model in struct:
        for chain in model:
            if want and chain.id not in want:
                continue
            for res in chain:
                if is_aa(res, standard=True) and "CA" in res:
                    residues.append(res)

    if not residues:
        return {"error": f"No protein residues found in {structure_file}"}

    # Compute distances
    D_result = compute_cbcb(residues, include_altlocs)

    # Save files
    matrices = []
    if include_altlocs and isinstance(D_result, list):
        matrices = D_result
        for i, D in enumerate(D_result):
            altloc_suffix = f"_altloc{i}" if len(D_result) > 1 else ""
            # Save matrices in organized folders
            np.save(
                os.path.join(
                    output_dirs["matrices"], f"{file_base}{altloc_suffix}_cbcb_dist.npy"
                ),
                D,
            )
            np.savetxt(
                os.path.join(
                    output_dirs["matrices"], f"{file_base}{altloc_suffix}_cbcb_dist.csv"
                ),
                D,
                delimiter=",",
                fmt="%.3f",
            )

            vmax = np.percentile(D, 99) if D.size else None
            plot_heatmap(
                D,
                f"CB–CB distances (Å) for {file_base}{altloc_suffix}",
                os.path.join(
                    output_dirs["heatmaps"],
                    f"{file_base}{altloc_suffix}_cbcb_heatmap.png",
                ),
                vmin=0.0,
                vmax=vmax,
            )
    else:
        D = D_result
        matrices = [D]
        np.save(os.path.join(output_dirs["matrices"], f"{file_base}_cbcb_dist.npy"), D)
        np.savetxt(
            os.path.join(output_dirs["matrices"], f"{file_base}_cbcb_dist.csv"),
            D,
            delimiter=",",
            fmt="%.3f",
        )

        vmax = np.percentile(D, 99) if D.size else None
        plot_heatmap(
            D,
            f"CB–CB distances (Å) for {file_base}",
            os.path.join(output_dirs["heatmaps"], f"{file_base}_cbcb_heatmap.png"),
            vmin=0.0,
            vmax=vmax,
        )

    # Index map
    with open(
        os.path.join(output_dirs["indices"], f"{file_base}_index.tsv"), "w"
    ) as fh:
        fh.write("idx\tchain\tresi\ticode\tresname\n")
        for i, r in enumerate(residues):
            chain = r.get_parent().id
            (_, resi, icode) = r.get_id()
            fh.write(
                f"{i}\t{chain}\t{resi}\t{icode if icode.strip() else ' '}\t{r.get_resname()}\n"
            )

    return {
        "matrices": matrices,
        "file_base": file_base,
        "file_format": file_format,
        "n_residues": len(residues),
    }


def setup_output_directories(base_dir: str) -> dict:
    """Create organized output directory structure."""
    dirs = {
        "base": base_dir,
        "matrices": os.path.join(base_dir, "distance_matrices"),
        "heatmaps": os.path.join(base_dir, "heatmaps"),
        "indices": os.path.join(base_dir, "index_files"),
        "variance": os.path.join(base_dir, "variance_analysis"),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def align_ensemble_structures(
    structure_files: list,
    reference_index: int = 0,
    use_ce_aligner: bool = True,
    ce_window_size: int = 8,
    ce_max_gap: int = 30,
    output_dir: str = None,
) -> dict:
    """Align ensemble structures to a reference structure for better distance calculations.

    Args:
        structure_files: List of structure file paths
        reference_index: Index of reference structure (default: 0)
        use_ce_aligner: Use CEAligner if True, else Superimposer
        ce_window_size: CEAligner window size parameter
        ce_max_gap: CEAligner max gap parameter
        output_dir: Directory to save aligned structures (optional)

    Returns:
        Dictionary with alignment results and metadata
    """
    if reference_index >= len(structure_files):
        reference_index = 0

    ref_file = structure_files[reference_index]
    ref_format = detect_file_format(ref_file)

    # Parse reference structure
    if ref_format == "mmcif":
        ref_parser = MMCIFParser(QUIET=True)
    else:
        ref_parser = PDBParser(QUIET=True)

    ref_structure = ref_parser.get_structure("reference", ref_file)

    alignment_results = {}
    alignment_metadata = {}

    # Initialize alignment tool
    if use_ce_aligner:
        try:
            aligner = CEAligner(window_size=ce_window_size, max_gap=ce_max_gap)
            aligner.set_reference(ref_structure)
            alignment_method = "CEAligner"
        except Exception as e:
            print(
                f"CEAligner initialization failed: {e}. Falling back to Superimposer."
            )
            aligner = None
            use_ce_aligner = False
            alignment_method = "Superimposer_fallback"
    else:
        aligner = None
        alignment_method = "Superimposer"

    for i, structure_file in enumerate(structure_files):
        file_base = os.path.splitext(os.path.basename(structure_file))[0]

        if i == reference_index:
            # Reference structure
            alignment_results[file_base] = {
                "rmsd": 0.0,
                "aligned": True,
                "method": "reference",
            }
            alignment_metadata[file_base] = {
                "alignment_length": 0,
                "sequence_coverage": 1.0,
                "method": "reference",
            }
            continue

        # Load mobile structure
        mobile_format = detect_file_format(structure_file)
        if mobile_format == "mmcif":
            mobile_parser = MMCIFParser(QUIET=True)
        else:
            mobile_parser = PDBParser(QUIET=True)

        try:
            mobile_structure = mobile_parser.get_structure("mobile", structure_file)

            # Perform alignment
            if use_ce_aligner and aligner is not None:
                try:
                    aligner.align(mobile_structure, transform=True)
                    rmsd = aligner.rms  # Use aligner.rms property directly
                    alignment_length = getattr(aligner, "alignment_length", 0)
                    sequence_coverage = getattr(aligner, "sequence_coverage", 0.0)
                    success = True
                except Exception as e:
                    print(
                        f"CEAligner failed for {file_base}: {e}. Using Superimposer fallback."
                    )
                    rmsd, success = _superimposer_align(ref_structure, mobile_structure)
                    alignment_length = 0
                    sequence_coverage = 0.0
                    alignment_method = "Superimposer_fallback"
            else:
                rmsd, success = _superimposer_align(ref_structure, mobile_structure)
                alignment_length = 0
                sequence_coverage = 0.0

            alignment_results[file_base] = {
                "rmsd": rmsd,
                "aligned": success,
                "method": alignment_method,
            }
            alignment_metadata[file_base] = {
                "alignment_length": alignment_length,
                "sequence_coverage": sequence_coverage,
                "method": alignment_method,
            }

            # Save aligned structure if output directory specified
            if output_dir and success:
                os.makedirs(output_dir, exist_ok=True)
                if mobile_format == "mmcif":
                    from Bio.PDB.mmcifio import MMCIFIO

                    io = MMCIFIO()
                    io.set_structure(mobile_structure)
                    io.save(os.path.join(output_dir, f"aligned_{file_base}.cif"))
                else:
                    from Bio.PDB import PDBIO

                    io = PDBIO()
                    io.set_structure(mobile_structure)
                    io.save(os.path.join(output_dir, f"aligned_{file_base}.pdb"))

        except Exception as e:
            print(f"Error processing {structure_file}: {e}")
            alignment_results[file_base] = {
                "rmsd": float("inf"),
                "aligned": False,
                "method": "failed",
                "error": str(e),
            }
            alignment_metadata[file_base] = {
                "alignment_length": 0,
                "sequence_coverage": 0.0,
                "method": "failed",
                "error": str(e),
            }

    return {
        "alignment_results": alignment_results,
        "alignment_metadata": alignment_metadata,
        "reference_structure": os.path.basename(ref_file),
        "reference_index": reference_index,
        "alignment_method": alignment_method,
        "n_structures": len(structure_files),
        "n_aligned": sum(1 for r in alignment_results.values() if r["aligned"]),
    }


def _superimposer_align(ref_structure, mobile_structure) -> tuple:
    """Helper function for Superimposer-based alignment"""
    try:
        # Extract CA atoms
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

        # Use minimum length for alignment
        min_len = min(len(ref_atoms), len(mobile_atoms))

        superimposer = Superimposer()
        superimposer.set_atoms(ref_atoms[:min_len], mobile_atoms[:min_len])
        superimposer.apply(mobile_structure.get_atoms())

        return superimposer.rms, True

    except Exception:
        return float("inf"), False


def main():
    ap = argparse.ArgumentParser(
        description="Compute CB-CB distance maps from PDB/mmCIF structures"
    )
    ap.add_argument(
        "--structure-folder", required=True, help="folder containing PDB/mmCIF files"
    )
    ap.add_argument("--out", required=True, help="output directory name")
    ap.add_argument("--chains", help="comma-separated chain IDs (optional)")
    ap.add_argument(
        "--include-altlocs",
        action="store_true",
        help="include altloc variance in calculations",
    )
    ap.add_argument(
        "--plot-name", default="CB-CB", help="name for plots (default: CB-CB)"
    )
    ap.add_argument(
        "--align-structures",
        action="store_true",
        help="align structures before computing distances",
    )
    ap.add_argument(
        "--no-ce-aligner",
        action="store_true",
        help="use Superimposer instead of CEAligner for alignment",
    )
    ap.add_argument(
        "--ce-window-size",
        type=int,
        default=8,
        help="CEAligner window size parameter (default: 8)",
    )
    ap.add_argument(
        "--ce-max-gap",
        type=int,
        default=30,
        help="CEAligner max gap parameter (default: 30)",
    )
    ap.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="index of reference structure for alignment (default: 0)",
    )
    ap.add_argument(
        "--reference-structure",
        help="Path to structure file to use as reference for sequence alignment (optional)",
    )
    ap.add_argument(
        "--reference-sequence",
        help="One-letter amino acid sequence to use as reference (used only if --reference-structure not provided)",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum number of matrices required to compute variance for a cell (default: 3)",
    )
    args = ap.parse_args()

    # Setup organized output directories
    output_dirs = setup_output_directories(args.out)

    # Find both PDB and mmCIF files
    structure_files = []
    structure_files.extend(glob.glob(f"{args.structure_folder}/*.pdb"))
    structure_files.extend(glob.glob(f"{args.structure_folder}/*.cif"))
    structure_files.extend(glob.glob(f"{args.structure_folder}/*.mmcif"))

    if not structure_files:
        raise SystemExit("No PDB/mmCIF files found in the specified folder.")

    # Optional structure alignment for better ensemble analysis
    alignment_info = None
    if args.align_structures:
        print(f"Aligning {len(structure_files)} structures to reference...")
        alignment_output_dir = os.path.join(output_dirs["base"], "aligned_structures")

        try:
            alignment_info = align_ensemble_structures(
                structure_files=structure_files,
                reference_index=args.reference_index,
                use_ce_aligner=not args.no_ce_aligner,
                ce_window_size=args.ce_window_size,
                ce_max_gap=args.ce_max_gap,
                output_dir=alignment_output_dir,
            )

            print(
                f"Alignment complete: {alignment_info['n_aligned']}/{alignment_info['n_structures']} structures aligned"
            )
            print(f"Alignment method: {alignment_info['alignment_method']}")
            print(f"Reference structure: {alignment_info['reference_structure']}")

            # Save alignment metadata
            alignment_file = os.path.join(
                output_dirs["base"], "alignment_metadata.json"
            )
            import json

            with open(alignment_file, "w") as f:
                json.dump(alignment_info, f, indent=2)
            print(f"Alignment metadata saved to: {alignment_file}")

        except Exception as e:
            print(f"Structure alignment failed: {e}")
            print("Proceeding with original structures...")
            alignment_info = None

    # ------------------------------------------------------------------
    # New sequence-alignment-based matrix construction with NaN padding
    # ------------------------------------------------------------------

    if args.include_altlocs:
        print(
            "[WARNING] Altloc variance currently not combined with sequence alignment padding; proceeding treating only primary conformers."
        )

    print(f"Parsing {len(structure_files)} structures & extracting sequences...")

    want = set(args.chains.split(",")) if args.chains else None

    struct_data: Dict[str, Dict] = {}
    failed_count = 0

    for structure_file in structure_files:
        file_format = detect_file_format(structure_file)
        file_base = os.path.splitext(os.path.basename(structure_file))[0]
        try:
            parser = (
                MMCIFParser(QUIET=True)
                if file_format == "mmcif"
                else PDBParser(QUIET=True)
            )
            struct = parser.get_structure("s", structure_file)
        except Exception as e:
            print(f"  [FAIL] {file_base}: parse error {e}")
            failed_count += 1
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
            print(f"  [SKIP] {file_base}: no residues after filtering")
            failed_count += 1
            continue

        # Build sequence
        seq_chars = []
        for r in residues:
            try:
                seq_chars.append(index_to_one(three_to_index(r.get_resname())))
            except KeyError:
                seq_chars.append("X")
        sequence = "".join(seq_chars)

        struct_data[file_base] = {
            "file": structure_file,
            "format": file_format,
            "residues": residues,
            "sequence": sequence,
        }

    if not struct_data:
        raise SystemExit("No valid structures to process.")

    # Determine reference sequence & (optional) structure
    ref_name = None
    if args.reference_structure:
        ref_path = args.reference_structure
        ref_base = os.path.splitext(os.path.basename(ref_path))[0]
        if ref_base not in struct_data:
            # Load reference structure additionally
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
            struct_data[ref_base] = {
                "file": ref_path,
                "format": fmt,
                "residues": residues,
                "sequence": "".join(seq_chars),
            }
        ref_name = ref_base
    elif args.reference_sequence:
        ref_name = "reference_sequence"
        struct_data[ref_name] = {
            "file": None,
            "format": "seq",
            "residues": [],
            "sequence": args.reference_sequence.strip(),
        }
    else:
        # Choose longest sequence as reference
        ref_name = max(struct_data.items(), key=lambda kv: len(kv[1]["sequence"]))[0]

    ref_seq = struct_data[ref_name]["sequence"]
    ref_len = len(ref_seq)
    print(f"Reference: {ref_name} (length {ref_len})")

    # First pass: align each sequence to ref using Biopython PairwiseAligner; record insertion maxima
    insertion_after = [0] * (ref_len + 1)
    alignments: Dict[str, Tuple[str, str]] = {}
    aligner = PairwiseAligner()
    # Use a mild gap penalty profile (defaults often fine); user can adjust if needed later

    for name, data in struct_data.items():
        seq = data["sequence"]
        if name == ref_name:
            alignments[name] = (ref_seq, seq)
            continue
        aln = aligner.align(ref_seq, seq)[0]
        # Reconstruct gapped sequences from aligned coordinate blocks
        blocksA = aln.aligned[0]
        blocksB = aln.aligned[1]
        aligned_ref_chars = []
        aligned_seq_chars = []
        ref_pos_raw = 0
        seq_pos_raw = 0
        for (a_start, a_end), (b_start, b_end) in zip(blocksA, blocksB):
            while seq_pos_raw < b_start:  # insertion relative to ref
                aligned_ref_chars.append("-")
                aligned_seq_chars.append(seq[seq_pos_raw])
                seq_pos_raw += 1
            while ref_pos_raw < a_start:  # deletion relative to ref
                aligned_ref_chars.append(ref_seq[ref_pos_raw])
                aligned_seq_chars.append("-")
                ref_pos_raw += 1
            # matched block
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
        # Record insertion sizes between reference residues
        ref_pos = -1
        i = 0
        L = len(aligned_ref)
        while i < L:
            if aligned_ref[i] == "-" and aligned_seq[i] != "-":
                block_len = 0
                while i < L and aligned_ref[i] == "-" and aligned_seq[i] != "-":
                    block_len += 1
                    i += 1
                block_index = ref_pos + 1
                if block_len > insertion_after[block_index]:
                    insertion_after[block_index] = block_len
            else:
                if aligned_ref[i] != "-":
                    ref_pos += 1
                i += 1

    # Precompute global index layout
    block_start_index = []  # start index of each insertion block
    ref_global_index = []  # global index of each reference residue
    idx = 0
    for k in range(ref_len + 1):
        block_start_index.append(idx)
        idx += insertion_after[k]
        if k < ref_len:
            ref_global_index.append(idx)
            idx += 1
    global_length = idx
    print(f"Global aligned length (with insertions): {global_length}")

    # Second pass: build per-structure global mappings & distance matrices
    matrices = []
    names_in_order = []

    def build_mapping(aligned_ref: str, aligned_seq: str) -> List[int]:
        mapping = [-1] * global_length  # global index -> residue idx in this structure
        ref_pos = -1
        ins_offsets_used = [0] * (ref_len + 1)
        seq_res_idx = 0
        i = 0
        L = len(aligned_ref)
        while i < L:
            if aligned_ref[i] == "-" and aligned_seq[i] != "-":
                # insertion
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
                else:
                    # gap in reference? (should be handled above) safe advance
                    pass
                if aligned_seq[i] == "-" and aligned_ref[i] != "-":
                    # gap in sequence; nothing to map
                    pass
                i += 1
        return mapping

    for name, data in struct_data.items():
        aligned_ref, aligned_seq = alignments[name]
        mapping = build_mapping(aligned_ref, aligned_seq)
        residues = data["residues"]
        # Build coordinate array with NaNs
        coords = np.full((global_length, 3), np.nan, dtype=np.float32)
        if residues:
            for gidx, r_idx in enumerate(mapping):
                if r_idx != -1:
                    coords[gidx] = get_cb_like_coord(
                        residues[r_idx], include_altlocs=False
                    )
        # Distance matrix with NaN propagation
        diff = coords[:, None, :] - coords[None, :, :]
        D = np.sqrt(np.nansum(diff**2, axis=-1))
        # Where either coordinate missing -> set distance NaN explicitly
        missing_mask = np.isnan(coords).any(axis=1)
        # Broadcast to square
        D[np.repeat(missing_mask[:, None], global_length, axis=1)] = np.nan
        D[np.repeat(missing_mask[None, :], global_length, axis=0)] = np.nan
        matrices.append(D.astype(np.float32))
        names_in_order.append(name)
        # Save individual matrix
        np.save(os.path.join(output_dirs["matrices"], f"{name}_cbcb_dist.npy"), D)
        np.savetxt(
            os.path.join(output_dirs["matrices"], f"{name}_cbcb_dist.csv"),
            D,
            delimiter=",",
            fmt="%.3f",
        )
        vmax = np.nanpercentile(D, 99) if np.isfinite(D).any() else None
        cmap_ind = plt.cm.viridis.copy()
        cmap_ind.set_bad("black")
        plt.figure(figsize=(6, 5), dpi=140)
        plt.imshow(D, origin="lower", vmin=0.0, vmax=vmax, cmap=cmap_ind)
        plt.colorbar(label="Å")
        plt.title(f"CB–CB distances (Å) {name}")
        plt.xlabel("Aligned index")
        plt.ylabel("Aligned index")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs["heatmaps"], f"{name}_cbcb_heatmap.png"))
        plt.close()

        # Altloc variance expansion (optimized partial recompute)
        if args.include_altlocs and residues:
            altloc_replacements: Dict[str, Dict[int, np.ndarray]] = {}
            for gidx, r_idx in enumerate(mapping):
                if r_idx == -1:
                    continue
                res = residues[r_idx]
                if res.get_resname() == "GLY":
                    atom_name = "CA" if "CA" in res else None
                else:
                    atom_name = "CB" if "CB" in res else ("CA" if "CA" in res else None)
                if atom_name is None or atom_name not in res:
                    continue
                atom = res[atom_name]
                if not atom.is_disordered():
                    continue
                for altloc_id, altloc_atom in getattr(atom, "child_dict", {}).items():
                    altloc_replacements.setdefault(altloc_id, {})[gidx] = (
                        altloc_atom.coord
                    )
            for altloc_id, repl_map in altloc_replacements.items():
                if not repl_map:
                    continue
                D_var = D.copy()
                # Precompute valid mask for base coords
                base_valid = ~np.isnan(coords).any(axis=1)
                # Create array of updated coords just at changed indices
                for gidx, coord in repl_map.items():
                    # If original was missing, skip (cannot create distances)
                    if np.isnan(coords[gidx]).any():
                        continue
                    # Compute distances to all valid residues
                    diffs = coords[base_valid] - coord
                    d_valid = np.sqrt(np.sum(diffs**2, axis=1))
                    # Assign back
                    D_var[gidx, base_valid] = d_valid
                    D_var[base_valid, gidx] = d_valid
                    D_var[gidx, gidx] = 0.0
                matrices.append(D_var.astype(np.float32))
                alt_name = f"{name}_altloc{altloc_id}"
                np.save(
                    os.path.join(output_dirs["matrices"], f"{alt_name}_cbcb_dist.npy"),
                    D_var,
                )
                np.savetxt(
                    os.path.join(output_dirs["matrices"], f"{alt_name}_cbcb_dist.csv"),
                    D_var,
                    delimiter=",",
                    fmt="%.3f",
                )
                vmax_v = (
                    np.nanpercentile(D_var, 99) if np.isfinite(D_var).any() else None
                )
                plt.figure(figsize=(5, 4), dpi=130)
                plt.imshow(D_var, origin="lower", vmin=0.0, vmax=vmax_v, cmap=cmap_ind)
                plt.colorbar(label="Å")
                plt.title(f"Altloc {altloc_id} ({name})")
                plt.xlabel("Aligned index")
                plt.ylabel("Aligned index")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        output_dirs["heatmaps"], f"{alt_name}_cbcb_heatmap.png"
                    )
                )
                plt.close()

        # Index map for this structure's aligned positions (only residues present)
        with open(os.path.join(output_dirs["indices"], f"{name}_index.tsv"), "w") as fh:
            fh.write("global_idx\tres_idx\tresname\n")
            if residues:
                for gidx, r_idx in enumerate(mapping):
                    if r_idx != -1:
                        fh.write(f"{gidx}\t{r_idx}\t{residues[r_idx].get_resname()}\n")

    # Stack matrices (base + altloc variants if requested)
    stack = np.stack(matrices)  # (N, L, L)
    counts = np.sum(~np.isnan(stack), axis=0)
    variance_matrix = np.nanvar(stack, axis=0, ddof=1)
    variance_matrix[counts < args.min_samples] = np.nan

    variance_base = os.path.join(output_dirs["variance"], "variance_decomposition")
    np.save(f"{variance_base}.npy", variance_matrix)
    np.savetxt(
        f"{variance_base}.csv",
        variance_matrix,
        delimiter=",",
        fmt="%.6f",
    )
    np.save(os.path.join(output_dirs["variance"], "variance_counts.npy"), counts)
    np.savetxt(
        os.path.join(output_dirs["variance"], "variance_counts.csv"),
        counts,
        fmt="%d",
        delimiter=",",
    )

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("black")
    finite_vals = variance_matrix[np.isfinite(variance_matrix)]
    vmax = np.percentile(finite_vals, 95) if finite_vals.size else 1.0
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(variance_matrix, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Variance (Å²)")
    ax.set_xlabel("Aligned index")
    ax.set_ylabel("Aligned index")
    ax.set_title(
        f"{args.plot_name} Distance Variance Decomposition (min n={args.min_samples})"
    )
    plt.tight_layout()
    fig.savefig(f"{variance_base}_heatmap.png")
    plt.close(fig)

    # Coverage matrix similar to ColabFold style
    ref_global_chars = ["-"] * global_length
    # Fill reference chars for reference mapping (reconstruct from alignment)
    # Simplify: place ref sequence letters at their global indices
    for ref_pos, gidx in enumerate(ref_global_index):
        ref_global_chars[gidx] = ref_seq[ref_pos]

    # Compute per-structure identity (global) & row matrix
    coverage_rows = []
    identities = []
    for name, data in struct_data.items():
        residues = data["residues"]
        sequence = data["sequence"]
        aligned_ref, aligned_seq = alignments[name]
        # Compute identity across aligned (ref positions only)
        matches = sum(
            (ar == as_ and ar != "-") for ar, as_ in zip(aligned_ref, aligned_seq)
        )
        seq_id = matches / ref_len if ref_len else 0.0
        identities.append((name, seq_id))
    # Sort sequences by identity descending
    identities.sort(key=lambda x: x[1], reverse=True)

    for name, seq_id in identities:
        aligned_ref, aligned_seq = alignments[name]
        # Build global mapping again to know where residues present
        mapping = build_mapping(aligned_ref, aligned_seq)
        row = np.full(global_length, np.nan, dtype=np.float32)
        for gidx, r_idx in enumerate(mapping):
            if r_idx != -1:
                row[gidx] = seq_id
        coverage_rows.append(row)
    coverage_matrix = np.vstack(coverage_rows)

    coverage_counts = np.sum(~np.isnan(coverage_matrix), axis=0)
    plt.figure(figsize=(10, 6), dpi=130)
    cmap_cov = plt.cm.turbo.copy()
    cmap_cov.set_bad("white")
    plt.imshow(
        coverage_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_cov,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(label="Sequence identity to reference")
    plt.plot(np.arange(global_length), coverage_counts, color="black", linewidth=1)
    plt.title("Sequence coverage")
    plt.xlabel("Positions")
    plt.ylabel("Sequences")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs["variance"], "sequence_coverage.png"))
    plt.close()

    # Summary report
    summary_file = os.path.join(output_dirs["base"], "analysis_summary.txt")
    with open(summary_file, "w") as f:
        f.write("CB-CB Distance Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Input folder: {args.structure_folder}\n")
        f.write(f"Output folder: {args.out}\n")
        f.write(f"Structures parsed: {len(struct_data)} (failed {failed_count})\n")
        f.write(f"Reference: {ref_name} length {ref_len}\n")
        f.write(f"Global aligned length: {global_length}\n")
        f.write(f"Distance matrices computed: {len(matrices)}\n")
        if args.chains:
            f.write(f"Chain filter: {args.chains}\n")
        f.write(f"Min samples for variance: {args.min_samples}\n")
        if args.align_structures:
            f.write("(Note) Rigid-body alignment skipped for internal distances.\n")
        finite_vals = variance_matrix[np.isfinite(variance_matrix)]
        f.write("\nVariance Statistics (finite cells):\n")
        if finite_vals.size:
            f.write(f"Mean variance: {np.mean(finite_vals):.6f} Å²\n")
            f.write(f"Max variance: {np.max(finite_vals):.6f} Å²\n")
            f.write(f"Std variance: {np.std(finite_vals):.6f} Å²\n")
        else:
            f.write("No variance values (all cells masked).\n")

    print(f"\nAnalysis complete! Results saved to: {args.out}/")
    print(f"  - Distance matrices: {output_dirs['matrices']}")
    print(f"  - Heatmaps: {output_dirs['heatmaps']}")
    print(f"  - Index files: {output_dirs['indices']}")
    print(f"  - Variance analysis: {output_dirs['variance']}")
    print(f"  - Summary report: {summary_file}")


if __name__ == "__main__":
    main()
