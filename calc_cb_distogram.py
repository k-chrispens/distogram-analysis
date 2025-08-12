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
from Bio.PDB import PDBParser, MMCIFParser, is_aa
import matplotlib.pyplot as plt
import glob
import os

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
    if hasattr(target_atom, 'child_dict'):
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
        coords = np.array([get_cb_like_coord(r, False) for r in residues], dtype=np.float32)  # Use float32 to save memory
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
    if ext in ['.cif', '.mmcif']:
        return 'mmcif'
    elif ext == '.pdb':
        return 'pdb'
    else:
        # Try to guess from content or default to PDB
        return 'pdb'

def process_single_structure(structure_file: str, output_dirs: dict, chains: str = None, include_altlocs: bool = False) -> dict:
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
        if file_format == 'mmcif':
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
            np.save(os.path.join(output_dirs['matrices'], f"{file_base}{altloc_suffix}_cbcb_dist.npy"), D)
            np.savetxt(os.path.join(output_dirs['matrices'], f"{file_base}{altloc_suffix}_cbcb_dist.csv"), 
                      D, delimiter=",", fmt="%.3f")
            
            vmax = np.percentile(D, 99) if D.size else None
            plot_heatmap(D, f"CB–CB distances (Å) for {file_base}{altloc_suffix}", 
                       os.path.join(output_dirs['heatmaps'], f"{file_base}{altloc_suffix}_cbcb_heatmap.png"), 
                       vmin=0.0, vmax=vmax)
    else:
        D = D_result
        matrices = [D]
        np.save(os.path.join(output_dirs['matrices'], f"{file_base}_cbcb_dist.npy"), D)
        np.savetxt(os.path.join(output_dirs['matrices'], f"{file_base}_cbcb_dist.csv"), 
                  D, delimiter=",", fmt="%.3f")
        
        vmax = np.percentile(D, 99) if D.size else None
        plot_heatmap(D, f"CB–CB distances (Å) for {file_base}", 
                   os.path.join(output_dirs['heatmaps'], f"{file_base}_cbcb_heatmap.png"), 
                   vmin=0.0, vmax=vmax)

    # Index map
    with open(os.path.join(output_dirs['indices'], f"{file_base}_index.tsv"), "w") as fh:
        fh.write("idx\tchain\tresi\ticode\tresname\n")
        for i, r in enumerate(residues):
            chain = r.get_parent().id
            (_, resi, icode) = r.get_id()
            fh.write(f"{i}\t{chain}\t{resi}\t{icode if icode.strip() else ' '}\t{r.get_resname()}\n")

    return {
        "matrices": matrices,
        "file_base": file_base,
        "file_format": file_format,
        "n_residues": len(residues)
    }

def setup_output_directories(base_dir: str) -> dict:
    """Create organized output directory structure."""
    dirs = {
        'base': base_dir,
        'matrices': os.path.join(base_dir, 'distance_matrices'),
        'heatmaps': os.path.join(base_dir, 'heatmaps'),
        'indices': os.path.join(base_dir, 'index_files'),
        'variance': os.path.join(base_dir, 'variance_analysis')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def main():
    ap = argparse.ArgumentParser(description="Compute CB-CB distance maps from PDB/mmCIF structures")
    ap.add_argument("--structure-folder", required=True, help="folder containing PDB/mmCIF files")
    ap.add_argument("--out", required=True, help="output directory name")
    ap.add_argument("--chains", help="comma-separated chain IDs (optional)")
    ap.add_argument("--include-altlocs", action="store_true", help="include altloc variance in calculations")
    ap.add_argument("--plot-name", default="CB-CB", help="name for plots (default: CB-CB)")
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
    
    # Process structures (optimized with better memory management)
    print(f"Processing {len(structure_files)} structures...")
    
    all_matrices = []
    processed_count = 0
    failed_count = 0

    for structure_file in structure_files:
        # Detect format and parse accordingly
        file_format = detect_file_format(structure_file)
        file_base = os.path.splitext(os.path.basename(structure_file))[0]
        
        try:
            # Initialize fresh parsers to avoid memory leaks
            if file_format == 'mmcif':
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)
            struct = parser.get_structure("s", structure_file)
        except Exception as e:
            print(f"Error parsing {structure_file}: {e}. Skipping.")
            failed_count += 1
            continue
            
        want = set(args.chains.split(",")) if args.chains else None
        residues = []
        for model in struct:
            for chain in model:
                if want and chain.id not in want:
                    continue
                for res in chain:
                    if is_aa(res, standard=True) and "CA" in res:
                        residues.append(res)
        if not residues:
            print(f"No protein residues found in {structure_file} (check chains/input). Skipping.")
            failed_count += 1
            continue

        # Compute distances
        D_result = compute_cbcb(residues, args.include_altlocs)
        
        if args.include_altlocs and isinstance(D_result, list):
            # Multiple matrices due to altlocs
            all_matrices.extend(D_result)
            for i, D in enumerate(D_result):
                altloc_suffix = f"_altloc{i}" if len(D_result) > 1 else ""
                # Save matrices in organized folders
                np.save(os.path.join(output_dirs['matrices'], f"{file_base}{altloc_suffix}_cbcb_dist.npy"), D)
                np.savetxt(os.path.join(output_dirs['matrices'], f"{file_base}{altloc_suffix}_cbcb_dist.csv"), 
                          D, delimiter=",", fmt="%.3f")
                
                vmax = np.percentile(D, 99) if D.size else None
                plot_heatmap(D, f"CB–CB distances (Å) for {file_base}{altloc_suffix}", 
                           os.path.join(output_dirs['heatmaps'], f"{file_base}{altloc_suffix}_cbcb_heatmap.png"), 
                           vmin=0.0, vmax=vmax)
        else:
            # Single matrix
            D = D_result
            all_matrices.append(D)
            np.save(os.path.join(output_dirs['matrices'], f"{file_base}_cbcb_dist.npy"), D)
            np.savetxt(os.path.join(output_dirs['matrices'], f"{file_base}_cbcb_dist.csv"), 
                      D, delimiter=",", fmt="%.3f")
            
            vmax = np.percentile(D, 99) if D.size else None
            plot_heatmap(D, f"CB–CB distances (Å) for {file_base}", 
                       os.path.join(output_dirs['heatmaps'], f"{file_base}_cbcb_heatmap.png"), 
                       vmin=0.0, vmax=vmax)

        # Index map (same for all cases)
        with open(os.path.join(output_dirs['indices'], f"{file_base}_index.tsv"), "w") as fh:
            fh.write("idx\tchain\tresi\ticode\tresname\n")
            for i, r in enumerate(residues):
                chain = r.get_parent().id
                (_, resi, icode) = r.get_id()
                fh.write(f"{i}\t{chain}\t{resi}\t{icode if icode.strip() else ' '}\t{r.get_resname()}\n")

        processed_count += 1
        print(f"Processed {processed_count}/{len(structure_files)}: {file_base} (format: {file_format})")
        
        # Clean up to free memory
        del struct, residues

    if not all_matrices:
        raise SystemExit("No valid matrices were computed from the structure files.")
    
    print("\nAnalysis Summary:")
    print(f"  - Successfully processed: {processed_count}/{len(structure_files)} structures")
    if failed_count > 0:
        print(f"  - Failed: {failed_count} structures")
    print(f"  - Total distance matrices: {len(all_matrices)}")
    if args.include_altlocs:
        print("  - Altloc variance: Included")
        print(f"  - Average matrices per structure: {len(all_matrices)/processed_count:.1f}")

    # Calculate Variance Decomposition (optimized)
    print("Computing variance decomposition...")
    
    # Check if all matrices have the same shape
    shapes = [m.shape for m in all_matrices]
    if len(set(shapes)) == 1:
        # All matrices have same shape - use vectorized approach
        matrices_array = np.array(all_matrices)  # Shape: (n_matrices, n_residues, n_residues)
        variance_matrix = np.var(matrices_array, axis=0, ddof=1)  # Sample variance
    else:
        # Different shapes - find common dimensions and compute variance element-wise
        max_shape = tuple(max(dim) for dim in zip(*shapes))
        print(f"  Note: Matrices have different shapes (max: {max_shape}), using element-wise calculation")
        
        variance_matrix = np.zeros(max_shape)
        
        for i in range(max_shape[0]):
            for j in range(max_shape[1]):
                # Collect values at position (i,j) from matrices that have this position
                values = []
                for matrix in all_matrices:
                    if i < matrix.shape[0] and j < matrix.shape[1]:
                        values.append(matrix[i, j])
                
                if len(values) > 1:
                    variance_matrix[i, j] = np.var(values, ddof=1)
                else:
                    variance_matrix[i, j] = 0.0  # No variance with <2 values

    # Save variance matrix and create organized outputs
    variance_base = os.path.join(output_dirs['variance'], 'variance_decomposition')
    
    # Save variance matrix as NumPy and CSV
    np.save(f"{variance_base}.npy", variance_matrix)
    np.savetxt(f"{variance_base}.csv", variance_matrix, delimiter=",", fmt="%.6f")
    
    # Plot the variance matrix as a heatmap
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    cax = ax.imshow(variance_matrix, cmap='coolwarm', vmin=0, vmax=np.percentile(variance_matrix, 95))
    fig.colorbar(cax, ax=ax, label='Variance (Å²)')
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')
    ax.set_title(f'{args.plot_name} Distance Variance Decomposition')
    plt.tight_layout()
    fig.savefig(f"{variance_base}_heatmap.png")
    plt.close(fig)
    
    # Create summary report
    summary_file = os.path.join(output_dirs['base'], 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("CB-CB Distance Analysis Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Input folder: {args.structure_folder}\n")
        f.write(f"Output folder: {args.out}\n")
        f.write(f"Structures processed: {len(structure_files)}\n")
        f.write(f"Distance matrices computed: {len(all_matrices)}\n")
        if args.chains:
            f.write(f"Chain filter: {args.chains}\n")
        if args.include_altlocs:
            f.write("Altloc variance: Included\n")
        f.write("\nVariance Statistics:\n")
        f.write(f"Mean variance: {np.mean(variance_matrix):.6f} Å\n")
        f.write(f"Max variance: {np.max(variance_matrix):.6f} Å\n")
        f.write(f"Std variance: {np.std(variance_matrix):.6f} Å\n")
    
    print(f"\nAnalysis complete! Results saved to: {args.out}/")
    print(f"  - Distance matrices: {output_dirs['matrices']}")
    print(f"  - Heatmaps: {output_dirs['heatmaps']}")
    print(f"  - Index files: {output_dirs['indices']}")
    print(f"  - Variance analysis: {output_dirs['variance']}")
    print(f"  - Summary report: {summary_file}")


if __name__ == "__main__":
    main()
