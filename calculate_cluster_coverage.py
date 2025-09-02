import argparse
import os
import re
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests


class ClusterHistogramPlotter:
    def __init__(self, seq_identity: float = 0.4):
        self.seq_identity = (
            int(seq_identity) if seq_identity > 1 else int(seq_identity * 100)
        )
        self.clusters_url = (
            f"https://cdn.rcsb.org/resources/sequence/clusters/"
            f"clusters-by-entity-{self.seq_identity}.txt"
        )

    def fetch_clusters(self) -> List[Set[str]]:
        """Fetch and parse the RCSB entity sequence clusters file.

        Returns a list of clusters, each as a set of PDB IDs (uppercase).
        """
        url = self.clusters_url

        def _parse_clusters(lines: List[str]) -> List[Set[str]]:
            lines = [ln.strip() for ln in lines if ln.strip()]
            if not lines:
                raise RuntimeError("Clusters file contains no non-empty lines.")

            token_pattern = re.compile(r"^[A-Za-z0-9]+_[0-9]+$")
            sampled = set(lines[:200])
            bad_tokens = 0
            total_tokens = 0
            clusters_local: List[Set[str]] = []

            for ln in lines:
                toks = ln.split()
                if not toks:
                    continue
                if ln in sampled:
                    for t in toks:
                        total_tokens += 1
                        if not token_pattern.match(t):
                            bad_tokens += 1
                pdb_ids = [t.split("_")[0].upper() for t in toks if t]
                s = set(pdb_ids)
                s.discard("AF")
                clusters_local.append(s)

            if total_tokens > 0 and (bad_tokens / total_tokens) > 0.5:
                raise RuntimeError(
                    "Clusters file format looks unexpected (too many invalid entity tokens)."
                )
            if not clusters_local:
                raise RuntimeError("No clusters parsed from file.")
            return clusters_local

        # Local fallback
        identity_suffix = url.rsplit("-", 1)[-1]
        local_fname = (
            url.rsplit("/", 1)[-1]
            if identity_suffix.endswith(".txt")
            and identity_suffix.startswith(str(self.seq_identity))
            else None
        )
        local_path = (
            os.path.join(os.path.dirname(__file__), local_fname)
            if local_fname
            else None
        )
        if local_path and os.path.exists(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as fh:
                    return _parse_clusters(fh.readlines())
            except Exception as parse_err:
                print(
                    f"Local clusters file '{local_path}' could not be parsed: {parse_err}. Will attempt download."
                )

        try:
            response = requests.get(url, timeout=20)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch clusters file: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch clusters file from {url}: status {response.status_code}"
            )

        return _parse_clusters(response.text.splitlines())

    def read_pdb_list(self, filename: str) -> Set[str]:
        pdb_ids = set()
        with open(filename, "r") as f:
            for line in f:
                pdb_id = line.strip().upper()
                if pdb_id and len(pdb_id) == 4:
                    pdb_ids.add(pdb_id)
                elif pdb_id:
                    print(f"Warning: Skipping invalid PDB ID: {pdb_id}")
        return pdb_ids

    def create_histogram(
        self,
        pdb_list_file: str,
        output_file: str | None = None,
        max_clusters: int | None = None,
        log_scale: bool = False,
        show_low_overlap: bool = False,
        low_overlap_threshold: int = 1,
    ) -> dict:
        print(f"Reading PDB IDs from {pdb_list_file}...")
        input_pdbs = self.read_pdb_list(pdb_list_file)
        print(f"Found {len(input_pdbs)} valid PDB IDs in input file")

        print(f"\nFetching sequence clusters at {self.seq_identity}% identity...")
        clusters = self.fetch_clusters()
        print(f"Loaded {len(clusters)} total clusters")

        # Build (cluster_size, overlap_count) for ALL clusters
        all_cluster_records: List[Tuple[int, int]] = []
        for cluster in clusters:
            overlap = len(cluster & input_pdbs)
            all_cluster_records.append((len(cluster), overlap))

        # Sort all clusters by size desc
        clusters_by_size: List[Tuple[int, int]] = sorted(
            all_cluster_records, key=lambda t: t[0], reverse=True
        )

        # Derive plotting data only from clusters with overlap
        cluster_data_plot: List[Tuple[int, int]] = [
            (size, overlap) for size, overlap in clusters_by_size if overlap > 0
        ]

        if not cluster_data_plot:
            raise RuntimeError(
                "No clusters found that intersect with the provided PDB list."
            )

        total_clusters_all = len(clusters)
        clusters_with_overlap_all = sum(1 for _, ov in clusters_by_size if ov > 0)

        if max_clusters:
            cluster_data_plot = cluster_data_plot[:max_clusters]
            x_label_suffix = f" (showing top {max_clusters})"
        else:
            x_label_suffix = ""

        cluster_sizes = [t[0] for t in cluster_data_plot]
        input_pdb_counts = [t[1] for t in cluster_data_plot]
        print(
            f"Filtered to {len(cluster_sizes)} clusters that intersect with input PDBs"
        )

        # Single shared y-axis with grouped bars
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(1, len(cluster_sizes) + 1)
        width = 0.4
        ax.bar(
            x - width / 2,
            cluster_sizes,
            width=width,
            color="steelblue",
            alpha=0.7,
            label="Total cluster size",
        )
        ax.bar(
            x + width / 2,
            input_pdb_counts,
            width=width,
            color="coral",
            alpha=0.8,
            label=f"PDBs from input file (n={len(input_pdbs)})",
        )

        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel(f"Cluster rank (ordered by size){x_label_suffix}", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        if len(x) > 50:
            step = max(1, len(x) // 20)
            ax.set_xticks(x[::step])

        ax.set_title(
            f"RCSB Cluster Size Distribution at {self.seq_identity}% Sequence Identity\n"
            f"with Coverage from {os.path.basename(pdb_list_file)} ({len(input_pdbs)} PDBs)",
            fontsize=14,
            pad=20,
        )
        ax.legend(loc="upper right", framealpha=0.9)

        coverage_stats = {
            "clusters_with_pdbs": clusters_with_overlap_all,
            "total_clusters": total_clusters_all,
            "clusters_shown": len(cluster_sizes),
            "total_pdbs_mapped": sum(input_pdb_counts),
            "largest_coverage": max(input_pdb_counts) if input_pdb_counts else 0,
            "largest_coverage_cluster": input_pdb_counts.index(max(input_pdb_counts))
            + 1
            if input_pdb_counts and max(input_pdb_counts) > 0
            else None,
        }

        overlap_pct = (
            100.0
            * coverage_stats["clusters_with_pdbs"]
            / coverage_stats["total_clusters"]
            if coverage_stats["total_clusters"] > 0
            else 0.0
        )
        stats_text = (
            f"Coverage Statistics:\n"
            f"Overlap clusters: {coverage_stats['clusters_with_pdbs']}/{coverage_stats['total_clusters']} ({overlap_pct:.1f}%)\n"
            f"PDBs mapped: {coverage_stats['total_pdbs_mapped']}/{len(input_pdbs)}\n"
            f"Max coverage: {coverage_stats['largest_coverage']} PDBs "
            f"(cluster {coverage_stats['largest_coverage_cluster']})"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"\nPlot saved to {output_file}")
        else:
            plt.show()

        # Optionally list clusters with no/low overlap
        if show_low_overlap:
            if low_overlap_threshold < 0:
                print(
                    f"Warning: low-overlap-threshold {low_overlap_threshold} is invalid; using 0"
                )
                low_overlap_threshold = 0  # type: ignore[assignment]

            considered = (
                clusters_by_size[:max_clusters] if max_clusters else clusters_by_size
            )
            low_overlap_list = [
                (rank + 1, size, overlap)
                for rank, (size, overlap) in enumerate(considered)
                if overlap <= low_overlap_threshold
            ]

            domain_note = (
                f"top {max_clusters} clusters by size"
                if max_clusters
                else "all clusters"
            )
            print(
                f"\nLow-overlap clusters (overlap <= {low_overlap_threshold}) in {domain_note}: {len(low_overlap_list)}"
            )
            if low_overlap_list:
                print("Rank\tClusterSize\tOverlap")
                for r, s, o in low_overlap_list:
                    print(f"{r}\t{s}\t{o}")

        return coverage_stats


def main():
    parser = argparse.ArgumentParser(
        description="Create histogram of RCSB cluster sizes with PDB coverage overlay",
    )
    parser.add_argument("pdb_list", help="Text file containing PDB IDs (one per line)")
    parser.add_argument(
        "--seq-identity",
        type=float,
        help="Sequence identity threshold for clustering (default: 0.4)",
        default=0.4,
    )
    parser.add_argument(
        "--output",
        help="Output file for the plot (e.g., cluster_histogram.png)",
        default=None,
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        help="Maximum number of clusters to display (starting from largest)",
        default=None,
    )
    parser.add_argument(
        "--log-scale",
        "--log",
        action="store_true",
        help="Use logarithmic scale for Y axis",
    )
    parser.add_argument(
        "--show-low-overlap",
        action="store_true",
        help=(
            "Print clusters with no or low overlap (<= threshold) with the input PDB list; "
            "respects --max-clusters domain if provided"
        ),
    )
    parser.add_argument(
        "--low-overlap-threshold",
        type=int,
        default=1,
        help=(
            "Threshold for 'very little overlap' (default: 1). "
            "Clusters with overlap <= this value will be listed."
        ),
    )

    args = parser.parse_args()

    print("=== Creating Cluster Size Histogram ===")
    print(f"Input file: {args.pdb_list}")
    print(
        f"Sequence identity: {args.seq_identity * 100 if args.seq_identity < 1 else args.seq_identity}%"
    )
    if args.max_clusters:
        print(f"Showing top {args.max_clusters} clusters")
    if args.log_scale:
        print("Using logarithmic scale")
    print()

    plotter = ClusterHistogramPlotter(seq_identity=args.seq_identity)

    try:
        stats = plotter.create_histogram(
            args.pdb_list,
            output_file=args.output,
            max_clusters=args.max_clusters,
            log_scale=args.log_scale,
            show_low_overlap=args.show_low_overlap,
            low_overlap_threshold=args.low_overlap_threshold,
        )

        print("\n=== Summary ===")
        print(f"Clusters containing input PDBs: {stats['clusters_with_pdbs']}")
        print(f"Total PDBs successfully mapped: {stats['total_pdbs_mapped']}")
        if stats["largest_coverage_cluster"]:
            print(
                f"Cluster with most input PDBs: Cluster {stats['largest_coverage_cluster']} ({stats['largest_coverage']} PDBs)"
            )

    except Exception as e:
        print(f"Plotting failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
