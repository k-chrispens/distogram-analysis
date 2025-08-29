import requests
import argparse
import re
from typing import List, Set, Dict
from collections import defaultdict
import os


class ClusterSpanCounter:
    def __init__(self, seq_identity: float = 0.4):
        self.seq_identity = (
            int(seq_identity) if seq_identity > 1 else int(seq_identity * 100)
        )
        self.clusters_url = f"https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{self.seq_identity}.txt"

    def fetch_clusters(self) -> List[Set[str]]:
        """Fetch and validate the RCSB entity sequence clusters file.

        Returns a list of clusters, each cluster being a set of entity IDs like '1ABC_1'.

        Raises:
            RuntimeError: If the file cannot be retrieved or is malformed/empty.
        """
        url = self.clusters_url

        # Helper: parse content into clusters with basic validation
        def _parse_clusters(lines: List[str]) -> List[Set[str]]:
            lines = [ln.strip() for ln in lines if ln.strip()]
            if not lines:
                raise RuntimeError("Clusters file contains no non-empty lines.")

            # Validate token format on a sample of lines
            token_pattern = re.compile(r"^[A-Za-z0-9]+_[0-9]+$")
            sampled = lines[:200]
            bad_tokens = 0
            total_tokens = 0
            clusters_local: List[Set[str]] = []
            for ln in lines:
                toks = ln.split()
                if not toks:
                    continue
                # Only count validation for the sampled lines to avoid O(file)
                if ln in sampled:
                    for t in toks:
                        total_tokens += 1
                        if not token_pattern.match(t):
                            bad_tokens += 1

                toks = [t.split("_")[0] for t in toks if t]  # Keep only PDB IDs
                clusters_local.append(set(toks))

            if total_tokens > 0 and (bad_tokens / total_tokens) > 0.5:
                raise RuntimeError(
                    "Clusters file format looks unexpected (too many invalid entity tokens)."
                )

            if len(clusters_local) == 0:
                raise RuntimeError("No clusters parsed from file.")

            return clusters_local

        # Try local file first if it exists (same identity percent) next to this script
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
            response = None
            http_error = e
        else:
            http_error = None

        if response is None or response.status_code != 200:
            # Provide an informative error and attempt a safe local fallback if present
            status_note = (
                f"status {response.status_code}"
                if response is not None
                else str(http_error)
            )

            raise RuntimeError(
                f"Failed to fetch clusters file from {url}: {status_note}. "
                "If you're offline, place the file next to this script and retry."
            )

        # Basic content-type sanity (don't hard-fail if missing but warn via exception on binary)
        ctype = (response.headers or {}).get("Content-Type", "").lower()
        if (
            "text" not in ctype
            and "plain" not in ctype
            and "csv" not in ctype
            and ctype
        ):
            # It's unusual to get non-text here; validate by parsing which will error if garbage
            pass

        try:
            return _parse_clusters(response.text.splitlines())
        except Exception as parse_err:
            raise RuntimeError(
                f"Downloaded clusters file from {url} but failed to parse: {parse_err}"
            ) from parse_err

    def build_pdb_to_cluster_map(self, clusters: List[Set[str]]) -> Dict[str, int]:
        pdb_to_cluster = {}
        for cluster_idx, cluster in enumerate(clusters, 1):
            for pdb_id in cluster:
                pdb_to_cluster[pdb_id.upper()] = cluster_idx
        return pdb_to_cluster

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

    def count_clusters(self, pdb_list_file: str) -> Dict:
        print(f"Reading PDB IDs from {pdb_list_file}...")
        input_pdbs = self.read_pdb_list(pdb_list_file)
        print(f"Found {len(input_pdbs)} valid PDB IDs in input file")

        if not input_pdbs:
            return {
                "total_clusters_spanned": 0,
                "input_pdb_count": 0,
                "pdbs_found_in_clustering": 0,
                "pdbs_not_found": [],
                "cluster_distribution": {},
            }

        print(f"\nFetching sequence clusters at {self.seq_identity}% identity...")
        clusters = self.fetch_clusters()
        print(f"Loaded {len(clusters)} total clusters")

        print("\nBuilding PDB to cluster mapping...")
        pdb_to_cluster = self.build_pdb_to_cluster_map(clusters)

        clusters_spanned = set()
        pdbs_not_found = []
        cluster_distribution = defaultdict(list)

        for pdb_id in input_pdbs:
            if pdb_id in pdb_to_cluster:
                cluster_idx = pdb_to_cluster[pdb_id]
                clusters_spanned.add(cluster_idx)
                cluster_distribution[cluster_idx].append(pdb_id)
            else:
                pdbs_not_found.append(pdb_id)

        results = {
            "total_clusters_spanned": len(clusters_spanned),
            "input_pdb_count": len(input_pdbs),
            "pdbs_found_in_clustering": len(input_pdbs) - len(pdbs_not_found),
            "pdbs_not_found": sorted(pdbs_not_found),
            "cluster_distribution": {
                k: sorted(v) for k, v in sorted(cluster_distribution.items())
            },
        }

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Count how many RCSB sequence clusters are spanned by a list of PDB IDs"
    )
    parser.add_argument("pdb_list", help="Text file containing PDB IDs (one per line)")
    parser.add_argument(
        "--seq-identity",
        type=float,
        help="Sequence identity threshold for clustering (default: 0.4)",
        default=0.4,
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed cluster distribution"
    )

    args = parser.parse_args()

    print("=== PDB Cluster Span Analysis ===")
    print(f"Input file: {args.pdb_list}")
    print(
        f"Sequence identity: {args.seq_identity * 100 if args.seq_identity < 1 else args.seq_identity}%"
    )
    print()

    counter = ClusterSpanCounter(seq_identity=args.seq_identity)

    try:
        results = counter.count_clusters(args.pdb_list)

        print("\n=== Results ===")
        print(f"Total PDB IDs in input: {results['input_pdb_count']}")
        print(f"PDB IDs found in clustering: {results['pdbs_found_in_clustering']}")
        print(f"PDB IDs not found: {len(results['pdbs_not_found'])}")
        print(f"\nNumber of clusters spanned: {results['total_clusters_spanned']}")

        if results["pdbs_not_found"]:
            print(
                f"\nPDB IDs not found in clustering: {', '.join(results['pdbs_not_found'][:10])}"
            )
            if len(results["pdbs_not_found"]) > 10:
                print(f"  ... and {len(results['pdbs_not_found']) - 10} more")

        if args.verbose and results["cluster_distribution"]:
            print("\n=== Cluster Distribution ===")
            for cluster_idx, pdbs in results["cluster_distribution"].items():
                print(f"Cluster {cluster_idx}: {len(pdbs)} PDBs")
                print(f"  Members: {', '.join(pdbs[:5])}")
                if len(pdbs) > 5:
                    print(f"  ... and {len(pdbs) - 5} more")

        cluster_sizes = [len(pdbs) for pdbs in results["cluster_distribution"].values()]
        if cluster_sizes:
            print("\nCluster size statistics:")
            print(
                f"  Mean PDBs per cluster: {sum(cluster_sizes) / len(cluster_sizes):.1f}"
            )
            print(f"  Max PDBs in one cluster: {max(cluster_sizes)}")
            print(f"  Min PDBs in one cluster: {min(cluster_sizes)}")

    except Exception as e:
        print(f"Analysis failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
