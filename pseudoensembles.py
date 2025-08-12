import os
import numpy as np
from Bio import PDB
from Bio.PDB import Superimposer
from Bio.PDB.mmcifio import MMCIFIO  # Added for mmCIF output
from datetime import datetime
import requests
from typing import List, Dict, Tuple
import json
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import urllib.request
import gzip


class CustomPseudoensemble:
    def __init__(
        self,
        center_pdb: str,
        sequence_identity: float = 40,
        cutoff_date: str = None,
        max_structures: int = 50,
    ):
        self.center_pdb = center_pdb.upper()
        self.sequence_identity = sequence_identity
        self.cutoff_date = (
            datetime.strptime(cutoff_date, "%Y-%m-%d") if cutoff_date else None
        )
        self.max_structures = max_structures
        self.parser = PDB.PDBParser(QUIET=True)
        self.superimposer = Superimposer()

    def search_similar_structures(self) -> List[str]:
        print(f"Searching for structures similar to {self.center_pdb}...")

        center_sequence = self.get_sequence(self.center_pdb)

        # Use REST API directly for sequence similarity search
        url = "https://search.rcsb.org/rcsbsearch/v2/query"

        query_json = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "evalue_cutoff": 10,
                    "identity_cutoff": self.sequence_identity / 100.0,
                    "target": "pdb_protein_sequence",
                    "value": center_sequence,
                },
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": self.max_structures * 2,  # Get extra in case some fail
                }
            },
        }

        if self.cutoff_date:
            # Add date filter using compound query
            date_query = {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_accession_info.deposit_date",
                    "operator": "less_or_equal",
                    "value": self.cutoff_date.strftime("%Y-%m-%d"),
                },
            }

            # Combine sequence and date queries
            query_json = {
                "query": {
                    "type": "group",
                    "logical_operator": "and",
                    "nodes": [query_json["query"], date_query],
                },
                "return_type": "entry",
                "request_options": query_json["request_options"],
            }

        response = requests.post(url, json=query_json)

        if response.status_code != 200:
            print(f"Search API error: {response.status_code}")
            print(response.text)
            return [self.center_pdb]

        result_data = response.json()

        # Extract PDB IDs from results
        results = []
        if "result_set" in result_data:
            for item in result_data["result_set"]:
                results.append(item["identifier"])

        # Ensure center PDB is first
        if self.center_pdb not in results:
            results.insert(0, self.center_pdb)
        else:
            results.remove(self.center_pdb)
            results.insert(0, self.center_pdb)

        results = results[: self.max_structures]
        print(f"Found {len(results)} similar structures")
        return results

    def get_sequence(self, pdb_id: str) -> str:
        """Get the protein sequence from RCSB PDB"""
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/display"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to get sequence for {pdb_id}")
            return ""

        lines = response.text.strip().split("\n")
        # Skip header lines (start with >)
        sequence_lines = [line for line in lines if not line.startswith(">")]
        sequence = "".join(sequence_lines)

        # If multiple chains, just use the first one
        if len(sequence) == 0:
            print(f"No sequence found for {pdb_id}")
            return ""

        return sequence

    def download_structure(self, pdb_id: str, base_folder: str = "structures") -> str:
        """Download PDB structure file"""
        os.makedirs(base_folder, exist_ok=True)
        filename = f"{base_folder}/{pdb_id}.pdb"
        if not os.path.exists(filename):
            gz_filename = f"{pdb_id}.pdb.gz"
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"

            try:
                urllib.request.urlretrieve(url, gz_filename)

                with gzip.open(gz_filename, "rt") as f_in:
                    with open(filename, "w") as f_out:
                        f_out.write(f_in.read())
                os.remove(gz_filename)
            except Exception as e:
                print(f"Error downloading {pdb_id}: {e}")
                raise

        return filename

    def extract_ca_atoms(self, structure) -> List:
        """Extract C-alpha atoms from structure"""
        ca_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        ca_atoms.append(residue["CA"])
                # Only use first chain
                break
            # Only use first model
            break
        return ca_atoms

    def align_to_center(
        self, center_structure, mobile_structure
    ) -> Tuple[float, np.ndarray]:
        """Align mobile structure to center structure"""
        center_atoms = self.extract_ca_atoms(center_structure)
        mobile_atoms = self.extract_ca_atoms(mobile_structure)

        if len(center_atoms) == 0 or len(mobile_atoms) == 0:
            return float("inf"), np.array([])

        min_len = min(len(center_atoms), len(mobile_atoms))
        center_atoms = center_atoms[:min_len]
        mobile_atoms = mobile_atoms[:min_len]

        try:
            self.superimposer.set_atoms(center_atoms, mobile_atoms)
            self.superimposer.apply(mobile_structure.get_atoms())

            rmsd = self.superimposer.rms
            aligned_coords = np.array([atom.get_coord() for atom in mobile_atoms])
            return rmsd, aligned_coords
        except Exception as e:
            print(f"Alignment error: {e}")
            return float("inf"), np.array([])

    def compute_structural_metrics(self, structures: List, center_idx: int = 0) -> Dict:
        """Compute various structural heterogeneity metrics"""
        n_structures = len(structures)
        rmsd_to_center = np.zeros(n_structures)
        pairwise_rmsd = np.zeros((n_structures, n_structures))

        center_structure = structures[center_idx]
        center_atoms = self.extract_ca_atoms(center_structure)
        n_residues = len(center_atoms)

        if n_residues == 0:
            return {"error": "No CA atoms found in center structure", "n_structures": 0}

        aligned_coords_list = []
        aligned_coords_list.append(
            np.array([atom.get_coord() for atom in center_atoms])
        )

        # Align all structures to center
        for i, structure in enumerate(structures):
            if i == center_idx:
                continue

            rmsd, aligned_coords = self.align_to_center(center_structure, structure)
            if rmsd != float("inf") and len(aligned_coords) > 0:
                rmsd_to_center[i] = rmsd
                pairwise_rmsd[center_idx, i] = rmsd
                pairwise_rmsd[i, center_idx] = rmsd
                aligned_coords_list.append(aligned_coords)

        # Compute pairwise RMSD matrix
        print("Computing pairwise RMSD matrix...")
        for i in range(n_structures):
            for j in range(i + 1, n_structures):
                if i != center_idx and j != center_idx:
                    mobile_atoms = self.extract_ca_atoms(structures[j])
                    fixed_atoms = self.extract_ca_atoms(structures[i])

                    if len(mobile_atoms) > 0 and len(fixed_atoms) > 0:
                        min_len = min(len(mobile_atoms), len(fixed_atoms))

                        try:
                            self.superimposer.set_atoms(
                                fixed_atoms[:min_len], mobile_atoms[:min_len]
                            )
                            pairwise_rmsd[i, j] = self.superimposer.rms
                            pairwise_rmsd[j, i] = pairwise_rmsd[i, j]
                        except Exception:
                            pairwise_rmsd[i, j] = float("inf")
                            pairwise_rmsd[j, i] = float("inf")

        # PCA analysis
        pca_variance = [1.0]
        if len(aligned_coords_list) > 3:
            try:
                # Stack coordinates for PCA
                coords_for_pca = []
                for coords in aligned_coords_list[
                    :10
                ]:  # Limit to 10 structures for PCA
                    if len(coords) >= n_residues:
                        coords_for_pca.append(coords[:n_residues].flatten())

                if len(coords_for_pca) > 1:
                    coords_matrix = np.array(coords_for_pca)
                    pca = PCA(n_components=min(3, coords_matrix.shape[0] - 1))
                    pca.fit(coords_matrix)
                    pca_variance = pca.explained_variance_ratio_.tolist()
            except Exception as e:
                print(f"PCA error: {e}")

        # Calculate RMSF
        rmsf = np.zeros(n_residues)
        for res_idx in range(n_residues):
            positions = []
            for coords in aligned_coords_list:
                if res_idx < len(coords):
                    positions.append(coords[res_idx])

            if len(positions) > 1:
                positions = np.array(positions)
                mean_pos = np.mean(positions, axis=0)
                rmsf[res_idx] = np.sqrt(
                    np.mean(np.sum((positions - mean_pos) ** 2, axis=1))
                )

        # Filter out infinite values for statistics
        valid_rmsd = rmsd_to_center[rmsd_to_center > 0]
        valid_pairwise = pairwise_rmsd[
            (pairwise_rmsd > 0) & (pairwise_rmsd != float("inf"))
        ]

        return {
            "n_structures": n_structures,
            "rmsd_to_center": rmsd_to_center.tolist(),
            "mean_rmsd_to_center": np.mean(valid_rmsd) if len(valid_rmsd) > 0 else 0,
            "max_rmsd_to_center": np.max(valid_rmsd) if len(valid_rmsd) > 0 else 0,
            "pairwise_rmsd_matrix": pairwise_rmsd.tolist(),
            "mean_pairwise_rmsd": np.mean(valid_pairwise)
            if len(valid_pairwise) > 0
            else 0,
            "rmsf": rmsf.tolist(),
            "mean_rmsf": np.mean(rmsf),
            "pca_variance_explained": pca_variance,
        }

    def identify_conformational_states(
        self, structures: List, eps: float = 2.0
    ) -> Dict:
        """Identify conformational clusters using DBSCAN"""
        n_structures = len(structures)

        if n_structures < 2:
            return {
                "n_clusters": 1,
                "labels": [0],
                "cluster_centers": [0],
                "noise_points": 0,
            }

        # Compute pairwise RMSD matrix
        pairwise_rmsd = np.zeros((n_structures, n_structures))

        for i in range(n_structures):
            for j in range(i + 1, n_structures):
                mobile_atoms = self.extract_ca_atoms(structures[j])
                fixed_atoms = self.extract_ca_atoms(structures[i])

                if len(mobile_atoms) > 0 and len(fixed_atoms) > 0:
                    min_len = min(len(mobile_atoms), len(fixed_atoms))

                    try:
                        self.superimposer.set_atoms(
                            fixed_atoms[:min_len], mobile_atoms[:min_len]
                        )
                        pairwise_rmsd[i, j] = self.superimposer.rms
                        pairwise_rmsd[j, i] = pairwise_rmsd[i, j]
                    except Exception:
                        pairwise_rmsd[i, j] = (
                            999.0  # Large distance for failed alignments
                        )
                        pairwise_rmsd[j, i] = 999.0

        # Replace inf values with large distance
        pairwise_rmsd[pairwise_rmsd == float("inf")] = 999.0

        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit(
            pairwise_rmsd
        )

        unique_labels = set(clustering.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        # Find cluster centers
        cluster_centers = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster_members = np.where(clustering.labels_ == label)[0]
            if len(cluster_members) > 0:
                within_cluster_rmsd = pairwise_rmsd[
                    np.ix_(cluster_members, cluster_members)
                ]
                mean_rmsd = np.mean(within_cluster_rmsd, axis=1)
                center_idx = cluster_members[np.argmin(mean_rmsd)]
                cluster_centers.append(int(center_idx))

        return {
            "n_clusters": n_clusters,
            "labels": clustering.labels_.tolist(),
            "cluster_centers": cluster_centers,
            "noise_points": int(np.sum(clustering.labels_ == -1)),
        }

    def create_pseudoensemble(self) -> Dict:
        """Main method to create and analyze pseudoensemble"""
        print(f"Creating pseudoensemble centered on {self.center_pdb}")

        # Search for similar structures
        similar_pdbs = self.search_similar_structures()

        if len(similar_pdbs) == 0:
            print("No similar structures found!")
            return {"error": "No similar structures found"}

        # Download and parse structures
        structures = []
        pdb_ids = []

        for pdb_id in similar_pdbs[: self.max_structures]:
            try:
                pdb_file = self.download_structure(pdb_id)
                structure = self.parser.get_structure(pdb_id, pdb_file)
                structures.append(structure)
                pdb_ids.append(pdb_id)
                print(f"Downloaded and parsed {pdb_id}")
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")
                continue

        if len(structures) < 2:
            print("Insufficient structures for analysis")
            return {"error": "Insufficient structures for analysis"}

        print(
            f"\nAnalyzing structural heterogeneity for {len(structures)} structures..."
        )
        metrics = self.compute_structural_metrics(structures, center_idx=0)
        metrics["pdb_ids"] = pdb_ids
        metrics["center_structure"] = self.center_pdb

        print("\nIdentifying conformational states...")
        clustering = self.identify_conformational_states(structures)
        metrics["conformational_states"] = clustering

        return metrics

    def save_aligned_ensemble(self, output_dir: str = "ensemble_output"):
        """Save aligned structures to mmCIF files using BioPython MMCIFIO.
        """
        os.makedirs(output_dir, exist_ok=True)

        similar_pdbs = self.search_similar_structures()

        if len(similar_pdbs) == 0:
            print("No structures to save")
            return

        center_file = self.download_structure(self.center_pdb)
        center_structure = self.parser.get_structure(self.center_pdb, center_file)

        io = MMCIFIO()

        # Save center structure as reference
        io.set_structure(center_structure)
        io.save(os.path.join(output_dir, f"{self.center_pdb}_reference.cif"))

        # Align and save other structures
        saved_count = 0
        for pdb_id in similar_pdbs[1 : min(len(similar_pdbs), self.max_structures)]:
            try:
                pdb_file = self.download_structure(pdb_id)
                structure = self.parser.get_structure(pdb_id, pdb_file)

                # Align to center
                rmsd, _ = self.align_to_center(center_structure, structure)

                if rmsd != float("inf"):
                    io.set_structure(structure)
                    io.save(os.path.join(output_dir, f"{pdb_id}_aligned.cif"))
                    print(f"Saved aligned structure: {pdb_id} (RMSD: {rmsd:.2f} Å)")
                    saved_count += 1
            except Exception as e:
                print(f"Error aligning {pdb_id}: {e}")

        print(
            f"\nSaved {saved_count} aligned structures to {output_dir}/"
        )


def main():
    # Create ensemble analyzer
    ensemble = CustomPseudoensemble(
        center_pdb="6B8X",
        sequence_identity=40,
        cutoff_date="2023-01-01",
        max_structures=100,
    )

    # Run analysis
    results = ensemble.create_pseudoensemble()

    # Check for errors
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    # Print results
    print("\n=== Pseudoensemble Analysis Results ===")
    print(f"Center structure: {results['center_structure']}")
    print(f"Total structures: {results['n_structures']}")
    print(f"Mean RMSD to center: {results['mean_rmsd_to_center']:.2f} Å")
    print(f"Max RMSD to center: {results['max_rmsd_to_center']:.2f} Å")
    print(f"Mean pairwise RMSD: {results['mean_pairwise_rmsd']:.2f} Å")
    print(f"Mean RMSF: {results['mean_rmsf']:.2f} Å")

    if "conformational_states" in results:
        print(
            f"\nConformational states: {results['conformational_states']['n_clusters']}"
        )
        print(f"Noise points: {results['conformational_states']['noise_points']}")
        print(
            f"Cluster centers (indices): {results['conformational_states']['cluster_centers']}"
        )

    print(f"\nPCA variance explained: {results['pca_variance_explained']}")

    # Save aligned structures
    ensemble.save_aligned_ensemble()

    # Save results to JSON
    with open("pseudoensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to pseudoensemble_results.json")
    print("Aligned structures saved to ensemble_output/")


if __name__ == "__main__":
    main()
