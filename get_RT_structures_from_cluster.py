import requests
from datetime import datetime, timezone
from Bio import PDB
from Bio.PDB import MMCIFParser
import urllib.request
import gzip
import os
import shutil
from typing import List, Dict, Set
import re
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover
    Retry = None  # type: ignore


class PDBClusterAnalyzer:
    def __init__(
        self,
        target_cluster: int,
        cutoff_date: str = None,
        temperature: float = 270.0,
        seq_identity: float = 0.4,
        max_workers: int | None = None,
    ):
        self.target_cluster = target_cluster
        self.seq_identity = (
            int(seq_identity) if seq_identity > 1 else int(seq_identity * 100)
        )
        self.temperature = float(temperature)
        if max_workers is None:
            cpu = os.cpu_count() or 4
            max_workers = min(32, max(8, cpu * 4))
        self.max_workers = int(max_workers)

        if cutoff_date:
            try:
                dt = datetime.fromisoformat(cutoff_date.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.strptime(cutoff_date[:10], "%Y-%m-%d")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            self.cutoff_date = dt
        else:
            self.cutoff_date = None

        self.parser = PDB.PDBParser(QUIET=True)
        self.mmcif_parser = MMCIFParser(QUIET=True)
        self.clusters_url = f"https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{int(seq_identity) if seq_identity > 1 else int(seq_identity * 100)}.txt"

        self.output_dir = f"{self.target_cluster}_roomtemp_clusters"

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

    def filter_by_temperature(self, cluster: Set[str]) -> List[str]:
        # Robust temperature extraction supporting dict or list under 'diffrn'
        def get_temperature(data) -> float | None:
            try:
                diffrn = data.get("diffrn", {})
                if isinstance(diffrn, list) and diffrn:
                    temp_str = diffrn[0].get("ambient_temp")
                elif isinstance(diffrn, dict):
                    temp_str = diffrn.get("ambient_temp")
                else:
                    temp_str = None
                if temp_str is not None:
                    return float(temp_str)
            except Exception:
                return None
            return None

        # Thread-local session for connection pooling and retries
        tls = threading.local()

        def get_session() -> requests.Session:
            sess = getattr(tls, "session", None)
            if sess is None:
                sess = requests.Session()
                if Retry is not None:
                    retry = Retry(
                        total=3,
                        read=3,
                        connect=3,
                        backoff_factor=0.3,
                        status_forcelist=(429, 500, 502, 503, 504),
                        allowed_methods=("GET",),
                        raise_on_status=False,
                    )
                    adapter = HTTPAdapter(
                        max_retries=retry, pool_connections=10, pool_maxsize=10
                    )
                    sess.mount("http://", adapter)
                    sess.mount("https://", adapter)
                tls.session = sess
            return sess

        def parse_iso_date(date_str: str) -> datetime:
            try:
                d = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                d = datetime.strptime(date_str[:10], "%Y-%m-%d")
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d

        def worker(entity: str) -> str | None:
            pdb_id = entity.split("_")[0]
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            try:
                resp = get_session().get(url, timeout=15)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                if not self.cutoff_date:
                    date_str = data.get("rcsb_accession_info", {}).get("deposit_date")
                    if not date_str:
                        return None
                    deposition_date = parse_iso_date(date_str)
                    if deposition_date > self.cutoff_date:
                        return None
                temp = get_temperature(data)
                if temp is not None and temp >= self.temperature:
                    return entity
            except Exception:
                return None
            return None

        entities = list(cluster)
        filtered: List[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(worker, ent): ent for ent in entities}
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    filtered.append(res)

        return filtered

    def download_structure(
        self, pdb_id: str, base_folder: str = None, format: str = "mmcif"
    ) -> str:
        """Download structure file in specified format (mmcif or pdb), checking multiple locations"""
        if base_folder is None:
            base_folder = f"{self.output_dir}/structures"

        os.makedirs(base_folder, exist_ok=True)

        # Define file paths and URLs
        ext = "cif" if format.lower() == "mmcif" else "pdb"
        target_filename = f"{base_folder}/{pdb_id}.{ext}"

        # Check if file already exists in target location or common locations
        for path in [target_filename, f"structures/{pdb_id}.{ext}", f"{pdb_id}.{ext}"]:
            if os.path.exists(path):
                if path != target_filename:
                    print(
                        f"Found existing {pdb_id} at {path}, copying to {target_filename}"
                    )
                    shutil.copy2(path, target_filename)
                return target_filename

        # Download if not found
        url = f"https://files.rcsb.org/download/{pdb_id}.{ext}.gz"
        gz_filename = f"{pdb_id}.{ext}.gz"

        try:
            print(f"Downloading {pdb_id}...")
            urllib.request.urlretrieve(url, gz_filename)

            with gzip.open(gz_filename, "rt") as f_in:
                with open(target_filename, "w") as f_out:
                    f_out.write(f_in.read())
            os.remove(gz_filename)
            print(f"Downloaded and saved {pdb_id} to {target_filename}")

        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")
            # Try alternative format if mmcif fails
            if format.lower() == "mmcif":
                print(f"Trying PDB format for {pdb_id}...")
                return self.download_structure(pdb_id, base_folder, "pdb")
            raise

        return target_filename

    def run_analysis(self) -> Dict:
        print("Fetching sequence clusters...")
        clusters = self.fetch_clusters()
        print(f"Found {len(clusters)} clusters")

        print(f"Getting cluster {self.target_cluster}...")
        try:
            target_cluster = clusters[self.target_cluster - 1]
        except IndexError:
            raise ValueError(f"Cluster index {self.target_cluster} out of range")

        print(f"Cluster size: {len(target_cluster)} entities")

        print("Filtering by date and temperature...")
        filtered_pdbs = self.filter_by_temperature(target_cluster)
        print(f"Filtered to {len(filtered_pdbs)} structures")

        return filtered_pdbs


def main():
    parser = argparse.ArgumentParser(
        description="Analyze structural heterogeneity within RCSB sequence clusters and create curated ensembles"
    )
    parser.add_argument(
        "target",
        help="Target cluster for analysis - the largest cluster is 1, decreasing in size up to the number of clusters available",
    )
    parser.add_argument(
        "--cutoff-date",
        help="Cutoff date for structure filtering (YYYY-MM-DD)",
        default=None,
    )
    parser.add_argument(
        "--seq-identity",
        type=float,
        help="Sequence identity threshold for clustering (default: 0.4)",
        default=0.4,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Data collection temperature cutoff in Kelvin (default: 270K)",
        default=270.0,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers for API requests (default scales with CPU)",
        default=None,
    )

    args = parser.parse_args()

    print("=== Getting Temperature Filtered Structures ===")
    print(f"Target cluster: {args.target}")
    print(
        f"Sequence identity threshold: {args.seq_identity * 100 if args.seq_identity < 1 else args.seq_identity}%"
    )
    print(f"Temperature cutoff: {args.temperature} K")
    if args.cutoff_date:
        print(f"Structure cutoff date: {args.cutoff_date}")
    print()

    analyzer = PDBClusterAnalyzer(
        target_cluster=int(args.target),
        cutoff_date=args.cutoff_date,
        temperature=args.temperature,
        seq_identity=args.seq_identity,
        max_workers=args.workers,
    )

    try:
        results = analyzer.run_analysis()

        print("PDB IDs passing filters:")
        for pdb_id in results:
            print(pdb_id)

    except Exception as e:
        print(f"Analysis failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
