import requests
from datetime import datetime, timezone
from Bio import PDB
from Bio.PDB import MMCIFParser
import urllib.request
import gzip
import os
import shutil
from typing import List, Dict, Set, Any
import re
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
import time
import json

try:
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover
    Retry = None  # type: ignore
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


class PDBClusterAnalyzer:
    def __init__(
        self,
        target_cluster: int,
        cutoff_date: str = None,
        temperature: float = 270.0,
        resolution: float = None,
        seq_identity: float = 0.4,
        max_workers: int | None = None,
        max_rate: float | None = None,
    ):
        self.target_cluster = target_cluster
        self.seq_identity = (
            int(seq_identity) if seq_identity > 1 else int(seq_identity * 100)
        )
        self.temperature = temperature  # Kelvin
        self.resolution = resolution  # Angstroms
        if max_workers is None:
            cpu = os.cpu_count() or 4
            max_workers = min(32, max(8, cpu * 4))
        self.max_workers = int(max_workers)
        self.max_rate = (
            max_rate  # requests/sec across all threads; None means unlimited
        )

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
        # Local cache for entry metadata to ensure deterministic results across runs
        self.cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "rcsb_entry")
        os.makedirs(self.cache_dir, exist_ok=True)

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

    def filter(self, cluster: Set[str]) -> List[str]:
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
                    temp = float(temp_str)
                    return temp if 0 < temp < 1000 else None
            except Exception:
                return None
            return None

        def get_resolution(data) -> float | None:
            try:
                r = data.get("rcsb_entry_info", {}).get("resolution_combined")
                if isinstance(r, list) and r:
                    return float(r[0])
                elif isinstance(r, (float, int)):
                    return float(r)
            except Exception:
                return None
            return None

        def get_experimental_deposited(data) -> bool | None:
            try:
                has_released = data.get("rcsb_accession_info", {}).get(
                    "has_released_experimental_data"
                )
                if isinstance(has_released, bool):
                    return has_released
                if isinstance(has_released, str):
                    if "Y" in has_released.upper():
                        return True
                    elif "N" in has_released.upper():
                        return False
                    else:
                        return None
            except Exception:
                return None
            return None

        def get_xray_experiment(data) -> bool | None:
            try:
                methods = data.get("exptl", [])
                if isinstance(methods, list) and methods:
                    method = methods[0].get("method")
                    if isinstance(method, str):
                        return "X-RAY" in method.upper()
                elif isinstance(methods, dict):
                    method = methods.get("method")
                    if isinstance(method, str):
                        return "X-RAY" in method.upper()
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
                        respect_retry_after_header=True,
                        raise_on_status=False,
                    )
                    adapter = HTTPAdapter(
                        max_retries=retry, pool_connections=10, pool_maxsize=10
                    )
                    sess.mount("http://", adapter)
                    sess.mount("https://", adapter)
                # Set a friendly User-Agent to avoid being treated as a bot
                sess.headers.update(
                    {"User-Agent": "distogram-analysis/0.1 (contact: research script)"}
                )
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

        def worker(entity: str) -> tuple[str, str | None]:
            pdb_id = entity.split("_")[0]
            if pdb_id == "AF":  # remove alphafold entries
                return ("filtered", None)
            data: Dict[str, Any] | None = None
            cache_path = os.path.join(self.cache_dir, f"{pdb_id}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception:
                    data = None

            if data is None:
                url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
                attempts = 3
                for i in range(attempts):
                    try:
                        if rate_limiter is not None:
                            rate_limiter.acquire()
                        resp = get_session().get(url, timeout=(5, 15))
                        if resp.status_code == 200:
                            data = resp.json()
                            try:
                                with open(cache_path, "w", encoding="utf-8") as fh:
                                    json.dump(data, fh)
                            except Exception:
                                pass
                            break
                    except Exception:
                        pass
                    time.sleep(0.5 * (i + 1))
                if data is None:
                    # Signal a fetch error for this entity
                    return ("error", entity)

            try:
                if self.cutoff_date is not None:
                    date_str = data.get("rcsb_accession_info", {}).get("deposit_date")
                    if not date_str:
                        return ("filtered", None)
                    deposition_date = parse_iso_date(date_str)
                    if deposition_date > self.cutoff_date:
                        return ("filtered", None)
                temp = get_temperature(data)
                if self.temperature is not None and (
                    temp is None or temp < self.temperature
                ):
                    return ("filtered", None)
                res = get_resolution(data)
                if self.resolution is not None and (
                    res is None or res > self.resolution
                ):
                    return ("filtered", None)
                has_released = get_experimental_deposited(data)
                if has_released is False:
                    return ("filtered", None)
                is_xray = get_xray_experiment(data)
                if is_xray is False:
                    return ("filtered", None)
            except Exception:
                return ("error", entity)
            return ("pass", entity)

        # Sort to ensure deterministic iteration order
        entities = sorted(list(cluster))
        filtered: List[str] = []
        errors: List[str] = []
        # Simple token-bucket rate limiter shared across threads
        rate_limiter = None
        if self.max_rate and self.max_rate > 0:

            class _RateLimiter:
                def __init__(self, rate: float, capacity: float):
                    self.rate = float(rate)
                    self.capacity = float(capacity)
                    self.tokens = float(capacity)
                    self.last = time.monotonic()
                    self.lock = threading.Lock()

                def acquire(self):
                    while True:
                        with self.lock:
                            now = time.monotonic()
                            elapsed = now - self.last
                            if elapsed > 0:
                                self.tokens = min(
                                    self.capacity, self.tokens + elapsed * self.rate
                                )
                                self.last = now
                            if self.tokens >= 1.0:
                                self.tokens -= 1.0
                                return
                            needed = 1.0 - self.tokens
                            sleep_for = needed / self.rate if self.rate > 0 else 0.05
                        time.sleep(min(0.5, max(0.0, sleep_for)))

            rate_limiter = _RateLimiter(
                rate=self.max_rate, capacity=max(1, self.max_workers)
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(worker, ent): ent for ent in entities}
            total = len(futures)
            pbar = (
                tqdm(total=total, desc="Fetching entries", unit="entry")
                if tqdm
                else None
            )
            done = 0
            step = max(1, total // 100) if total else 1
            for fut in as_completed(futures):
                status, res = fut.result()
                if status == "pass" and res is not None:
                    filtered.append(res)
                elif status == "error" and res is not None:
                    errors.append(res)
                done += 1
                if pbar:
                    pbar.update(1)
                else:
                    if done % step == 0 or done == total:
                        print(
                            f"\rProcessed {done}/{total} ({(done / total) * 100:.0f}%)",
                            end="",
                            flush=True,
                        )
            if pbar:
                pbar.close()
            else:
                print()

        if errors:
            # Deterministic failure instead of silently dropping items
            errors_sorted = ", ".join(sorted(errors))
            raise RuntimeError(
                f"Failed to fetch metadata for {len(errors)} entries: {errors_sorted}. "
                "Re-run later or set a lower --workers/--max-rate."
            )
        return sorted(filtered)

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
        filtered_pdbs = self.filter(target_cluster)
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
        help="Data collection temperature cutoff in Kelvin",
        default=None,
    )
    parser.add_argument(
        "--resolution",
        type=float,
        help="Resolution cutoff in Angstroms (e.g., 2.5)",
        default=None,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers for API requests (default scales with CPU)",
        default=None,
    )
    parser.add_argument(
        "--max-rate",
        type=float,
        help="Max request rate across all workers (requests/sec). Default unlimited.",
        default=None,
    )

    args = parser.parse_args()

    print("=== Getting Temperature Filtered Structures ===")
    print(f"Target cluster: {args.target}")
    print(
        f"Sequence identity threshold: {args.seq_identity * 100 if args.seq_identity < 1 else args.seq_identity}%"
    )
    if args.temperature:
        print(f"Temperature cutoff: {args.temperature} K")
    if args.cutoff_date:
        print(f"Structure cutoff date: {args.cutoff_date}")
    print()

    analyzer = PDBClusterAnalyzer(
        target_cluster=int(args.target),
        cutoff_date=args.cutoff_date,
        temperature=args.temperature,
        resolution=args.resolution,
        seq_identity=args.seq_identity,
        max_workers=args.workers,
        max_rate=args.max_rate,
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
