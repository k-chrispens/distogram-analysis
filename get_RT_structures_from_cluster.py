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
        find_altlocs: bool = False,
    ):
        # Core settings
        self.target_cluster = int(target_cluster)
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
        self.find_altlocs = bool(find_altlocs)

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
        self.clusters_url = f"https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{self.seq_identity}.txt"

        # Output and caches
        self.output_dir = f"{self.target_cluster}_roomtemp_clusters"
        self.cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        self.entry_cache_dir = os.path.join(self.cache_dir, "rcsb_entry")
        self.mmcif_cache_dir = os.path.join(self.cache_dir, "mmcif")
        os.makedirs(self.entry_cache_dir, exist_ok=True)
        os.makedirs(self.mmcif_cache_dir, exist_ok=True)

        # Altloc results
        self._altloc_results_lock = threading.Lock()
        self.altloc_results = {}

    # -------------------- AltLoc helpers --------------------
    def _download_mmcif_cached(self, pdb_id: str) -> str | None:
        """Download and cache the mmCIF file for a PDB ID, returning path to .cif.

        Saves to .cache/mmcif/{pdb_id}.cif. Returns None on hard failure.
        """
        target = os.path.join(self.mmcif_cache_dir, f"{pdb_id}.cif")
        if os.path.exists(target) and os.path.getsize(target) > 0:
            return target
        # Download gz and decompress to target
        url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
        tmp_gz = os.path.join(self.mmcif_cache_dir, f"{pdb_id}.cif.gz.tmp")
        final_gz = os.path.join(self.mmcif_cache_dir, f"{pdb_id}.cif.gz")
        try:
            # Avoid stampede: if another thread already fetched, reuse
            if os.path.exists(final_gz) and os.path.getsize(final_gz) > 0:
                gz_path = final_gz
            else:
                urllib.request.urlretrieve(url, tmp_gz)
                os.replace(tmp_gz, final_gz)
                gz_path = final_gz
            with (
                gzip.open(gz_path, "rt") as fin,
                open(target, "w", encoding="utf-8") as fout,
            ):
                shutil.copyfileobj(fin, fout)
            return target
        except Exception:
            # Clean up tmp file if present
            try:
                if os.path.exists(tmp_gz):
                    os.remove(tmp_gz)
            except Exception:
                pass
            return None

    def _analyze_altloc_segments(self, cif_path: str) -> Dict[str, Any]:
        """Analyze altloc segments using Biotite.

        Returns a dict with the following shape (JSON-serializable):
          {
            "present": bool,
            "max_segment": int,
            "segments": {
               <chain_id>: [
                   {
                       "length": int,
                       "residues": [ {"res_id": int, "res_name": str or None}, ... ]
                   },
                   ...
               ],
               ...
            }
          }

        This records the specific residues that comprise each contiguous altloc segment
        (instead of only the segment lengths).
        """
        try:
            # Lazy imports to avoid hard dependency if flag not used
            import numpy as np  # type: ignore
            import biotite.structure.io.pdbx as pdbx  # type: ignore
        except Exception as e:  # pragma: no cover
            return {
                "present": False,
                "max_segment": 0,
                "error": f"biotite import failed: {e}",
            }

        # Resolve PDBx/mmCIF file class and get_structure function dynamically
        PDBxLike = getattr(pdbx, "PDBxFile", None)
        if PDBxLike is None:
            for alt_name in ("MMCIFFile", "CIFFile", "BinaryCIFFile"):
                PDBxLike = getattr(pdbx, alt_name, None)
                if PDBxLike is not None:
                    break
        get_structure = getattr(pdbx, "get_structure", None)
        if PDBxLike is None or get_structure is None:
            # Provide available attributes to aid debugging
            avail = sorted([a for a in dir(pdbx) if not a.startswith("_")])
            return {
                "present": False,
                "max_segment": 0,
                "error": f"biotite pdbx reader not available (have: {', '.join(avail)})",
            }

        try:
            with open(cif_path, "r", encoding="utf-8") as f:
                pdbx_file = PDBxLike.read(f)  # type: ignore[call-arg]
            # Ensure we include altloc-related fields
            atoms = get_structure(
                pdbx_file,
                model=1,
                altloc="all",
            )  # type: ignore[arg-type]
        except Exception as e:
            return {"present": False, "max_segment": 0, "error": f"parse failed: {e}"}

        def has_alt(aid):
            return aid not in (None, "", ".", "?")

        segments_per_chain: Dict[str, list[Dict[str, Any]]] = {}
        max_seg = 0

        # Choose altloc field name supported by this Biotite version
        alt_field = (
            "altloc_id"
            if hasattr(atoms, "altloc_id")
            else (
                "label_alt_id"
                if hasattr(atoms, "label_alt_id")
                else ("alt_id" if hasattr(atoms, "alt_id") else None)
            )
        )

        # Iterate per chain preserving order
        for chain in np.unique(atoms.chain_id):  # type: ignore[attr-defined]
            sel = atoms.chain_id == chain  # type: ignore[index]
            chain_atoms = atoms[sel]
            # Walk residues by res_id in encountered order
            res_ids = chain_atoms.res_id  # type: ignore[attr-defined]
            alt_ids = getattr(chain_atoms, alt_field) if alt_field else None  # type: ignore[attr-defined]
            res_names = getattr(chain_atoms, "res_name", None)  # type: ignore[attr-defined]

            # Build an ordered list of residues and a boolean per residue indicating
            # presence of any altloc atom on that residue.
            residues_ordered: list[tuple[int, Any]] = []  # (res_id, res_name)
            res_flags: list[bool] = []
            last_res = None
            current_has_alt = False
            current_name = None
            n_atoms = len(res_ids)

            # Helper to safely index optional arrays
            def _get_or_none(arr, i):
                try:
                    return arr[i] if arr is not None else None
                except Exception:
                    return None

            if alt_ids is None:
                # No altloc field available in this Biotite version -> record residues but no segments
                seen: set[int] = set()
                for i in range(n_atoms):
                    rid = int(res_ids[i])  # type: ignore[index]
                    if rid not in seen:
                        rname = _get_or_none(res_names, i)
                        residues_ordered.append(
                            (rid, None if rname is None else str(rname))
                        )
                        res_flags.append(False)
                        seen.add(rid)
            else:
                for i in range(n_atoms):
                    rid = res_ids[i]  # type: ignore[index]
                    aid = alt_ids[i]
                    rname = _get_or_none(res_names, i)
                    if last_res is None:
                        last_res = rid
                        current_has_alt = has_alt(aid)
                        current_name = rname
                    elif rid != last_res:
                        # push previous residue aggregate
                        residues_ordered.append(
                            (
                                int(last_res),
                                None if current_name is None else str(current_name),
                            )
                        )
                        res_flags.append(bool(current_has_alt))
                        # reset for new residue
                        last_res = rid
                        current_has_alt = has_alt(aid)
                        current_name = rname
                    else:
                        # same residue, accumulate altloc presence
                        if has_alt(aid):
                            current_has_alt = True
                        # keep first seen name
                if last_res is not None:
                    residues_ordered.append(
                        (
                            int(last_res),
                            None if current_name is None else str(current_name),
                        )
                    )
                    res_flags.append(bool(current_has_alt))

            # Compute contiguous segments and list the specific residues for each segment
            segs: list[Dict[str, Any]] = []
            current_segment: list[Dict[str, Any]] = []
            for flag, (rid, rname) in zip(res_flags, residues_ordered):
                if flag:
                    current_segment.append({"res_id": int(rid), "res_name": rname})
                else:
                    if current_segment:
                        segs.append(
                            {
                                "length": len(current_segment),
                                "residues": current_segment,
                            }
                        )
                        max_seg = max(max_seg, len(current_segment))
                        current_segment = []
            if current_segment:
                segs.append(
                    {
                        "length": len(current_segment),
                        "residues": current_segment,
                    }
                )
                max_seg = max(max_seg, len(current_segment))

            if segs:
                segments_per_chain[str(chain)] = segs

        present = any(len(v) > 0 for v in segments_per_chain.values())
        return {
            "present": present,
            "max_segment": int(max_seg),
            "segments": segments_per_chain,
        }

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
            cache_path = os.path.join(self.entry_cache_dir, f"{pdb_id}.json")
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
            # if not filtered yet and altloc analysis requested
            if self.find_altlocs:
                cif_path = self._download_mmcif_cached(pdb_id)
                if cif_path:
                    altres = self._analyze_altloc_segments(cif_path)
                    with self._altloc_results_lock:
                        self.altloc_results[pdb_id] = altres
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

        # Persist altloc results if requested
        if self.find_altlocs and self.altloc_results:
            out_path = os.path.join(
                self.cache_dir,
                f"altloc_segments_cluster{self.target_cluster}_seqid{self.seq_identity}.json",
            )
            try:
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(self.altloc_results, fh, indent=2, sort_keys=True)
                print(
                    f"Altloc analysis written to {out_path} ({len(self.altloc_results)} entries)"
                )
            except Exception as e:
                print(f"Warning: failed to write altloc results: {e}")

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
    parser.add_argument(
        "--find-altlocs",
        action="store_true",
        help="For structures that pass filters, cache mmCIFs and record altloc segment presence and lengths using Biotite.",
        default=False,
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
        find_altlocs=args.find_altlocs,
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
