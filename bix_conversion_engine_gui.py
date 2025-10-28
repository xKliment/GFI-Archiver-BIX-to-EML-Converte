import gzip
import os
import shutil
import threading
import time
import queue
import hashlib
import logging
import gc
import psutil
from typing import Optional, List, Dict, Any, Callable, Iterator
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Precompute inversion translation table for fast byte-wise inversion
INVERT_TABLE = bytes.maketrans(bytes(range(256)), bytes((~b & 0xFF) for b in range(256)))


class BandwidthLimiter:
    """Thread-safe token bucket for global bandwidth throttling."""

    def __init__(self, rate_bytes_per_sec: Optional[int]):
        self.rate = rate_bytes_per_sec or 0
        self._tokens = float(self.rate)
        self._last = time.time()
        self._lock = threading.Lock()

    def set_rate(self, rate_bytes_per_sec: Optional[int]):
        with self._lock:
            self.rate = rate_bytes_per_sec or 0
            self._tokens = float(self.rate)
            self._last = time.time()

    def consume(self, nbytes: int):
        if self.rate <= 0:
            return
        with self._lock:
            now = time.time()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.rate, self._tokens + elapsed * self.rate)
            needed = nbytes - self._tokens
            if needed <= 0:
                self._tokens -= nbytes
                return
            # Need to wait
            wait_sec = needed / self.rate
        # Sleep outside lock
        time.sleep(wait_sec)
        # After sleeping, deduct tokens
        with self._lock:
            self._tokens = max(0.0, self._tokens - nbytes)


class _InvertingReader:
    """File-like wrapper that inverts bits on-the-fly when reading.

    Optionally updates a provided hashlib hasher with the original bytes to
    avoid extra passes over the input for hashing.
    """

    def __init__(self, raw: io.BufferedReader, hasher: Optional[hashlib._hashlib.HASH] = None,
                 limiter: Optional[BandwidthLimiter] = None, chunk_size: int = 1024 * 1024):
        self._raw = raw
        self._hasher = hasher
        self._limiter = limiter
        self._chunk = max(64 * 1024, int(chunk_size))

    def read(self, size: int = -1) -> bytes:
        # Normalize to fixed chunk sizes for more predictable throttling
        if size < 0:
            size = self._chunk
        data = self._raw.read(size)
        if not data:
            return data
        if self._hasher is not None:
            self._hasher.update(data)
        out = data.translate(INVERT_TABLE)
        if self._limiter is not None:
            self._limiter.consume(len(out))
        return out


class ConversionResult:
    """Result of a single file conversion"""
    def __init__(self, success: bool, source_path: str, target_path: str = "", 
                 error_message: str = "", conversion_time: float = 0.0, 
                 file_size: int = 0, file_hash: str = ""):
        self.success = success
        self.source_path = source_path
        self.target_path = target_path
        self.error_message = error_message
        self.conversion_time = conversion_time
        self.file_size = file_size
        self.file_hash = file_hash
        self.timestamp = datetime.now()


class BatchProgressTracker:
    """Tracks progress for batch processing with throttled updates"""
    
    def __init__(self, total_files: int, batch_size: int = 10000, 
                 update_interval: float = 0.1, progress_callback: Optional[Callable] = None):
        self.total_files = total_files
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.progress_callback = progress_callback
        
        self.processed_files = 0
        self.failed_files = 0
        self.last_update_time = time.time()
        self.lock = threading.Lock()
        
        # Statistics for sampling
        self.sample_times = []
        self.sample_size = min(1000, total_files // 100)  # Sample 1% or 1000 files
        # Bytes tracking for MB-based progress
        self.processed_bytes = 0
        self.estimated_total_bytes = 0
        
    def update_progress(self, success: bool, result: Optional[ConversionResult] = None,
                        processed_bytes_delta: int = 0, estimated_total_bytes: Optional[int] = None):
        """Update progress with throttling"""
        with self.lock:
            if success:
                self.processed_files += 1
            else:
                self.failed_files += 1
            if processed_bytes_delta:
                try:
                    self.processed_bytes += max(0, int(processed_bytes_delta))
                except Exception:
                    pass
            if estimated_total_bytes is not None:
                try:
                    self.estimated_total_bytes = max(self.estimated_total_bytes, int(estimated_total_bytes))
                except Exception:
                    pass
            
            # Track sample times for speed calculation
            if result and len(self.sample_times) < self.sample_size:
                self.sample_times.append(result.conversion_time)
            
            current_time = time.time()
            time_since_last_update = current_time - self.last_update_time
            
            # Throttle updates based on interval or progress milestones
            # Guard against division by zero when total_files < 100
            divisor = max(1, self.total_files // 100)
            progress_milestone = ((self.processed_files + self.failed_files) % divisor == 0)
            
            if time_since_last_update >= self.update_interval or progress_milestone:
                self.last_update_time = current_time
                if self.progress_callback:
                    try:
                        self.progress_callback(
                            self.processed_files,
                            self.failed_files,
                            self.total_files,
                            self.processed_bytes,
                            self.estimated_total_bytes
                        )
                    except TypeError:
                        # Backwards compatibility if GUI expects only 3 args
                        self.progress_callback(self.processed_files, self.failed_files, self.total_files)
    
    def get_estimated_speed(self) -> float:
        """Get estimated processing speed based on sample"""
        if not self.sample_times:
            return 0.0
        
        avg_time_per_file = sum(self.sample_times) / len(self.sample_times)
        if avg_time_per_file > 0:
            return 60.0 / avg_time_per_file  # files per minute
        return 0.0


class BixConversionEngine:
    """BIX to EML conversion engine with batch processing for large datasets"""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 10000,
                 progress_callback: Optional[Callable] = None, memory_limit_gb: float = 2.0,
                 database: Optional['ConversionDatabase'] = None,
                 duplication_strategy: str = "output_exists",
                 throttle_bytes_per_sec: Optional[int] = None,
                 chunk_size: int = 1024 * 1024,
                 fsync_on_write: bool = False,
                 retry_attempts: int = 2,
                 retry_backoff_base: float = 0.5):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.db = database
        # duplication_strategy: "none" | "output_exists" | "ledger_hash"
        self.duplication_strategy = duplication_strategy
        self.throttle_bytes_per_sec = throttle_bytes_per_sec
        self._limiter = BandwidthLimiter(throttle_bytes_per_sec)
        self.chunk_size = max(64 * 1024, int(chunk_size))
        self.fsync_on_write = bool(fsync_on_write)
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_backoff_base = max(0.0, float(retry_backoff_base))
        
        self.is_running = False
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.start_time = None
        self.progress_tracker = None
        self.bytes_processed = 0
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in bytes"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit"""
        return self.get_memory_usage() > self.memory_limit_bytes
    
    def force_garbage_collection(self):
        """Force garbage collection if memory is high"""
        if self.check_memory_limit():
            gc.collect()
            logger.info(f"Forced garbage collection. Memory usage: {self.get_memory_usage() / (1024*1024):.1f} MB")
    
    def is_gzip(self, data: bytes) -> bool:
        """Check if data starts with gzip magic bytes"""
        return len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B
    
    def invert_bits(self, data: bytes) -> bytes:
        """Invert all bits in the data using a precomputed translation table"""
        return data.translate(INVERT_TABLE)
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash for file (separate pass)."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb", buffering=1024 * 1024) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def bix_to_eml(self, src_path: str, dst_path: str, compute_hash: bool = True) -> ConversionResult:
        """Convert a single .bix file to .eml with streaming I/O and bit inversion.

        - Reads source in chunks; avoids loading entire file into memory.
        - Detects gzip header after inversion using the first two bytes.
        - Streams gzip decompression when needed via a bit-inverting wrapper.
        - Writes to a temporary file and renames atomically.
        """
        start_time = time.time()
        file_size = 0
        file_hash = ""

        tmp_path = f"{dst_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            # Stat once
            file_size = os.path.getsize(src_path)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)

            # Prepare optional hasher
            hasher = hashlib.md5() if compute_hash else None

            with open(src_path, 'rb', buffering=self.chunk_size) as src:
                # Peek first 2 bytes (non-destructive)
                head = src.read(2)
                if not head:
                    raise Exception("Source file is empty")
                inv_head = head.translate(INVERT_TABLE)
                # Reset to start for streaming
                src.seek(0)

                # Decide path: gzip or plain inverted stream
                if len(inv_head) >= 2 and inv_head[0] == 0x1F and inv_head[1] == 0x8B:
                    # Stream-decompress via inverting reader
                    inv_reader = _InvertingReader(src, hasher, limiter=self._limiter, chunk_size=self.chunk_size)
                    try:
                        with gzip.GzipFile(fileobj=inv_reader, mode='rb') as gz, open(tmp_path, 'wb', buffering=self.chunk_size) as out:
                            while True:
                                chunk = gz.read(self.chunk_size)
                                if not chunk:
                                    break
                                if self._limiter is not None:
                                    self._limiter.consume(len(chunk))
                                out.write(chunk)
                            if self.fsync_on_write:
                                try:
                                    out.flush()
                                    os.fsync(out.fileno())
                                except Exception:
                                    pass
                    except gzip.BadGzipFile as e:
                        # Fallback: write inverted bytes as-is
                        logger.warning(f"Gzip decompression failed for {src_path}: {e}, writing inverted data")
                        src.seek(0)
                        with open(tmp_path, 'wb', buffering=self.chunk_size) as out:
                            for chunk in iter(lambda: src.read(self.chunk_size), b""):
                                if hasher is not None:
                                    hasher.update(chunk)
                                inv = chunk.translate(INVERT_TABLE)
                                if self._limiter is not None:
                                    self._limiter.consume(len(inv))
                                out.write(inv)
                            if self.fsync_on_write:
                                try:
                                    out.flush()
                                    os.fsync(out.fileno())
                                except Exception:
                                    pass
                else:
                    # Plain inverted stream copy
                    with open(tmp_path, 'wb', buffering=self.chunk_size) as out:
                        for chunk in iter(lambda: src.read(self.chunk_size), b""):
                            if hasher is not None:
                                hasher.update(chunk)
                            inv = chunk.translate(INVERT_TABLE)
                            if self._limiter is not None:
                                self._limiter.consume(len(inv))
                            out.write(inv)
                        if self.fsync_on_write:
                            try:
                                out.flush()
                                os.fsync(out.fileno())
                            except Exception:
                                pass

            # Atomically move to final destination
            os.replace(tmp_path, dst_path)

            if hasher is not None:
                file_hash = hasher.hexdigest()

            conversion_time = time.time() - start_time
            return ConversionResult(
                success=True,
                source_path=src_path,
                target_path=dst_path,
                conversion_time=conversion_time,
                file_size=file_size,
                file_hash=file_hash
            )

        except Exception as e:
            # Clean up temp file on failure
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

            conversion_time = time.time() - start_time
            error_msg = f"Conversion failed: {str(e)}"
            logger.error(f"Error converting {src_path}: {error_msg}")

            return ConversionResult(
                success=False,
                source_path=src_path,
                error_message=error_msg,
                conversion_time=conversion_time,
                file_size=file_size,
                file_hash=file_hash
            )
    
    def find_bix_files_batch(self, root_path: str, batch_callback: Callable[[List[str]], None]) -> int:
        """
        Find all .bix files in directory tree and process in batches
        Returns total count of files found
        """
        total_count = 0
        current_batch = []
        root_path = Path(root_path)
        
        if not root_path.exists() or not root_path.is_dir():
            raise Exception(f"Invalid directory: {root_path}")
        
        try:
            # Use pathlib for better performance with large directories
            for bix_file in root_path.rglob("*.bix"):
                if bix_file.is_file() and os.access(bix_file, os.R_OK):
                    current_batch.append(str(bix_file))
                    total_count += 1
                    
                    # Process batch when it reaches the batch size
                    if len(current_batch) >= self.batch_size:
                        batch_callback(current_batch.copy())
                        current_batch.clear()
                        
                        # Force garbage collection periodically
                        if total_count % (self.batch_size * 10) == 0:
                            self.force_garbage_collection()
            
            # Process remaining files in the last batch
            if current_batch:
                batch_callback(current_batch)
                
        except Exception as e:
            raise Exception(f"Error scanning directory: {str(e)}")
            
        return total_count
    
    def convert_single_file(self, args: tuple) -> ConversionResult:
        """Convert a single file (for thread pool)"""
        src_path, dst_path, compute_hash = args
        # Retry wrapper for transient I/O errors
        attempts = self.retry_attempts + 1
        last_result: Optional[ConversionResult] = None
        for i in range(attempts):
            result = self.bix_to_eml(src_path, dst_path, compute_hash=compute_hash)
            last_result = result
            if result.success:
                break
            # Simple heuristic: backoff on common transient errors
            emsg = (result.error_message or "").lower()
            transient = any(k in emsg for k in [
                "temporarily", "timeout", "resource busy", "sharing violation",
                "access is denied", "network", "permission", "device or resource busy"
            ])
            if i < attempts - 1 and transient:
                delay = self.retry_backoff_base * (2 ** i)
                time.sleep(delay)
                continue
            else:
                break
        result = last_result if last_result is not None else ConversionResult(False, src_path)
        if result.success:
            # Track processed bytes for stats
            try:
                self.bytes_processed += int(result.file_size)
            except Exception:
                pass
        return result
    
    def process_file_batch(self, file_batch: List[str], input_path: str, output_path: str,
                          failed_folder: Optional[str] = None,
                          session_id: Optional[str] = None,
                          executor: Optional[ThreadPoolExecutor] = None) -> List[ConversionResult]:
        """Process a batch of files"""
        # Prepare conversion tasks for this batch
        conversion_tasks = []
        skipped_results = []
        
        for bix_file in file_batch:
            # Determine output path
            rel_path = os.path.relpath(bix_file, input_path)
            eml_path = os.path.join(output_path, os.path.splitext(rel_path)[0] + '.eml')

            # Duplicate/skip strategy
            skip = False
            compute_hash = True
            if self.duplication_strategy == "output_exists":
                if os.path.exists(eml_path):
                    skip = True
                    compute_hash = False
            elif self.duplication_strategy == "ledger_hash" and self.db is not None:
                # Hash to check ledger before converting (extra I/O)
                try:
                    file_hash = self.get_file_hash(bix_file)
                except Exception:
                    file_hash = ""
                if file_hash and self.db.is_already_converted(bix_file, file_hash):
                    skip = True
                # If converting, we don't need to recompute hash inside conversion
                compute_hash = not skip
            else:
                # "none" or unknown: always convert
                compute_hash = True

            if skip:
                # Update progress and optionally record as skipped (no DB write to avoid hash conflicts)
                self.progress_tracker.update_progress(True)
                continue

            conversion_tasks.append((bix_file, eml_path, compute_hash))

        # Convert files using thread pool
        results = []
        created_local_executor = False
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=self.max_workers)
            created_local_executor = True
        # Submit all tasks
        future_to_task = {
            executor.submit(self.convert_single_file, task): task
            for task in conversion_tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            if not self.is_running:
                # Cancel remaining tasks
                for f in future_to_task:
                    f.cancel()
                break
            
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                
                # Update progress tracker with bytes
                bytes_delta = result.file_size if result.success else 0
                self.progress_tracker.update_progress(result.success, result,
                                                      processed_bytes_delta=bytes_delta)

                # Flush to DB per result if available
                if session_id and self.db is not None and result:
                    try:
                        self.db.log_conversion_batch([result], session_id)
                    except Exception:
                        pass
                
                # Handle failed files
                if not result.success and failed_folder:
                    try:
                        failed_filename = f"{os.path.basename(result.source_path)}.error"
                        failed_path = os.path.join(failed_folder, failed_filename)
                        shutil.copy2(result.source_path, failed_path)
                        
                        # Create error file
                        error_file = failed_path + ".error.txt"
                        with open(error_file, 'w') as f:
                            f.write(f"Error: {result.error_message}\n")
                            f.write(f"Original path: {result.source_path}\n")
                            f.write(f"File size: {result.file_size} bytes\n")
                            f.write(f"File hash: {result.file_hash}\n")
                            f.write(f"Timestamp: {result.timestamp}\n")
                            f.write(f"Conversion time: {result.conversion_time:.3f}s\n")
                    except Exception as copy_error:
                        logger.error(f"Failed to copy failed file: {copy_error}")
                        
            except Exception as e:
                logger.error(f"Error processing task {task}: {e}")
                self.progress_tracker.update_progress(False)

        if created_local_executor:
            executor.shutdown(wait=True)

        return results
    
    def convert_directory(self, input_path: str, output_path: str,
                          failed_folder: Optional[str] = None,
                          session_id: Optional[str] = None) -> List[ConversionResult]:
        """Streaming convert: scan and convert concurrently with MB-based progress."""
        if not self.is_running:
            self.is_running = True
            self.start_time = time.time()

        if failed_folder:
            os.makedirs(failed_folder, exist_ok=True)

        # Reset stats
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.bytes_processed = 0
        self.total_bytes = 0
        self.estimated_total_bytes = 0

        file_queue: "queue.Queue[Tuple[str,int]]" = queue.Queue(maxsize=max(1, self.batch_size * 2))
        scanning_done = threading.Event()

        def scanner():
            try:
                local_total_files = 0
                local_total_bytes = 0
                for bix in Path(input_path).rglob("*.bix"):
                    if not self.is_running:
                        break
                    try:
                        if bix.is_file() and os.access(bix, os.R_OK):
                            st = bix.stat()
                            size = int(getattr(st, 'st_size', 0))
                            file_queue.put((str(bix), size))
                            local_total_files += 1
                            local_total_bytes += size
                            # Periodically update estimates
                            if local_total_files % 500 == 0 and self.progress_callback:
                                try:
                                    # New: explicit scan update (message sentinel)
                                    try:
                                        self.progress_callback(("scan", local_total_files, local_total_bytes))
                                    except TypeError:
                                        pass
                                    # Back-compat: numeric tuple
                                    self.progress_callback(
                                        self.processed_files,
                                        self.failed_files,
                                        local_total_files,
                                        self.bytes_processed,
                                        local_total_bytes
                                    )
                                except TypeError:
                                    self.progress_callback(self.processed_files, self.failed_files, local_total_files)
                    except Exception:
                        continue
                # Done
                self.total_files = local_total_files
                self.total_bytes = local_total_bytes
                self.estimated_total_bytes = local_total_bytes
                # Final scan update so UI can stop scan indicator
                if self.progress_callback:
                    try:
                        self.progress_callback(("scan", local_total_files, local_total_bytes))
                    except Exception:
                        pass
            finally:
                scanning_done.set()

        threading.Thread(target=scanner, daemon=True).start()

        # Progress tracker initially with unknown totals
        self.progress_tracker = BatchProgressTracker(
            total_files=0,
            batch_size=self.batch_size,
            update_interval=0.1,
            progress_callback=self.progress_callback
        )

        results: List[ConversionResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            batch: List[str] = []
            while self.is_running:
                try:
                    bix_path, size = file_queue.get(timeout=0.2)
                    batch.append(bix_path)
                    if len(batch) >= self.batch_size:
                        r = self.process_file_batch(batch, input_path, output_path, failed_folder,
                                                    session_id=session_id, executor=executor)
                        results.extend(r)
                        self.progress_tracker.total_files += len(batch)
                        batch.clear()
                except queue.Empty:
                    if scanning_done.is_set():
                        if batch:
                            r = self.process_file_batch(batch, input_path, output_path, failed_folder,
                                                        session_id=session_id, executor=executor)
                            results.extend(r)
                            self.progress_tracker.total_files += len(batch)
                            batch.clear()
                        break

        # Finalize stats
        self.processed_files = self.progress_tracker.processed_files
        self.failed_files = self.progress_tracker.failed_files
        return results
    
    def stop_conversion(self):
        """Stop the conversion process"""
        self.is_running = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversion statistics"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Use progress tracker for speed if available
        if self.progress_tracker:
            speed = self.progress_tracker.get_estimated_speed()
        else:
            speed = (self.processed_files / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        return {
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'elapsed_time': elapsed_time,
            'speed_files_per_min': speed,
            'success_rate': (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0,
            'memory_usage_mb': self.get_memory_usage() / (1024 * 1024),
            'bytes_processed': self.bytes_processed
        }


class ConversionDatabase:
    """Database for tracking conversion history and statistics with batch operations"""
    
    def __init__(self, db_path: str, batch_size: int = 1000):
        self.db_path = db_path
        self.batch_size = batch_size
        self.pending_conversions = []
        self.pending_failures = []
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
            except Exception:
                pass
            # Conversions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT UNIQUE,
                    bix_path TEXT,
                    eml_path TEXT,
                    file_size INTEGER,
                    conversion_time REAL,
                    converted_at TIMESTAMP,
                    success BOOLEAN,
                    session_id TEXT
                )
            ''')
            
            # Failed files table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS failed_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bix_path TEXT,
                    error_message TEXT,
                    failed_at TIMESTAMP,
                    file_size INTEGER,
                    file_hash TEXT,
                    session_id TEXT
                )
            ''')
            
            # Sessions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    total_files INTEGER,
                    successful_files INTEGER,
                    failed_files INTEGER,
                    total_size_mb REAL,
                    processing_time REAL,
                    avg_speed_files_per_min REAL
                )
            ''')
            
            conn.commit()

    def update_session_progress(self, session_id: str, *, total_files: int, successful: int,
                                failed: int, total_size_mb: float, processing_time: float,
                                avg_speed_files_per_min: float) -> None:
        """Update an ongoing session row with live stats so readers (Overview) can reflect progress.

        Does not set ended_at; end_session() will finalize the row.
        """
        import sqlite3
        try:
            with sqlite3.connect(self.db_path, timeout=2.0) as conn:
                conn.execute('''
                    UPDATE sessions
                    SET total_files = ?, successful_files = ?, failed_files = ?,
                        total_size_mb = ?, processing_time = ?,
                        avg_speed_files_per_min = ?
                    WHERE id = ?
                ''', (int(total_files), int(successful), int(failed), float(total_size_mb),
                      float(processing_time), float(avg_speed_files_per_min), session_id))
                conn.commit()
        except Exception:
            # Swallow transient DB errors; UI will retry later
            pass
    
    def log_conversion_batch(self, results: List[ConversionResult], session_id: str):
        """Log a batch of conversion results"""
        with self.lock:
            for result in results:
                if result.success:
                    self.pending_conversions.append(result)
                else:
                    self.pending_failures.append(result)
            
            # Flush to database if batch size reached
            if (len(self.pending_conversions) >= self.batch_size or 
                len(self.pending_failures) >= self.batch_size):
                self.flush_pending_conversions(session_id)
    
    def flush_pending_conversions(self, session_id: str):
        """Flush pending conversions to database"""
        import sqlite3
        
        if not self.pending_conversions and not self.pending_failures:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert successful conversions
            if self.pending_conversions:
                conn.executemany('''
                    INSERT OR REPLACE INTO conversions 
                    (file_hash, bix_path, eml_path, file_size, conversion_time, 
                     converted_at, success, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                ''', [(r.file_hash, r.source_path, r.target_path, 
                       r.file_size, r.conversion_time, r.timestamp, session_id) 
                      for r in self.pending_conversions])
                self.pending_conversions.clear()
            
            # Insert failed conversions
            if self.pending_failures:
                conn.executemany('''
                    INSERT INTO failed_files 
                    (bix_path, error_message, failed_at, file_size, file_hash, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', [(r.source_path, r.error_message, r.timestamp, 
                       r.file_size, r.file_hash, session_id) 
                      for r in self.pending_failures])
                self.pending_failures.clear()
            
            conn.commit()
    
    def start_session(self, session_id: str):
        """Start a new conversion session"""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO sessions (id, started_at)
                VALUES (?, ?)
            ''', (session_id, datetime.now()))
            conn.commit()
    
    def end_session(self, session_id: str, total_files: int, successful: int, failed: int, 
                   total_size_mb: float, processing_time: float):
        """End a conversion session with statistics"""
        import sqlite3
        
        # Flush any pending conversions first
        self.flush_pending_conversions(session_id)
        
        avg_speed = (successful / processing_time) * 60 if processing_time > 0 else 0
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE sessions 
                SET ended_at = ?, total_files = ?, successful_files = ?, 
                    failed_files = ?, total_size_mb = ?, processing_time = ?, 
                    avg_speed_files_per_min = ?
                WHERE id = ?
            ''', (datetime.now(), total_files, successful, failed, total_size_mb, 
                   processing_time, avg_speed, session_id))
            conn.commit()
    
    def is_already_converted(self, file_path: str, file_hash: str) -> bool:
        """Check if file is already converted"""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT eml_path FROM conversions WHERE file_hash = ? AND success = 1',
                (file_hash,)
            )
            result = cursor.fetchone()
            if result:
                eml_path = result[0]
                return os.path.exists(eml_path)
        return False
