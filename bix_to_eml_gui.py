import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
import time
import queue
from typing import Optional
import sys
from datetime import datetime
import logging
from pathlib import Path

from bix_conversion_engine_gui import OptimizedBixConversionEngine, ConversionResult, ConversionDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedEmlConverterGUI:
    """Modern single-screen ttk GUI with Sun Valley dark theme, network-safe options, and throttled updates."""

    def __init__(self, root):
        self.root = root
        self.root.title("EML Converter GFI")
        self.root.geometry("1280x860")
        self.root.resizable(True, True)
        self.root.minsize(1100, 760)

        self.colors = {
            'bg': '#f3f4f6',
            'surface': '#ffffff',
            'primary': '#2563eb',
            'success': '#10b981',
            'error': '#ef4444',
            'warning': '#f59e0b',
            'text': '#111827',
            'text_secondary': '#6b7280',
            'border': '#e5e7eb',
            'accent': '#7c3aed'
        }

        # Apply Sun Valley theme if available; fallback gracefully
        self.theme_mode = tk.StringVar(value="dark")
        try:
            import sv_ttk  # type: ignore
            sv_ttk.set_theme('dark')
        except Exception:
            # Fallback to clam and light custom styles
            try:
                style = ttk.Style()
                style.theme_use('clam')
                style.configure('TFrame', background=self.colors['surface'])
                style.configure('TLabel', background=self.colors['surface'], foreground=self.colors['text'])
                style.configure('Heading.TLabel', font=('Segoe UI', 16, 'bold'))
                style.configure('Subheading.TLabel', font=('Segoe UI', 12))
                style.configure('Accent.TButton', background=self.colors['primary'], foreground='white')
                style.configure('Success.TButton', background=self.colors['success'], foreground='white')
                style.configure('Warn.TButton', background=self.colors['warning'], foreground='white')
                style.configure('Error.TButton', background=self.colors['error'], foreground='white')
                style.configure('Info.TLabel', foreground=self.colors['text_secondary'])
            except Exception:
                pass

        db_path = self._resolve_db_path()
        self.db = ConversionDatabase(db_path, batch_size=1000)

        self.engine = OptimizedBixConversionEngine(
            max_workers=4,
            batch_size=10000,
            progress_callback=self.on_conversion_progress,
            memory_limit_gb=2.0,
            database=self.db,
            duplication_strategy="output_exists",
            throttle_bytes_per_sec=None,
            chunk_size=512 * 1024,
            fsync_on_write=False,
            retry_attempts=2,
            retry_backoff_base=0.5
        )

        # State
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.is_running = False
        self.progress_queue = queue.Queue()
        self.session_id = None
        self.session_start_time = None
        self.last_ui_update = time.time()
        self.ui_update_interval = 0.2
        self.last_session_db_update = 0.0

        # Build UI (single-screen layout)
        self.setup_ui()
        self.update_ui_loop()
        self.update_statistics_display()
        # Apply initial theme-dependent widget colors
        try:
            self.apply_theme_colors()
        except Exception:
            pass

        # Attach logging handler to stream all app logs into the GUI log panel
        try:
            handler = _TkQueueHandler(self.progress_queue)
            handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(handler)  # root logger
            logging.getLogger(__name__).addHandler(handler)  # this module
        except Exception:
            pass

    def setup_ui(self):
        root_frame = ttk.Frame(self.root)
        root_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # Header
        self.create_header(root_frame)

        # Top: IO + Start/Stop and Settings (two columns)
        top = ttk.Frame(root_frame)
        top.pack(fill=tk.X, pady=(8, 8))

        left = ttk.Frame(top)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        right = ttk.Frame(top)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(8, 0))

        self.create_io_section(left)
        self.create_control_section(left)

        self.create_settings_panel(right)

        # Middle: progress
        self.create_progress_section(root_frame)

        # Bottom: two columns - Overview/Session Stats and Log
        bottom = ttk.Frame(root_frame)
        bottom.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        stats_col = ttk.Frame(bottom)
        stats_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        log_col = ttk.Frame(bottom)
        log_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        # Overview (DB history)
        ov_box = ttk.LabelFrame(stats_col, text="Overview")
        ov_box.pack(fill=tk.X, pady=(0, 8))
        self.overview_text = tk.Text(ov_box, height=6, font=('Consolas', 10), wrap=tk.WORD,
                                     relief='solid', bd=1)
        self.overview_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Current Session Statistics
        stats_box = ttk.LabelFrame(stats_col, text="Current Session Statistics")
        stats_box.pack(fill=tk.BOTH, expand=True)
        self.stats_text = tk.Text(stats_box, height=18, font=('Consolas', 10), wrap=tk.WORD,
                                  relief='solid', bd=1)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Log
        log_box = ttk.LabelFrame(log_col, text="Conversion Log")
        log_box.pack(fill=tk.BOTH, expand=True)
        self.create_log_panel(log_box)

    def create_header(self, parent):
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 12))

        title = ttk.Label(header, text="BIX to EML Converter",
                          font=('Segoe UI', 18, 'bold'), foreground=self.colors['primary'])
        title.pack(anchor='w')
        self.session_label = ttk.Label(header, text="Session: Ready", style='Info.TLabel')
        self.session_label.pack(anchor='w', pady=(2, 4))
        subtitle = ttk.Label(header,
                             text="Batch conversion with throttled UI and network-safe I/O",
                             style='Info.TLabel')
        subtitle.pack(anchor='w')

    def create_io_section(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(8, 8), padx=4)

        ttk.Label(frame, text="Input Directory:", font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=(2, 6))
        self.input_entry = ttk.Entry(row1, textvariable=self.input_path, font=('Segoe UI', 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(row1, text="Browse", command=self.browse_input, style='Accent.TButton').pack(side=tk.RIGHT)

        ttk.Label(frame, text="Output Directory:", font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(6, 0))
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=(2, 6))
        self.output_entry = ttk.Entry(row2, textvariable=self.output_path, font=('Segoe UI', 11))
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(row2, text="Browse", command=self.browse_output, style='Accent.TButton').pack(side=tk.RIGHT)

    def create_control_section(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(4, 8), padx=4)
        row = ttk.Frame(frame)
        row.pack(pady=6)
        self.start_button = ttk.Button(row, text="Start Conversion", command=self.start_conversion, style='Success.TButton')
        self.start_button.pack(side=tk.LEFT, padx=8)
        self.stop_button = ttk.Button(row, text="Stop", command=self.stop_conversion, style='Error.TButton')
        self.stop_button.pack(side=tk.LEFT, padx=8)
        self.stop_button.config(state=tk.DISABLED)

    def create_progress_section(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(4, 8), padx=4)
        # Scanning progress
        self.scan_status_label = ttk.Label(frame, text="Scanning: Idle", style='Info.TLabel')
        self.scan_status_label.pack(anchor='w', pady=(0, 4))
        self.scan_bar = ttk.Progressbar(frame, orient='horizontal', mode='indeterminate', length=900)
        self.scan_bar.pack(fill=tk.X)
        # Conversion progress
        self.progress_var = tk.DoubleVar(value=0)
        self.status_label = ttk.Label(frame, text="Ready", font=('Segoe UI', 12, 'bold'))
        self.status_label.pack(anchor='w', pady=(6, 6))
        self.progress_bar = ttk.Progressbar(frame, orient='horizontal', mode='determinate', length=900)
        self.progress_bar.config(variable=self.progress_var, maximum=100.0)
        self.progress_bar.pack(fill=tk.X)
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=(8, 0))
        self.progress_stats_label = ttk.Label(row, text="Files: 0/0 | Failed: 0", style='Info.TLabel')
        self.progress_stats_label.pack(side=tk.LEFT)
        self.speed_label = ttk.Label(row, text="Speed: 0 files/min", style='Info.TLabel')
        self.speed_label.pack(side=tk.LEFT, padx=(16, 0))
        ttk.Label(row, text="ETA:").pack(side=tk.LEFT, padx=(16, 4))
        self.eta_label = ttk.Label(row, text="--:--:--", style='Info.TLabel')
        self.eta_label.pack(side=tk.LEFT)

    def create_settings_panel(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        head_row = ttk.Frame(container)
        head_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(head_row, text="Settings", style='Heading.TLabel').pack(side=tk.LEFT)

        # Thread Count
        f1 = ttk.Frame(container)
        f1.pack(fill=tk.X, pady=4)
        ttk.Label(f1, text="Thread Count:").pack(side=tk.LEFT)
        self.thread_var = tk.IntVar(value=4)
        self._spinbox(f1, from_=1, to=8, increment=1, textvariable=self.thread_var, width=8).pack(side=tk.RIGHT)

        # Batch Size
        f2 = ttk.Frame(container)
        f2.pack(fill=tk.X, pady=4)
        ttk.Label(f2, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_var = tk.IntVar(value=10000)
        self._spinbox(f2, from_=1000, to=50000, increment=1000, textvariable=self.batch_var, width=8).pack(side=tk.RIGHT)

        # Memory Limit
        f3 = ttk.Frame(container)
        f3.pack(fill=tk.X, pady=4)
        ttk.Label(f3, text="Memory Limit (GB):").pack(side=tk.LEFT)
        self.memory_var = tk.DoubleVar(value=2.0)
        self._spinbox(f3, from_=0.5, to=8.0, increment=0.5, textvariable=self.memory_var, width=8).pack(side=tk.RIGHT)

        # Network Mode
        ttk.Label(container, text="Network Mode (10–50 Mbit/s)", style='Subheading.TLabel').pack(anchor='w', pady=(10, 4))
        f4 = ttk.Frame(container)
        f4.pack(fill=tk.X, pady=2)
        self.net_mode = tk.BooleanVar(value=True)
        ttk.Checkbutton(f4, text="Enable network-safe throttling", variable=self.net_mode).pack(side=tk.LEFT)

        f5 = ttk.Frame(container)
        f5.pack(fill=tk.X, pady=2)
        ttk.Label(f5, text="Max Bandwidth (MB/s):").pack(side=tk.LEFT)
        self.bw_var = tk.DoubleVar(value=3.0)
        self._spinbox(f5, from_=0.5, to=8.0, increment=0.5, textvariable=self.bw_var, width=8).pack(side=tk.RIGHT)

        f6 = ttk.Frame(container)
        f6.pack(fill=tk.X, pady=2)
        ttk.Label(f6, text="I/O Chunk Size (KB):").pack(side=tk.LEFT)
        self.chunk_var = tk.IntVar(value=512)
        self._spinbox(f6, from_=64, to=2048, increment=64, textvariable=self.chunk_var, width=8).pack(side=tk.RIGHT)

        f7 = ttk.Frame(container)
        f7.pack(fill=tk.X, pady=2)
        ttk.Label(f7, text="Retry Attempts:").pack(side=tk.LEFT)
        self.retry_var = tk.IntVar(value=2)
        self._spinbox(f7, from_=0, to=5, increment=1, textvariable=self.retry_var, width=8).pack(side=tk.RIGHT)

        f8 = ttk.Frame(container)
        f8.pack(fill=tk.X, pady=2)
        self.fsync_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f8, text="Flush to disk before finalize (fsync)", variable=self.fsync_var).pack(side=tk.LEFT)

    def create_stats_panel(self, parent):
        stats = ttk.Frame(parent)
        stats.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        ttk.Label(stats, text="Current Session", style='Heading.TLabel').pack(anchor='w', pady=(0, 8))
        self.stats_text = tk.Text(stats, height=20, width=60, font=('Consolas', 10),
                                  bg=self.colors['bg'], fg=self.colors['text'], relief='solid', bd=1, padx=8, pady=8)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

    def create_log_panel(self, parent):
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(header, text="Clear", command=self.clear_log).pack(side=tk.RIGHT)
        self.log_text = scrolledtext.ScrolledText(parent, height=18, wrap=tk.WORD, font=('Consolas', 10),
                                                  relief='flat', bd=0, padx=8, pady=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.log_text.tag_configure("INFO", foreground='#9ca3af')
        self.log_text.tag_configure("SUCCESS", foreground='#10b981')
        self.log_text.tag_configure("ERROR", foreground='#ef4444')
        self.log_text.tag_configure("WARNING", foreground='#f59e0b')

    def browse_input(self):
        path = filedialog.askdirectory(title="Select directory containing BIX files")
        if path:
            self.input_path.set(path)
            self.status_label.config(text="Input selected. Ready to scan on start…")
            self.log_message(f"Input set to: {path}", "INFO")

    def browse_output(self):
        path = filedialog.askdirectory(title="Select output directory for EML files")
        if path:
            self.output_path.set(path)

    def scan_files(self):
        ipath = self.input_path.get().strip()
        if not ipath or not os.path.exists(ipath):
            return
        try:
            self.log_message("Scanning directory for BIX files...", "INFO")
            self.root.update_idletasks()
            file_count = 0
            total_size = 0
            sample_size = 1000
            for i, bix_file in enumerate(Path(ipath).rglob("*.bix")):
                if i < sample_size:
                    try:
                        total_size += bix_file.stat().st_size
                    except Exception:
                        pass
                file_count += 1
                if i % 10000 == 0 and i > 0:
                    self.log_message(f"Scanned {i} files...", "INFO")
                    self.root.update_idletasks()
            if file_count > 0:
                avg = total_size / min(sample_size, file_count)
                est = avg * file_count
                self.log_message(f"Found approximately {file_count:,} files (~{est / (1024*1024*1024):.1f} GB)", "INFO")
        except Exception as e:
            self.log_message(f"Error scanning files: {str(e)}", "ERROR")

    def start_conversion(self):
        input_path = self.input_path.get().strip()
        output_path = self.output_path.get().strip()
        if not input_path or not output_path:
            messagebox.showerror("Error", "Please select input and output directories")
            return
        if not os.path.exists(input_path):
            messagebox.showerror("Error", "Input directory does not exist")
            return

        failed_folder = os.path.join(output_path, "failed_files")
        os.makedirs(failed_folder, exist_ok=True)

        # Apply settings
        self.engine.max_workers = int(self.thread_var.get())
        self.engine.batch_size = int(self.batch_var.get())
        self.engine.memory_limit_bytes = float(self.memory_var.get()) * 1024 * 1024 * 1024
        self.engine.db = self.db
        if self.net_mode.get():
            bw = int(self.bw_var.get() * 1024 * 1024)
            self.engine._limiter.set_rate(bw)
            self.engine.throttle_bytes_per_sec = bw
            self.engine.chunk_size = max(64 * 1024, int(self.chunk_var.get() * 1024))
            self.engine.fsync_on_write = bool(self.fsync_var.get())
            self.engine.retry_attempts = int(self.retry_var.get())
        else:
            self.engine._limiter.set_rate(None)
            self.engine.throttle_bytes_per_sec = None
            self.engine.chunk_size = max(64 * 1024, 1024 * 1024)
            self.engine.fsync_on_write = False
            self.engine.retry_attempts = int(self.retry_var.get())

        self.is_running = True
        self.session_start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            self.db.start_session(self.session_id)
        except Exception:
            import random
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{random.randint(1000, 9999)}"
            self.db.start_session(self.session_id)

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        # Switch progress bar to indeterminate during initial scanning
        try:
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start(20)
        except Exception:
            pass
        self.status_label.config(text="Scanning for files…")
        self.clear_log()
        self.log_message(f"Starting conversion with {self.engine.max_workers} threads, batch {self.engine.batch_size}", "INFO")
        if self.net_mode.get():
            self.log_message(f"Network mode: {self.bw_var.get():.1f} MB/s, chunk {self.engine.chunk_size//1024} KB, fsync={self.fsync_var.get()}", "INFO")
        self.log_message(f"Session ID: {self.session_id}", "INFO")
        self.session_label.config(text=f"Session: {self.session_id}")

        def run_worker():
            try:
                _ = self.engine.convert_directory(input_path, output_path, failed_folder, session_id=self.session_id)
                stats = self.engine.get_statistics()
                total_size_mb = stats.get('bytes_processed', 0) / (1024 ** 2)
                self.db.end_session(self.session_id, stats['total_files'], stats['processed_files'],
                                    stats['failed_files'], total_size_mb, stats['elapsed_time'])
                if self.is_running:
                    sr = stats['success_rate']
                    summary = f"Complete! {stats['processed_files']:,}/{stats['total_files']:,} successful ({sr:.1f}%)"
                    if stats['failed_files'] > 0:
                        summary += f", {stats['failed_files']:,} failed"
                    self.progress_queue.put(("complete", summary))
                else:
                    self.progress_queue.put(("stopped", "Conversion stopped"))
            except Exception as e:
                self.progress_queue.put(("error", f"Critical error: {str(e)}"))

        threading.Thread(target=run_worker, daemon=True).start()
        self.update_ui_loop()

    def stop_conversion(self):
        self.is_running = False
        self.engine.stop_conversion()
        self.status_label.config(text="Stopped")
        self.log_message("Conversion stopped by user", "WARNING")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def on_conversion_progress(self, *args):
        self.progress_queue.put(("progress", args))

    def update_ui_loop(self):
        try:
            progress_updates = []
            scan_updates = []
            while True:
                try:
                    msg_type, message = self.progress_queue.get_nowait()
                    if msg_type == "success":
                        self.log_message(message, "SUCCESS")
                    elif msg_type == "error":
                        self.log_message(message, "ERROR")
                    elif msg_type == "log":
                        level, text = message
                        self.log_message(text, level.upper())
                    elif msg_type == "progress":
                        # Detect scan sentinel patterns
                        if (len(message) == 1 and isinstance(message[0], tuple) and message[0] and message[0][0] == "scan"):
                            scan_updates.append(message[0])
                        elif (len(message) >= 1 and isinstance(message[0], str) and message[0] == "scan"):
                            scan_updates.append(tuple(message[:3]))
                        else:
                            progress_updates.append(message)
                    elif msg_type == "complete":
                        self.status_label.config(text=message)
                        self.log_message(message, "SUCCESS")
                        self.is_running = False
                        self.start_button.config(state=tk.NORMAL)
                        self.stop_button.config(state=tk.DISABLED)
                        self.update_statistics_display()
                        # Stop scanning bar
                        try:
                            self.scan_bar.stop()
                            self.scan_status_label.config(text="Scanning: Complete")
                        except Exception:
                            pass
                        messagebox.showinfo("Complete", message)
                    elif msg_type == "stopped":
                        self.status_label.config(text=message)
                        self.is_running = False
                        self.start_button.config(state=tk.NORMAL)
                        self.stop_button.config(state=tk.DISABLED)
                        try:
                            self.scan_bar.stop()
                            self.scan_status_label.config(text="Scanning: Stopped")
                        except Exception:
                            pass
                except queue.Empty:
                    break

            # Update scanning UI (found files and estimated size)
            if scan_updates:
                last_scan = scan_updates[-1]
                try:
                    # last_scan expected form: ("scan", files, bytes)
                    _, scan_files, scan_bytes = (last_scan + (0, 0))[:3]
                except Exception:
                    scan_files, scan_bytes = 0, 0
                try:
                    gb = (scan_bytes/(1024*1024*1024)) if scan_bytes else 0.0
                    self.scan_status_label.config(text=f"Scanning: Found {int(scan_files):,} files | ~{gb:.2f} GB")
                    if str(self.scan_bar.cget('mode')) != 'indeterminate':
                        self.scan_bar.config(mode='indeterminate')
                    # Ensure indicator animates
                    self.scan_bar.start(20)
                except Exception:
                    pass

            if progress_updates:
                current_time = time.time()
                if current_time - self.last_ui_update >= self.ui_update_interval:
                    data = progress_updates[-1]
                    if len(data) >= 5:
                        processed, failed, total, p_bytes, t_bytes = data[:5]
                    else:
                        processed, failed, total = data[:3]
                        p_bytes, t_bytes = None, None
                    self.update_statistics(processed, failed, total, p_bytes, t_bytes)
                    self.last_ui_update = current_time

                    # Periodically persist live session stats to DB so Overview updates during processing
                    try:
                        if self.is_running and self.session_id:
                            if current_time - (self.last_session_db_update or 0) >= 2.0:
                                elapsed = (time.time() - self.session_start_time) if self.session_start_time else 0.0
                                used_bytes = int(p_bytes or 0)
                                total_size_mb = (used_bytes / (1024 * 1024)) if used_bytes else 0.0
                                avg_speed = ((processed / elapsed) * 60.0) if elapsed > 0 else 0.0
                                total_files = int(total or 0)
                                successful = int(processed or 0)
                                failed_files = int(failed or 0)
                                try:
                                    self.db.update_session_progress(
                                        self.session_id,
                                        total_files=total_files,
                                        successful=successful,
                                        failed=failed_files,
                                        total_size_mb=total_size_mb,
                                        processing_time=elapsed,
                                        avg_speed_files_per_min=avg_speed,
                                    )
                                except Exception:
                                    pass
                                self.last_session_db_update = current_time
                    except Exception:
                        pass

            # Periodically refresh overview from DB
            now = time.time()
            if getattr(self, 'overview_last_update', 0) == 0 or now - getattr(self, 'overview_last_update', 0) >= 2.0:
                self.update_statistics_display()
                self.overview_last_update = now

            self.root.after(50, self.update_ui_loop)
        except Exception as e:
            self.log_message(f"UI error: {str(e)}", "ERROR")
            self.root.after(100, self.update_ui_loop)

    def update_statistics(self, processed: int, failed: int, total: int,
                              processed_bytes: Optional[int] = None,
                              total_bytes: Optional[int] = None):
            self.root.after(0, lambda: self._update_statistics_ui(processed, failed, total, processed_bytes, total_bytes))
    
    def _update_statistics_ui(self, processed: int, failed: int, total: int,
                                  processed_bytes: Optional[int], total_bytes: Optional[int]):
            try:
                used_bytes = processed_bytes if processed_bytes is not None else 0
                total_b = total_bytes if total_bytes is not None else 0
                if total_b:
                    progress = (used_bytes / total_b) * 100 if total_b > 0 else 0
                    # Switch progress bar back to determinate when conversion begins
                    try:
                        if str(self.progress_bar.cget('mode')) != 'determinate':
                            self.progress_bar.stop()
                            self.progress_bar.config(mode='determinate')
                    except Exception:
                        pass
                    current = self.progress_var.get()
                    if abs(current - progress) > 0.1:
                        self.progress_var.set(progress)
                elif total > 0:
                    progress = (processed / total) * 100
                    # Ensure progress bar is determinate once total files are known
                    try:
                        if str(self.progress_bar.cget('mode')) != 'determinate':
                            self.progress_bar.stop()
                            self.progress_bar.config(mode='determinate')
                    except Exception:
                        pass
                    current = self.progress_var.get()
                    if abs(current - progress) > 0.1:
                        self.progress_var.set(progress)
                elapsed = time.time() - self.session_start_time if self.session_start_time else 0
                speed_files = (processed / elapsed) * 60 if elapsed > 0 else 0
                if total_b and elapsed > 0:
                    bps = used_bytes / elapsed if elapsed > 0 else 0
                    rem = max(0, total_b - used_bytes)
                    eta_seconds = (rem / bps) if bps > 0 else 0
                    h, r = divmod(int(eta_seconds), 3600)
                    m, s = divmod(r, 60)
                    eta_text = f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    # Fallback ETA based on files/min if byte totals aren't available
                    if elapsed > 0 and processed > 0 and total > processed:
                        files_per_sec = processed / elapsed
                        rem_files = max(0, total - processed)
                        eta_seconds = int(rem_files / files_per_sec) if files_per_sec > 0 else 0
                        h, r = divmod(int(eta_seconds), 3600)
                        m, s = divmod(r, 60)
                        eta_text = f"{h:02d}:{m:02d}:{s:02d}"
                    else:
                        eta_text = "--:--:--"
                stats_text = f"MB: {used_bytes/(1024*1024):.1f}/{(total_b/(1024*1024)) if total_b else 0:.1f} | Files: {processed:,}/{total:,} | Failed: {failed:,}"
                if self.progress_stats_label.cget("text") != stats_text:
                    self.progress_stats_label.config(text=stats_text)
                speed_mb = ((used_bytes/(1024*1024)) / elapsed * 60) if elapsed > 0 and used_bytes else 0
                speed_text = f"Speed: {speed_mb:.1f} MB/min | {speed_files:.0f} files/min"
                if self.speed_label.cget("text") != speed_text:
                    self.speed_label.config(text=speed_text)
                if self.eta_label.cget("text") != eta_text:
                    self.eta_label.config(text=eta_text)
                # Always update current session statistics (content diff prevents flicker)
                self.update_detailed_statistics(processed, failed, total, elapsed, speed_files)
            except Exception:
                pass
    
    def update_detailed_statistics(self, processed: int, failed: int, total: int, elapsed: float, speed: float):
        try:
            mem_mb = self.engine.get_memory_usage() / (1024 * 1024)
            bytes_done = getattr(self.engine, 'bytes_processed', 0)
            data_mb = bytes_done / (1024 * 1024)
            data_gb = data_mb / 1024
            thr_mb_per_min = (data_mb / elapsed * 60) if elapsed > 0 else 0.0
            stats_text = f"""Session: {self.session_id}
Total Files: {total:,}
Processed: {processed:,}
Failed: {failed:,}
Success Rate: {(processed/total*100) if total > 0 else 0:.1f}%

Elapsed Time: {elapsed:.1f}s
Current Speed: {speed:.0f} files/min
Throughput: {thr_mb_per_min:.1f} MB/min
Data Processed: {data_gb:.2f} GB
Active Threads: {self.engine.max_workers}
Batch Size: {self.engine.batch_size}

Memory Usage: {mem_mb:.1f} MB
Files Remaining: {total - processed:,}
"""
            current = self.stats_text.get(1.0, tk.END).strip()
            if current != stats_text.strip():
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, stats_text)
        except Exception:
            pass

    def update_statistics_display(self):
        """Render overall stats from the conversion database into the Overview panel."""
        try:
            import sqlite3
            db_path = getattr(self.db, 'db_path', self._resolve_db_path())
            if not os.path.exists(db_path):
                return
            with sqlite3.connect(db_path, timeout=2.0) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT 
                        COUNT(*),
                        COALESCE(SUM(total_files),0),
                        COALESCE(SUM(successful_files),0),
                        COALESCE(SUM(failed_files),0),
                        COALESCE(AVG(avg_speed_files_per_min),0),
                        COALESCE(SUM(total_size_mb),0),
                        COALESCE(SUM(processing_time),0)
                    FROM sessions
                    """
                )
                row = cur.fetchone() or (0, 0, 0, 0, 0, 0, 0)
                sessions, total, ok, fail, avg_speed_files, sum_size_mb, sum_time = row
                cur.execute(
                    """
                    SELECT id, started_at, ended_at, total_files, successful_files, failed_files, processing_time, total_size_mb
                    FROM sessions
                    ORDER BY started_at DESC LIMIT 1
                    """
                )
                last = cur.fetchone()
            overview_lines = []
            overview_lines.append(f"Sessions: {sessions}")
            overview_lines.append(f"Total Files: {int(total):,}")
            overview_lines.append(f"Successful: {int(ok):,} | Failed: {int(fail):,}")
            if total:
                sr = (ok / total) * 100.0 if total > 0 else 0.0
                overview_lines.append(f"Overall Success Rate: {sr:.1f}%")
            overview_lines.append(f"Average Speed: {avg_speed_files:.0f} files/min")
            # Overall data metrics
            total_gb = (sum_size_mb / 1024.0)
            overview_lines.append(f"Total Data: {total_gb:.2f} GB")
            if sum_time and sum_time > 0:
                overall_thr = (sum_size_mb / sum_time) * 60.0
                overview_lines.append(f"Overall Throughput: {overall_thr:.1f} MB/min")
            if last:
                lid, started, ended, ltotal, lok, lfail, ltime, lsize_mb = last
                overview_lines.append("")
                overview_lines.append(f"Last Session: {lid}")
                overview_lines.append(f"  Files: {lok:,}/{ltotal:,} | Failed: {lfail:,}")
                overview_lines.append(f"  Duration: {ltime:.1f}s")
                overview_lines.append(f"  Data: {lsize_mb/1024.0:.2f} GB")
                if ltime and ltime > 0:
                    lthr = (lsize_mb / ltime) * 60.0
                    overview_lines.append(f"  Throughput: {lthr:.1f} MB/min")
            text = "\n".join(overview_lines)
            current = self.overview_text.get(1.0, tk.END).strip()
            if current != text.strip():
                self.overview_text.delete(1.0, tk.END)
                self.overview_text.insert(tk.END, text)
        except Exception as e:
            # Don't spam logs if DB temporarily busy
            pass

    def log_message(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {level}: {message}\n"
        self.root.after(1, lambda: self._update_log_text(formatted, level))

    def _update_log_text(self, formatted_message: str, level: str):
        try:
            line_count = int(self.log_text.index('end-1c').split('.')[0])
            if line_count > 1000:
                self.log_text.delete('1.0', '100.0')
            self.log_text.insert(tk.END, formatted_message, level)
            self.log_text.see(tk.END)
            if line_count % 10 == 0:
                self.root.update_idletasks()
        except Exception:
            pass

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def _resolve_db_path(self) -> str:
        """Resolve DB path next to the executable (or script); fallback to user profile if not writable."""
        try:
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))

            db_path = os.path.join(base_dir, 'conversion_ledger.db')
            # Test writability by creating a small temp file
            try:
                test_path = os.path.join(base_dir, '.db_write_test.tmp')
                with open(test_path, 'w') as tf:
                    tf.write('ok')
                os.remove(test_path)
                return db_path
            except Exception:
                pass

            # Fallback: user data directory
            if os.name == 'nt':
                base = os.environ.get('LOCALAPPDATA') or os.path.expanduser(r'~\AppData\Local')
                appdir = os.path.join(base, 'EMLConverter')
            else:
                base = os.environ.get('XDG_DATA_HOME') or os.path.expanduser('~/.local/share')
                appdir = os.path.join(base, 'eml_converter')
            os.makedirs(appdir, exist_ok=True)
            return os.path.join(appdir, 'conversion_ledger.db')
        except Exception:
            return os.path.join(os.path.dirname(__file__), 'conversion_ledger.db')

    def apply_theme_colors(self):
        """Adjust Text/Log backgrounds to match light/dark theme."""
        mode = (self.theme_mode.get() or "dark").lower()
        if mode == 'dark':
            log_bg, log_fg = '#111827', '#f3f4f6'
            sel_bg, ins_fg = '#374151', '#f3f4f6'
            text_bg, text_fg = '#0f172a', '#e5e7eb'
        else:
            log_bg, log_fg = '#ffffff', '#111827'
            sel_bg, ins_fg = '#e5e7eb', '#111827'
            text_bg, text_fg = '#ffffff', '#111827'
        try:
            self.log_text.configure(bg=log_bg, fg=log_fg, selectbackground=sel_bg, insertbackground=ins_fg)
        except Exception:
            pass
        for t in (getattr(self, 'stats_text', None), getattr(self, 'overview_text', None)):
            if t is not None:
                try:
                    t.configure(bg=text_bg, fg=text_fg)
                except Exception:
                    pass

    def _spinbox(self, parent, from_, to, increment, textvariable, width):
        """Create a (ttk) Spinbox if available, else fall back to tk.Spinbox."""
        try:
            sb = ttk.Spinbox(parent, from_=from_, to=to, increment=increment, textvariable=textvariable, width=width)
        except Exception:
            sb = tk.Spinbox(parent, from_=from_, to=to, increment=increment, textvariable=textvariable, width=width,
                            font=('Segoe UI', 11), relief='solid', bd=1)
        return sb


class _TkQueueHandler(logging.Handler):
    """Logging handler that forwards records to the GUI's queue."""
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level = record.levelname or "INFO"
            self.q.put(("log", (level, msg)))
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = OptimizedEmlConverterGUI(root)
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.mainloop()


if __name__ == "__main__":
    main()

