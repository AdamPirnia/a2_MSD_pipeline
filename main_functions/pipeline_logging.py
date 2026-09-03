"""Per-step run logging for the generated Module 1 (Coordinates) pipeline scripts.

``step_log()`` writes a ``mod1_stN.log`` next to the generated pipeline script.
The log mirrors everything the step prints to the terminal -- including output
from parallel worker processes, captured at the file-descriptor level -- and adds
a header with the user's configuration and a footer with wall-clock timing.
``summarize_coordinate_output()`` appends particle / frame counts read back from
the files the step produced.

Logging here is strictly best-effort: every failure path is swallowed so it can
never interrupt, slow, or alter the analysis itself.
"""
from __future__ import annotations

import contextlib
import datetime
import os
import re
import sys
import threading
import time

_SAFE_INDEX_EXPR = re.compile(r"^i(?:\s*[+\-*/]\s*\d+)?$")


def _expand_pattern(pattern, common_term="", file_index=None):
    """Local copy of the project's path-pattern expansion (``*`` and ``{i}``)."""
    if not pattern:
        return ""
    result = str(pattern)
    if common_term is not None:
        result = result.replace("*", str(common_term))
    if file_index is not None:
        def _replace(match):
            expr = match.group(1).strip()
            if _SAFE_INDEX_EXPR.match(expr):
                try:
                    return str(eval(expr, {"__builtins__": {}}, {"i": file_index}))
                except Exception:
                    return match.group(0)
            return match.group(0)

        result = re.sub(r"\{([^}]+)\}", _replace, result)
    return result


def _fmt_hms(seconds):
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class _FdTee:
    """Best-effort duplication of OS file descriptors 1 and 2 into a log file.

    Captures output from this process *and* any child / worker processes while
    leaving the real terminal (or SLURM ``-o`` file) output intact. If the
    platform refuses any of the descriptor juggling, ``start()`` rolls back and
    reports failure so the caller can fall back to a Python-level tee.
    """

    def __init__(self, log_handle):
        self._log = log_handle
        self.is_active = False
        self._saved_out = None
        self._saved_err = None
        self._read_fd = None
        self._pump = None

    def start(self):
        try:
            self._saved_out = os.dup(1)
            self._saved_err = os.dup(2)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, 1)
            os.dup2(write_fd, 2)
            os.close(write_fd)
            self._read_fd = read_fd
            self._pump = threading.Thread(target=self._pump_loop, daemon=True)
            self._pump.start()
            self.is_active = True
            return True
        except Exception:
            self._restore_fds()
            self.is_active = False
            return False

    def _pump_loop(self):
        pending = b""
        try:
            while True:
                try:
                    chunk = os.read(self._read_fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                try:
                    os.write(self._saved_out, chunk)
                except Exception:
                    pass
                pending += chunk
                cut = pending.rfind(b"\n")
                if cut >= 0:
                    text, pending = pending[: cut + 1], pending[cut + 1:]
                    self._write_log(text)
        finally:
            if pending:
                self._write_log(pending)

    def _write_log(self, raw):
        try:
            self._log.write(raw.decode("utf-8", "replace"))
            self._log.flush()
        except Exception:
            pass

    def _restore_fds(self):
        for saved, target in ((self._saved_out, 1), (self._saved_err, 2)):
            if saved is not None:
                try:
                    os.dup2(saved, target)
                except Exception:
                    pass

    def stop(self):
        if not self.is_active:
            return
        self.is_active = False
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        # Restoring fds 1/2 drops the last references to the pipe write end, so
        # the pump thread reads EOF and exits.
        self._restore_fds()
        if self._pump is not None:
            self._pump.join(timeout=10)
        for fd in (self._saved_out, self._saved_err, self._read_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass


class _StreamTee:
    """Fallback tee that duplicates Python-level writes to one extra stream."""

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        try:
            self._primary.write(data)
        except Exception:
            pass
        try:
            self._secondary.write(data)
            self._secondary.flush()
        except Exception:
            pass

    def flush(self):
        for stream in (self._primary, self._secondary):
            try:
                stream.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._primary, name)


def _emit(text, handle, direct_to_handle):
    print(text)
    if direct_to_handle and handle is not None:
        try:
            handle.write(text + "\n")
            handle.flush()
        except Exception:
            pass


@contextlib.contextmanager
def step_log(log_path, title, info=None):
    """Mirror a pipeline step's console output to ``log_path`` with a header/footer."""
    start = time.time()
    handle = None
    fd_tee = None
    stream_fallback = False
    saved_stdout, saved_stderr = sys.stdout, sys.stderr

    # Flush anything the caller already buffered so it lands on the real terminal
    # before fds are swapped, rather than leaking into the log after the swap.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.flush()
        except Exception:
            pass

    try:
        log_path = os.fspath(log_path)
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        handle = open(log_path, "w", encoding="utf-8", buffering=1)
    except Exception as exc:
        try:
            sys.stderr.write(f"[pipeline_logging] cannot open {log_path!r}: {exc}\n")
        except Exception:
            pass
        handle = None

    if handle is not None:
        fd_tee = _FdTee(handle)
        if not fd_tee.start():
            fd_tee = None
            try:
                sys.stdout = _StreamTee(saved_stdout, handle)
                sys.stderr = _StreamTee(saved_stderr, handle)
                stream_fallback = True
            except Exception:
                sys.stdout, sys.stderr = saved_stdout, saved_stderr
                stream_fallback = False

    # A header line is written straight to the file only when neither tee is in
    # place to carry it there.
    direct = handle is not None and fd_tee is None and not stream_fallback

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "=" * 72,
        str(title),
        "=" * 72,
        f"{'Started':<20}: {now}",
        f"{'Python':<20}: {sys.executable}",
        f"{'Working directory':<20}: {os.getcwd()}",
        f"{'Log file':<20}: {log_path}",
    ]
    for key, value in (info or {}).items():
        header.append(f"{str(key):<20}: {value}")
    if fd_tee is None and not stream_fallback and handle is not None:
        header.append(f"{'Note':<20}: file-descriptor capture unavailable; "
                      "parallel-worker output may be missing from this log")
    header.append("-" * 72)
    _emit("\n".join(header), handle, direct)

    try:
        yield
    finally:
        elapsed = time.time() - start
        footer = [
            "-" * 72,
            f"{'Wall time':<20}: {elapsed:.2f} s ({_fmt_hms(elapsed)})",
            f"{'Finished':<20}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 72,
        ]
        _emit("\n".join(footer), handle, direct)
        if fd_tee is not None:
            try:
                fd_tee.stop()
            except Exception:
                pass
        if stream_fallback:
            sys.stdout, sys.stderr = saved_stdout, saved_stderr
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass


def _array_shape(path, io_spec):
    """Return ``(rows, cols)`` for a coordinate file without loading it fully."""
    import numpy as np

    mode = None
    if isinstance(io_spec, dict):
        mode = str(io_spec.get("mode") or "").strip().lower()

    if mode == "text":
        with open(path, "r") as handle:
            first = handle.readline()
            cols = len(first.split())
            rows = (1 if first.strip() else 0) + sum(1 for line in handle if line.strip())
        return rows, cols

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim == 1:
        return 1, int(array.shape[0])
    return int(array.shape[0]), int(array.shape[1])


def summarize_coordinate_output(label, path_pattern, indices, io_spec=None,
                                column_meaning="atoms"):
    """Print frame / particle counts for the first readable file matching a pattern.

    ``path_pattern`` is a full path (base directory already joined, common terms
    already expanded) that still contains a ``{i}`` placeholder. ``column_meaning``
    names what one xyz triplet represents for this step, e.g. ``"atoms"`` for raw
    or unwrapped coordinates, ``"residues/groups"`` for a centre-of-mass output.
    Best-effort and read-only.
    """
    try:
        index_list = list(indices) if indices is not None else [0]
    except TypeError:
        index_list = [indices]
    if not index_list:
        index_list = [0]

    for idx in index_list[:5]:
        try:
            path = _expand_pattern(path_pattern, "", idx)
            if not path or not os.path.isfile(path):
                continue
            rows, cols = _array_shape(path, io_spec)
            particles = cols // 3 if cols else 0
            remainder = "" if cols % 3 == 0 else f" ({cols % 3} trailing column(s) ignored)"
            print(
                f"{label}: {os.path.basename(path)} -> {rows} frame(s), "
                f"{cols} value(s)/frame{remainder}; ~{particles} {column_meaning} per frame"
            )
            return
        except Exception:
            continue
    print(f"{label}: (no output file available yet for a particle-count summary)")
