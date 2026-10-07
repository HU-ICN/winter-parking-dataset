"""Single-instance lock for a formal annotation output directory (tools 1.0.8+).

Atomic and process-bound: the lock is an advisory exclusive `fcntl.flock` on an open file
descriptor of `<out_dir>/.session.lock`. The descriptor is held for the whole process lifetime
(construction, startup checks, main loop); the kernel releases it on any exit, including a crash
or SIGKILL. The PID/tool/start time written into the file are diagnostics only and are never used
to decide liveness. The lock file is never unlinked (unlinking while another process has it open
would allow two locks on different inodes).
"""
import atexit, fcntl, json, os, sys
from datetime import datetime
from pathlib import Path


class OutputLock:
    def __init__(self, out_dir, tool_name):
        self.path = Path(out_dir) / ".session.lock"; self.tool_name = tool_name; self.fd = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError, OSError):
            holder = ""
            try:
                holder = os.read(fd, 4096).decode(errors="replace").strip()
            except OSError:
                pass
            os.close(fd)
            raise SystemExit(f"another process holds {self.path}{' (' + holder + ')' if holder else ''}; close it first")
        self.fd = fd
        os.ftruncate(fd, 0); os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps(dict(pid=os.getpid(), tool=self.tool_name, started=datetime.now().astimezone().isoformat(timespec="seconds"))).encode()); os.fsync(fd)
        atexit.register(self.release)
        return self

    def release(self):
        if self.fd is None:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd); self.fd = None


def acquire_output_lock(out_dir, tool_name):
    return OutputLock(out_dir, tool_name).acquire()


def release_output_lock(lock):
    if lock is not None:
        lock.release()


if __name__ == "__main__":  # helper for the isolated tests: hold the lock until killed (SIGTERM), independent of stdin
    import signal, time
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # SIGTERM = clean exit through release(); SIGKILL tests the kernel path
    lock = acquire_output_lock(sys.argv[1], "holder"); print("ACQUIRED", os.getpid(), flush=True)
    while True:
        time.sleep(3600)
