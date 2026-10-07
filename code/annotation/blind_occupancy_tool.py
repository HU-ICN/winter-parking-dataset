#!/usr/bin/env python3
"""Independent per-stall occupancy annotation (protocol section 8; label set INDEPENDENT-OCC-120).

Rules enforced by the interface:
- Shows the image and the 82 registered stall polygons (they define the labelling units) with a
  per-frame overlay offset [dx, dy] that the annotator aligns with the arrow keys.
- NEVER shows detector boxes, predictions, box-derived states or any prior label.
- Every stall has a state: empty (default), occupied, unresolved. Click inside a stall to cycle
  empty -> occupied -> unresolved -> empty. Unresolved is kept for adjudication and is never
  mapped to empty.
- A frame is complete only after an explicit, confirmed completion (Return, default No).
- Completion guards: 2 s dwell after load, 1 s debounce, physical key release required.
- Timing: active seconds accrue only within 60 s of an input event; wall seconds always.
- Autosave every 30 s; resuming reloads only the annotator's own saved output.

Keys: click stall = cycle state | arrows = shift overlay (Shift = 5 px) | n / p = next / previous
frame (saves draft) | g = next incomplete | u = undo last state change | Return = complete (confirm)
| f = fullscreen | 0 = fit | s = save | q = quit (saves). Esc only leaves fullscreen.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "occ-1.0.3"
IDLE_THRESHOLD_SECONDS = 60.0
AUTOSAVE_MS = 30000
OVERLAY_RGB = {"empty_outline": (127, 127, 127), "occupied": (255, 59, 48), "unresolved": (255, 214, 10)}
STATES = ("empty", "occupied", "unresolved")
COLORS = {"empty": "#7f7f7f", "occupied": "#ff3b30", "unresolved": "#ffd60a"}


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def clock_increments(last_clock: float, now: float, last_input_at: float | None, idle=IDLE_THRESHOLD_SECONDS):
    wall = max(0.0, now - last_clock)
    if last_input_at is None or wall == 0.0:
        return 0.0, wall
    active = max(0.0, min(now, last_input_at + idle) - last_clock)
    return min(active, wall), wall


def point_in_poly(x: float, y: float, poly) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        json.dump(payload, h, ensure_ascii=False, indent=2)
        h.write("\n")
    os.replace(tmp, path)
    os.chmod(path, 0o600)


class OccupancyApp:
    def __init__(self, rows, image_dir: Path, slots: list[dict], out_dir: Path, meta: dict, progress_gate: int, ack_gate: bool, sealed=None):
        import tkinter as tk
        from tkinter import messagebox
        from PIL import Image, ImageTk

        self.tk, self.messagebox, self.Image, self.ImageTk = tk, messagebox, Image, ImageTk
        self.rows, self.image_dir, self.slots, self.out_dir, self.meta = rows, image_dir, slots, out_dir, meta
        self.progress_gate, self.ack_gate = progress_gate, ack_gate
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.payload: dict = {}
        self.scale = 1.0
        self.offset_x = self.offset_y = 0.0
        self.photo = None
        self.undo_stack: list[dict] = []
        self.last_clock = time.monotonic()
        self.last_input_at: float | None = None
        self._frame_loaded_at = 0.0
        self._last_complete_at = 0.0
        self._return_armed = True

        self.root = tk.Tk()
        self.root.title("Independent per-stall occupancy annotation")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{int(sw * 0.9)}x{int((sh - 60) * 0.9)}")
        self.root.configure(bg="#151515")
        self.status = tk.Label(self.root, bg="#151515", fg="#dddddd", anchor="w", font=("Menlo", 12))
        self.status.pack(fill="x")
        tk.Label(self.root, bg="#151515", fg="#aaaaaa", anchor="w", font=("Menlo", 10),
                 text="click stall: empty->occupied->unresolved | arrows: shift overlay (Shift=5px) | n/p: next/prev | g: next incomplete | u: undo | Return: complete (confirm) | f: fullscreen | 0: fit | s: save | q: quit").pack(fill="x")
        self.canvas = tk.Canvas(self.root, bg="#090909", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.root.bind("<Key>", self.on_key)
        self.root.bind("<KeyRelease-Return>", lambda e: setattr(self, "_return_armed", True))
        self.root.bind("<KeyRelease-KP_Enter>", lambda e: setattr(self, "_return_armed", True))
        self.canvas.bind("<Configure>", lambda e: self.fit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.sealed = dict(sealed or {})  # filename -> sealed digest
        self.current_sealed = False
        self.root.after(AUTOSAVE_MS, self.autosave)
        self.load_index(self.first_incomplete())

    # ---------- persistence ----------
    def out_path(self, row) -> Path:
        return self.out_dir / f"{row['anonymous_id']}.json"

    def blank(self, row, w: int, h: int) -> dict:
        return {
            "anonymous_id": row["anonymous_id"], "image_name": row["image_name"], "image_size": [w, h],
            "offset": [0.0, 0.0], "states": {str(s["slot_id"]): "empty" for s in self.slots},
            "revision_annotation": {
                "tool": "blind_occupancy_tool", "tool_version": TOOL_VERSION, "label_set_id": self.meta["label_set_id"],
                "protocol_sha256": self.meta["protocol_sha256"], "manifest_sha256": self.meta["manifest_sha256"],
                "slots_sha256": self.meta["slots_sha256"], "image_sha256": sha256_file(self.image_dir / row["image_name"]),
                "started_at": iso_now(), "last_saved_at": None, "completed_at": None,
                "active_seconds": 0.0, "wall_seconds": 0.0, "complete": False, "frame_note": "",
            },
        }

    def load_payload(self, row, w, h) -> dict:
        p = self.out_path(row)
        if p.exists():
            with p.open(encoding="utf-8") as h_:
                d = json.load(h_)
            if d.get("image_name") != row["image_name"]:
                raise ValueError(f"output {p} belongs to another image")
            return d
        return self.blank(row, w, h)

    def is_sealed(self, row) -> bool:
        return self.out_path(row).name in self.sealed

    def _sealed_block(self) -> bool:
        if self.current_sealed:
            self.hint("sealed frame: read-only, editing disabled"); return True
        return False

    def update_clock(self, now=None):
        now = time.monotonic() if now is None else now
        if self.current_sealed:
            self.last_clock = now; return  # sealed frame: timing frozen with the seal
        m = self.payload["revision_annotation"]
        a, w = clock_increments(self.last_clock, now, self.last_input_at)
        m["active_seconds"] = round(m["active_seconds"] + a, 2)
        m["wall_seconds"] = round(m["wall_seconds"] + w, 2)
        m["last_saved_at"] = iso_now()
        self.last_clock = now

    def mark_activity(self):
        now = time.monotonic()
        self.update_clock(now)
        self.last_input_at = now

    def save(self):
        if self.payload and not self.current_sealed:
            self.update_clock()
            atomic_write(self.out_path(self.rows[self.index]), self.payload)

    def autosave(self):
        self.save()
        self.root.after(AUTOSAVE_MS, self.autosave)

    # ---------- navigation ----------
    def completed_count(self):
        n = 0
        for r in self.rows:
            p = self.out_path(r)
            if p.exists():
                with p.open(encoding="utf-8") as h_:
                    n += bool(json.load(h_)["revision_annotation"].get("complete"))
        return n

    def first_incomplete(self):
        for i, r in enumerate(self.rows):
            p = self.out_path(r)
            if not p.exists():
                return i
            with p.open(encoding="utf-8") as h_:
                if not json.load(h_)["revision_annotation"].get("complete"):
                    return i
        return len(self.rows) - 1

    def load_index(self, i):
        if self.payload:
            self.save()
        self.index = max(0, min(len(self.rows) - 1, i))
        row = self.rows[self.index]
        self.image = self.Image.open(self.image_dir / row["image_name"]).convert("RGB")
        w, h = self.image.size
        self.payload = self.load_payload(row, w, h)
        self.current_sealed = self.is_sealed(row)
        self.undo_stack = []
        self.last_clock = time.monotonic()
        self._frame_loaded_at = self.last_clock
        self.last_input_at = None
        self.fit()
        if self.current_sealed:
            self.hint("SEALED frame: read-only, nothing is saved (view only)")

    # ---------- drawing ----------
    def fit(self):
        if not hasattr(self, "image"):
            return
        self.root.update_idletasks()
        cw, ch = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200)
        w, h = self.image.size
        self.scale = min(cw / w, ch / h, 1.0)
        self.offset_x, self.offset_y = (cw - w * self.scale) / 2, (ch - h * self.scale) / 2
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        w, h = self.image.size
        rendered = self.image.resize((max(1, int(w * self.scale)), max(1, int(h * self.scale))))
        self.photo = self.ImageTk.PhotoImage(rendered)
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor="nw")
        dx, dy = self.payload["offset"]
        for s in self.slots:
            st = self.payload["states"][str(s["slot_id"])]
            pts = [c for x, y in s["polygon"] for c in (self.offset_x + (x + dx) * self.scale, self.offset_y + (y + dy) * self.scale)]
            self.canvas.create_polygon(*pts, outline=COLORS[st], fill=COLORS[st] if st != "empty" else "", stipple="gray25" if st != "empty" else "", width=2)
        m = self.payload["revision_annotation"]
        n_occ = sum(v == "occupied" for v in self.payload["states"].values())
        n_unr = sum(v == "unresolved" for v in self.payload["states"].values())
        self.status.config(text=f"[{self.index + 1}/{len(self.rows)}] {self.rows[self.index]['anonymous_id']} | occupied={n_occ} unresolved={n_unr} | offset dx={dx:+.0f} dy={dy:+.0f} | {'COMPLETE' if m['complete'] else 'DRAFT'} | completed={self.completed_count()}/{len(self.rows)}")

    # ---------- interaction ----------
    def image_point(self, cx, cy):
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    def on_click(self, event):
        self.mark_activity()
        if self._sealed_block():
            return
        x, y = self.image_point(event.x, event.y)
        dx, dy = self.payload["offset"]
        for s in self.slots:
            if point_in_poly(x, y, [(px + dx, py + dy) for px, py in s["polygon"]]):
                sid = str(s["slot_id"])
                self.undo_stack.append(dict(self.payload["states"]))
                self.undo_stack = self.undo_stack[-50:]
                cur = self.payload["states"][sid]
                self.payload["states"][sid] = STATES[(STATES.index(cur) + 1) % 3]
                self.uncomplete()
                break
        self.redraw()

    def uncomplete(self):
        m = self.payload["revision_annotation"]
        m["complete"] = False
        m["completed_at"] = None

    def shift(self, ddx, ddy):
        if self._sealed_block():
            return
        self.payload["offset"][0] += ddx
        self.payload["offset"][1] += ddy
        self.uncomplete()
        self.redraw()

    def complete_and_next(self):
        now = time.monotonic()
        if self.current_sealed:
            self.hint("sealed frame: cannot be completed again; moving on"); self.load_index(min(self.index + 1, len(self.rows) - 1)); return
        if not self._return_armed:
            return self.hint("Return ignored: release the key first")
        if now - self._frame_loaded_at < 2.0:
            return self.hint("Return ignored: frame loaded < 2 s ago")
        if now - self._last_complete_at < 1.0:
            return self.hint("Return ignored: debounce")
        self._return_armed = False
        n_occ = sum(v == "occupied" for v in self.payload["states"].values())
        n_unr = sum(v == "unresolved" for v in self.payload["states"].values())
        if not self.messagebox.askyesno("Complete frame", f"Mark this frame complete?\n\noccupied = {n_occ}\nunresolved = {n_unr}\nempty = {len(self.slots) - n_occ - n_unr}\n\nOnly confirm after checking every stall.", default="no", icon="question"):
            return
        self._last_complete_at = now
        m = self.payload["revision_annotation"]
        m["complete"] = True
        m["completed_at"] = iso_now()
        self.save()
        if self.completed_count() == self.progress_gate and not self.ack_gate:
            self.messagebox.showinfo("Progress gate", f"{self.progress_gate} frames completed. Stop and report before resuming with --acknowledge-progress-gate.")
            self.root.destroy()
            return
        self.load_index(min(self.index + 1, len(self.rows) - 1))

    def hint(self, text):
        self.status.config(text=self.status.cget("text") + "  |  " + text)

    def on_key(self, event):
        self.mark_activity()
        k = event.keysym
        step = 5 if (event.state & 0x0001) else 1
        if k == "Left": self.shift(-step, 0)
        elif k == "Right": self.shift(step, 0)
        elif k == "Up": self.shift(0, -step)
        elif k == "Down": self.shift(0, step)
        elif k in ("Return", "KP_Enter"): self.complete_and_next()
        elif k == "n": self.load_index(self.index + 1)
        elif k == "p": self.load_index(self.index - 1)
        elif k == "g": self.load_index(self.first_incomplete())
        elif k == "u" and self.undo_stack:
            if self._sealed_block(): return
            self.payload["states"] = self.undo_stack.pop(); self.uncomplete(); self.redraw()
        elif k == "s": self.save(); self.redraw()
        elif k == "f": self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))
        elif k == "Escape": self.root.attributes("-fullscreen", False)
        elif k == "0": self.fit()
        elif k == "q": self.quit()

    def quit(self):
        if not self.messagebox.askyesno("Quit", "Save the current draft and quit the tool?", default="no"):
            return
        self.save(); self.root.destroy()

    def enforce_startup_gate(self, progress_gate: int, acknowledged: bool) -> None:
        """Refuse to continue past the progress gate without explicit acknowledgement (cannot be bypassed by restarting)."""
        done = self.completed_count()
        if done >= progress_gate and not acknowledged:
            msg = f"{done} frames are complete; the {progress_gate}-frame gate requires the report to be reviewed before continuing. Restart with --acknowledge-progress-gate."
            if os.environ.get("SNOW_REVIEW_NO_DIALOG"): sys.stderr.write("PROGRESS GATE: " + msg + "\n")
            else: self.messagebox.showerror("Progress gate", msg)
            self.root.destroy(); raise SystemExit(3)

    def assert_startup_render(self) -> dict:
        """Normal-mode hard check: the whole image must be visible on the canvas; otherwise refuse to run."""
        self.root.update()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height(); w, h = self.image.size
        items = self.canvas.find_all(); imgs = [i for i in items if self.canvas.type(i) == "image"]
        ok = (len(imgs) == 1 and self.offset_x >= 0 and self.offset_y >= 0 and self.offset_x + w * self.scale <= cw + 0.5 and self.offset_y + h * self.scale <= ch + 0.5)
        rep = dict(canvas=[cw, ch], photo=[int(w * self.scale), int(h * self.scale)], offset=[self.offset_x, self.offset_y], image_items=len(imgs),
                   polygon_items=sum(1 for i in items if self.canvas.type(i) == "polygon"), rectangle_items=sum(1 for i in items if self.canvas.type(i) == "rectangle"), image_fully_visible=ok)
        if not ok:
            self.messagebox.showerror("Render check failed", f"The image is not fully visible on the canvas ({rep}). Refusing to annotate."); self.root.destroy(); raise SystemExit(4)
        return rep

    def capture_render_evidence(self, out_dir) -> dict:
        """Evidence for validation runs: canvas inventory, a macOS window screenshot (screencapture) and its pixel variance."""
        import subprocess, json as _json
        from pathlib import Path as _P
        out_dir = _P(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        rep = self.assert_startup_render() if True else {}
        self.root.update(); x, y = self.root.winfo_rootx(), self.root.winfo_rooty(); ww, wh = self.root.winfo_width(), self.root.winfo_height()
        png = out_dir / "canvas_render.png"; var = None
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height(); w, h = self.image.size
        try:
            ps = out_dir / "canvas_snapshot.ps"
            # export the WHOLE canvas: page size = canvas size, origin at the canvas corner (no paper clipping)
            self.canvas.postscript(file=str(ps), colormode="color", x=0, y=0, width=cw, height=ch, pagewidth=cw, pageheight=ch, pagex=0, pagey=0, pageanchor="sw")
            subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dEPSCrop", "-sDEVICE=png16m", "-r72", f"-sOutputFile={png}", str(ps)], check=True, timeout=60)
            from PIL import Image as _I, ImageStat as _S
            im = _I.open(png).convert("RGB"); st = _S.Stat(im); var = sum(st.var) / 3.0
            rw, rh = im.size; rep["raster_size"] = [rw, rh]; rep["raster_matches_canvas"] = (abs(rw - cw) <= 2 and abs(rh - ch) <= 2)
            # image region corners inside the raster must not be uniform background (sample 8x8 patches)
            sx, sy = rw / max(1, cw), rh / max(1, ch); x0, y0 = int(self.offset_x * sx), int(self.offset_y * sy); x1, y1 = int((self.offset_x + w * self.scale) * sx) - 1, int((self.offset_y + h * self.scale) * sy) - 1
            def patch_var(px, py):
                box = (max(0, px - 4), max(0, py - 4), min(rw, px + 4), min(rh, py + 4)); return sum(_S.Stat(im.crop(box)).var) / 3.0
            rep["image_corner_patch_variance"] = dict(top_left=patch_var(x0 + 4, y0 + 4), top_right=patch_var(x1 - 4, y0 + 4), bottom_left=patch_var(x0 + 4, y1 - 4), bottom_right=patch_var(x1 - 4, y1 - 4))
            rep["image_corners_in_raster"] = all(0 <= v <= rw for v in (x0, x1)) and all(0 <= v <= rh for v in (y0, y1))
            # overlay colour present? count pixels close to the overlay colours drawn by this tool
            px = im.load(); targets = OVERLAY_RGB; hits = {k: 0 for k in targets}
            for yy in range(0, rh, 2):
                for xx in range(0, rw, 2):
                    r_, g_, b_ = px[xx, yy]
                    for k, (tr, tg, tb) in targets.items():
                        if abs(r_ - tr) < 40 and abs(g_ - tg) < 40 and abs(b_ - tb) < 40: hits[k] += 1
            rep["overlay_pixels_sampled"] = hits; rep["overlay_present"] = any(v > 20 for v in hits.values())
        except Exception as e:
            rep["render_error"] = str(e)
        try:  # optional window screenshot; requires Screen Recording permission for the terminal
            subprocess.run(["screencapture", "-x", "-R", f"{x},{y},{ww},{wh}", str(out_dir / "window_screenshot.png")], check=True, timeout=20)
        except Exception as e:
            rep["screenshot_error"] = str(e)
        rep.update(window=[x, y, ww, wh], canvas_render=str(png) if png.exists() else None, canvas_pixel_variance=var, canvas_variance_ok=(var is not None and var > 25.0),
                   interpreter=sys.executable, tk_version=self.tk.TkVersion, tool_version=TOOL_VERSION, generated_at=iso_now())
        rep["render_assertion_passed"] = bool(rep["image_fully_visible"] and rep["canvas_variance_ok"] and rep.get("raster_matches_canvas") and rep.get("image_corners_in_raster") and rep.get("overlay_present") and min(rep.get("image_corner_patch_variance", {"x": 0}).values()) > 0.0)
        (out_dir / "render_evidence.json").write_text(_json.dumps(rep, indent=1))
        return rep

from output_lock import acquire_output_lock, release_output_lock
from sealed_frames import load_sealed_lists, discover_seal_lists, startup_check, exit_check  # sealed frames read-only (occ-1.0.3)  # flock-based single-instance lock (tools 1.0.8+)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, default=Path.home() / "Downloads" / "image3")
    ap.add_argument("--slots", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--label-set-id", default="INDEPENDENT-OCC-120")
    ap.add_argument("--progress-gate", type=int, default=20)
    ap.add_argument("--acknowledge-progress-gate", action="store_true")
    ap.add_argument("--render-validation-dir", type=Path, help="validation mode: isolated outputs, screenshot evidence, no formal output")
    ap.add_argument("--sealed-list", type=Path, nargs="*", default=[], help="seal lists whose frames are read-only; lists for the label set in --seals-dir are added automatically")
    ap.add_argument("--seals-dir", type=Path, default=Path(__file__).resolve().parent / "seals")
    ap.add_argument("--sealed-guard-test", action="store_true", help="validation mode: prove sealed frames cannot be changed (internal temporary directory; the requested --output-dir is never written)")
    a = ap.parse_args()
    with a.manifest.open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    slots = json.load(a.slots.open())["slots"]
    meta = dict(label_set_id=a.label_set_id, protocol_sha256=sha256_file(a.protocol), manifest_sha256=sha256_file(a.manifest), slots_sha256=sha256_file(a.slots))
    if a.sealed_guard_test:
        run_sealed_guard_test(a, rows, slots, meta); return
    out_dir = (a.render_validation_dir / "probe_outputs") if a.render_validation_dir else a.output_dir
    formal = not a.render_validation_dir
    sealed = load_sealed_lists(sorted(set(list(a.sealed_list) + (discover_seal_lists(a.seals_dir, [a.label_set_id]) if formal else [])))) if formal else {}
    lock = acquire_output_lock(out_dir, "blind_occupancy_tool")
    try:
        startup_check(out_dir, sealed, "blind_occupancy_tool")
        app = OccupancyApp(rows, a.image_dir, slots, out_dir, meta, a.progress_gate, a.acknowledge_progress_gate, sealed=sealed)
        if a.render_validation_dir:
            app.root.after(1800, lambda: (app.capture_render_evidence(a.render_validation_dir), app.root.destroy()))
        else:
            app.enforce_startup_gate(a.progress_gate, a.acknowledge_progress_gate)
            app.root.after(600, app.assert_startup_render)
        app.root.mainloop()
        exit_check(out_dir, sealed, "blind_occupancy_tool")
    finally:
        release_output_lock(lock)


def run_sealed_guard_test(a, rows, slots, meta):
    """Isolated proof (occ-1.0.3): in an internal temporary directory, build 3 completed frames, seal 2, drive the real
    app through click / shift / undo / complete / autosave / save on the sealed frames and one normal edit+save on the
    unsealed frame; pass only if the sealed digests are unchanged, the unsealed frame was saved and the requested
    --output-dir listing is unchanged."""
    import tempfile
    requested = a.output_dir; req_before = sorted(p.name for p in requested.glob("*")) if requested.exists() else None
    out = Path(tempfile.mkdtemp(prefix="snow_review_occ_guard_")); rows = rows[:3]
    fx = OccupancyApp(rows, a.image_dir, slots, out, meta, a.progress_gate, True)  # fixture builder (its draft is discarded)
    for r in rows:
        im = fx.Image.open(a.image_dir / r["image_name"]); pl = fx.blank(r, *im.size); pl["states"][str(slots[0]["slot_id"])] = "occupied"
        m = pl["revision_annotation"]; m["complete"] = True; m["completed_at"] = iso_now(); m["active_seconds"] = 5.0; m["wall_seconds"] = 5.0; atomic_write(fx.out_path(r), pl)
    seal = out / "guard_test.sha256"; seal.write_text("".join(f"{sha256_file(fx.out_path(r))}  {fx.out_path(r).name}\n" for r in rows[:2]))
    fx.payload = {}; fx.root.destroy()
    sealed = load_sealed_lists([seal]); before = {n: sha256_file(out / n) for n in sealed}
    app = OccupancyApp(rows, a.image_dir, slots, out, meta, a.progress_gate, True, sealed=sealed)  # opened in the sealed state, as the formal path does
    result = {}

    class Ev:
        def __init__(self, x, y): self.x = x; self.y = y; self.state = 0; self.keysym = ""

    def probe():
        for i in range(2):
            app.load_index(i); app._frame_loaded_at = 0.0; app._return_armed = True
            cx, cy = slots[0]["polygon"][0]; px = app.offset_x + (cx + 2) * app.scale; py = app.offset_y + (cy + 2) * app.scale
            app.on_click(Ev(px, py)); app.shift(3, 0); app.undo_stack.append(dict(app.payload["states"])); e = Ev(0, 0); e.keysym = "u"; app.on_key(e)
            app.mark_activity(); app.save(); app.autosave(); app.complete_and_next()
        app.load_index(2); sid = str(slots[1]["slot_id"]); app.payload["states"][sid] = "unresolved"; app.save(); app.root.destroy()

    app.root.after(800, probe); app.root.mainloop()
    after = {n: sha256_file(out / n) for n in app.sealed}; result["sealed_unchanged"] = after == before
    result["unsealed_saved"] = json.load(app.out_path(rows[2]).open())["states"][str(slots[1]["slot_id"])] == "unresolved"
    req_after = sorted(p.name for p in requested.glob("*")) if requested.exists() else None; result["requested_dir_untouched"] = req_before == req_after
    ok = all(result.values()); print(("SEALED GUARD TEST PASSED " if ok else "SEALED GUARD TEST FAILED ") + json.dumps(result) + f"; fixture in {out}")
    raise SystemExit(0 if ok else 7)


if __name__ == "__main__":
    main()
