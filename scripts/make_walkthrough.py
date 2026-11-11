"""Render the short walkthrough video (docs/walkthrough.mp4) and README GIF.

Frames are drawn with Pillow from the pipeline exports and encoded with
ffmpeg, one file per scene, then joined. Requires Pillow and ffmpeg.

Usage: python scripts/make_walkthrough.py [--stills DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H, S, FPS = 1280, 720, 2, 30
BG, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3dc"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#b9b8ae"
FONT_DIR = Path("/usr/share/fonts/opentype/inter")
_fonts: dict = {}


def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    key = (size, weight)
    if key not in _fonts:
        path = FONT_DIR / f"Inter-{weight}.otf"
        fallback = "DejaVuSans-Bold.ttf" if weight in ("Bold", "SemiBold") else "DejaVuSans.ttf"
        _fonts[key] = ImageFont.truetype(str(path) if path.exists() else fallback, size * S)
    return _fonts[key]


def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def ramp(t: float, start: float, dur: float) -> float:
    return ease((t - start) / dur)


def mix(color: str, alpha: float) -> str:
    """Blend a hex color toward the background; used for fades."""
    a = min(1.0, max(0.0, alpha))
    c = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(BG[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(b[i] + (c[i] - b[i]) * a):02x}" for i in range(3))


class Frame:
    def __init__(self) -> None:
        self.im = Image.new("RGB", (W * S, H * S), BG)
        self.d = ImageDraw.Draw(self.im)

    def text(self, x, y, s, size=28, weight="Regular", fill=INK, anchor="la", alpha=1.0):
        if alpha > 0.02:
            self.d.text((x * S, y * S), s, font=font(size, weight), fill=mix(fill, alpha), anchor=anchor)

    def rect(self, x0, y0, x1, y1, fill, r=0):
        if x1 > x0 and y1 > y0:
            self.d.rounded_rectangle((x0 * S, y0 * S, x1 * S, y1 * S), radius=r * S, fill=fill)

    def line(self, x0, y0, x1, y1, fill, w=1.5, dash=None):
        if dash is None:
            self.d.line((x0 * S, y0 * S, x1 * S, y1 * S), fill=fill, width=round(w * S))
            return
        n = int(math.hypot(x1 - x0, y1 - y0) // (2 * dash))
        for i in range(n + 1):
            a = i * 2 * dash / max(1e-9, math.hypot(x1 - x0, y1 - y0))
            b = min(1.0, a + dash / max(1e-9, math.hypot(x1 - x0, y1 - y0)))
            self.d.line(
                ((x0 + (x1 - x0) * a) * S, (y0 + (y1 - y0) * a) * S, (x0 + (x1 - x0) * b) * S, (y0 + (y1 - y0) * b) * S),
                fill=fill,
                width=round(w * S),
            )

    def dot(self, x, y, r, fill):
        self.d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S), fill=fill)

    def out(self) -> Image.Image:
        return self.im.resize((W, H), Image.LANCZOS)


def header(f: Frame, kicker: str, title: str, sub: str = "", a_title=1.0, a_sub=1.0) -> None:
    f.text(80, 56, kicker.upper(), 17, "SemiBold", MUTED)
    f.text(80, 86, title, 38, "Bold", INK, alpha=a_title)
    if sub:
        f.text(80, 140, sub, 23, "Regular", INK2, alpha=a_sub)


# ----------------------------------------------------------------------
# Scenes: each takes local time in seconds and returns a Frame
# ----------------------------------------------------------------------
def scene_title(t: float, c: dict) -> Frame:
    f = Frame()
    f.text(80, 220, "Catching a sensor change", 62, "Bold", alpha=ramp(t, 0.1, 0.6))
    f.text(80, 296, "in semiconductor fab data", 62, "Bold", alpha=ramp(t, 0.3, 0.6))
    f.text(80, 400, f"{c['runs']:,} production runs.  590 sensors.  One process line.", 28, fill=INK2, alpha=ramp(t, 1.0, 0.6))
    f.text(80, 446, "Statistical process control on the UCI SECOM dataset", 28, fill=INK2, alpha=ramp(t, 1.4, 0.6))
    f.text(80, 610, "Shyam Iyer", 22, "SemiBold", MUTED, alpha=ramp(t, 1.8, 0.6))
    return f


def scene_spe(t: float, c: dict) -> Frame:
    f = Frame()
    x0, x1, y0, y1 = 120, 1200, 210, 640
    lo, hi = math.log(30), math.log(1000)
    X = lambda r: x0 + (r - 1) / (c["runs"] - 1) * (x1 - x0)  # noqa: E731
    Y = lambda v: y1 - (math.log(v) - lo) / (hi - lo) * (y1 - y0)  # noqa: E731
    shown = int(c["runs"] * ramp(t, 2.5, 6.5)) if t > 2.5 else 0
    past = shown >= c["cp_run"]
    done = ramp(t, 10.0, 0.6)
    if not past:
        header(f, "Multivariate monitor", "One number per run summarises 321 sensors",
               "Higher means the sensors are behaving unlike the baseline period.", ramp(t, 0.1, 0.5), ramp(t, 0.6, 0.5))
    else:
        header(f, "Multivariate monitor", f"Run {c['cp_run']}: the pattern shifts and never comes back",
               f"After the change, {c['spe_p2']:.0f}% of runs break a limit designed to flag 1%.", 1.0, done)
    for v in (50, 100, 200, 500):
        f.line(x0, Y(v), x1, Y(v), GRID, 1)
        f.text(x0 - 12, Y(v), str(v), 17, fill=MUTED, anchor="rm")
    f.line(x0, y1, x1, y1, NEUTRAL, 1.5)
    f.text(x1, y1 + 14, "production run, in time order", 17, fill=MUTED, anchor="ra")
    f.text(x0, y0 - 26, "SPE (log scale)", 17, fill=MUTED)
    a = ramp(t, 1.4, 0.6)
    f.line(x0, Y(c["spe_ucl"]), x1, Y(c["spe_ucl"]), mix(INK2, a), 2, dash=7)
    for r, v in zip(c["run"][:shown], c["spe"][:shown]):
        f.dot(X(r), Y(min(990, max(31, v))), 2.6, ORANGE if v > c["spe_ucl"] else BLUE)
    if a > 0.02:
        f.rect(x0 + 6, Y(c["spe_ucl"]) - 36, x0 + 262, Y(c["spe_ucl"]) - 8, BG, 4)
        f.text(x0 + 14, Y(c["spe_ucl"]) - 22, "99% limit, set on early data", 18, "Medium", INK2, anchor="lm", alpha=a)
    if past:
        f.line(X(c["cp_run"]), y0 - 6, X(c["cp_run"]), y1, INK, 2)
        f.text(X(c["cp_run"]) - 10, y0 + 2, f"Change point, {c['cp_label']}", 19, "SemiBold", INK, anchor="ra")
    if shown:
        f.dot(900, 182, 6, BLUE)
        f.text(914, 182, "within limit", 17, fill=INK2, anchor="lm")
        f.dot(1050, 182, 6, ORANGE)
        f.text(1064, 182, "alarm", 17, fill=INK2, anchor="lm")
    return f


def scene_cause(t: float, c: dict) -> Frame:
    f = Frame()
    header(f, "Root cause", "Seven sensors went from a frozen value to live readings",
           f"They carry {c['share_p2']:.0f}% of the signal after the change, against {c['share_p1']:.0f}% before.",
           ramp(t, 0.1, 0.5), ramp(t, 3.0, 0.6))
    rows = c["contrib"][:8]
    top = max(r[2] for r in rows)
    x0, x1, y = 190, 1080, 230
    for i, (name, p1, p2, frozen) in enumerate(rows):
        g = ramp(t, 0.8 + 0.12 * i, 1.2)
        f.text(x0 - 16, y + 15, name, 20, "Medium", INK2, anchor="rm")
        f.rect(x0, y, x0 + (x1 - x0) * p2 / top * g, y + 20, ORANGE if frozen else BLUE, 4)
        f.rect(x0, y + 23, x0 + (x1 - x0) * p1 / top * g, y + 33, NEUTRAL, 3)
        f.text(x0 + (x1 - x0) * p2 / top * g + 10, y + 10, f"{p2:.1f}%", 18, fill=INK2, anchor="lm", alpha=g)
        y += 50
    f.rect(190, 650, 206, 666, ORANGE, 3)
    f.text(214, 658, "after change, frozen-then-live sensor", 17, fill=INK2, anchor="lm")
    f.rect(560, 650, 576, 666, BLUE, 3)
    f.text(584, 658, "after change, other sensor", 17, fill=INK2, anchor="lm")
    f.rect(840, 650, 856, 666, NEUTRAL, 3)
    f.text(864, 658, "before change", 17, fill=INK2, anchor="lm")
    f.text(1200, 205, "share of total SPE", 17, fill=MUTED, anchor="ra")
    return f


def scene_cost(t: float, c: dict) -> Frame:
    f = Frame()
    header(f, "What it costs", f"Stale limits: about {c['frozen']:.0f} alarms a day, nearly all false",
           "Re-baselining after the change brings that under one a day.", ramp(t, 0.1, 0.5), ramp(t, 4.0, 0.6))
    bars = [
        ("Limits frozen from the early period", c["frozen"], BLUE, 0.8, "{:.1f}"),
        ("Rolling limits, refitted daily", c["rolling"], BLUE, 2.6, "{:.2f}"),
        ("Re-baselined after the change", c["rebased"], ORANGE, 4.0, "{:.2f}"),
    ]
    x0, x1, y = 120, 1000, 250
    for label, v, color, start, fmt in bars:
        g = ramp(t, start, 1.2)
        f.text(x0, y, label, 23, "Medium", INK2, alpha=ramp(t, start - 0.2, 0.5))
        f.rect(x0, y + 38, x0 + max(4, (x1 - x0) * v / c["frozen"]) * g, y + 82, color, 6)
        f.text(x0 + max(4, (x1 - x0) * v / c["frozen"]) * g + 16, y + 60, fmt.format(v * g), 34, "Bold", INK, anchor="lm", alpha=g)
        y += 130
    f.text(x0, 650, f"alarms per production day over the {c['days']}-day monitoring period", 18, fill=MUTED)
    return f


def scene_honest(t: float, c: dict) -> Frame:
    f = Frame()
    header(f, "The honest part", "The monitor detects process change, not failed product",
           "Can the sensors predict which runs fail? Barely, once tested on later data.", ramp(t, 0.1, 0.5), ramp(t, 0.8, 0.5))
    a, b = ramp(t, 1.6, 0.7), ramp(t, 3.0, 0.7)
    f.text(120, 270, "Shuffled cross-validation", 24, "Medium", INK2, alpha=a)
    f.text(120, 316, c["cv"], 72, "Bold", NEUTRAL, alpha=a)
    f.text(120, 416, "AUC, what a random split reports", 20, fill=MUTED, alpha=a)
    f.text(680, 270, "Tested on later runs", 24, "Medium", INK2, alpha=b)
    f.text(680, 316, c["holdout"], 72, "Bold", BLUE, alpha=b)
    f.text(680, 416, "AUC, what production would see (0.5 is chance)", 20, fill=MUTED, alpha=b)
    f.text(120, 560, "Random splits leak the time trend and overstate performance by about 0.2 AUC.", 25, fill=INK, alpha=ramp(t, 4.4, 0.7))
    return f


def scene_end(t: float, c: dict) -> Frame:
    f = Frame()
    f.text(80, 210, "Re-baseline after a change.", 52, "Bold", alpha=ramp(t, 0.1, 0.6))
    f.text(80, 276, "Check data quality before a sensor", 52, "Bold", alpha=ramp(t, 0.5, 0.6))
    f.text(80, 342, "enters a control plan.", 52, "Bold", alpha=ramp(t, 0.5, 0.6))
    f.text(80, 470, "Python pipeline  ·  SPC and PCA monitoring  ·  Spotfire  ·  JMP", 26, fill=INK2, alpha=ramp(t, 1.4, 0.6))
    f.text(80, 520, "github.com/Shim-1122/secom-spc", 30, "SemiBold", BLUE, alpha=ramp(t, 1.9, 0.6))
    return f


SCENES = [(scene_title, 4.5), (scene_spe, 13.0), (scene_cause, 8.0), (scene_cost, 9.0), (scene_honest, 8.5), (scene_end, 5.5)]


def load() -> dict:
    s = json.loads((ROOT / "reports" / "summary.json").read_text())
    runs = pd.read_csv(ROOT / "reports" / "exports" / "runs.csv")
    con = pd.read_csv(ROOT / "reports" / "exports" / "spe_contribution_by_phase.csv")
    burden = [b for b in s["monitoring"]["burden"] if b["period"] == "after re-baselining window"]
    cv = [m["cv10_roc_auc"] for m in s["models_random_cv10"].values()]
    ho = [m["test_roc_auc"] for k, m in s["models_temporal"].items() if k in s["models_random_cv10"]]  # same four models
    ts = pd.Timestamp(s["changepoint"]["timestamp"])
    return {
        "runs": len(runs),
        "run": runs.run_id.tolist(),
        "spe": runs.spe_q.tolist(),
        "spe_ucl": float(runs.spe_q_ucl_empirical.iloc[0]),
        "spe_p2": s["mspc"]["spe_alarm_rate_phase2_empirical_pct"],
        "cp_run": s["changepoint"]["run_id"],
        "cp_label": f"{ts.day} {ts.strftime('%b %Y')}",
        "share_p1": s["mspc"]["spe_share_frozen_then_live_phase1_pct"],
        "share_p2": s["mspc"]["spe_share_frozen_then_live_phase2_pct"],
        "contrib": [(r.sensor, 100 * r.share_phase1, 100 * r.share_phase2, bool(r.frozen_then_live)) for r in con.itertuples()],
        "frozen": burden[0]["alarms_per_day"],
        "rebased": burden[1]["alarms_per_day"],
        "rolling": burden[2]["alarms_per_day"],
        "days": burden[0]["days"],
        "cv": f"{min(cv):.2f} to {max(cv):.2f}",
        "holdout": f"{min(ho):.2f} to {max(ho):.2f}",
    }


def render_scene(fn, dur: float, c: dict, path: Path) -> None:
    tmp = path.with_name(path.stem + ".part.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(tmp)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    white = Image.new("RGB", (W, H), BG)
    n = round(dur * FPS)
    for i in range(n):
        t = i / FPS
        im = fn(t, c).out()
        fade = min(1.0, (dur - t) / 0.35)  # fade out to the background at the end of each scene
        if fade < 1:
            im = Image.blend(white, im, max(0.0, fade))
        p.stdin.write(im.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}")
    tmp.rename(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", type=Path, help="write one still per scene to this folder and exit")
    args = ap.parse_args()
    c = load()
    if args.stills:
        args.stills.mkdir(parents=True, exist_ok=True)
        for i, (fn, dur) in enumerate(SCENES):
            fn(dur - 0.6, c).out().save(args.stills / f"scene{i}.png")
            if fn is scene_spe:
                fn(5.5, c).out().save(args.stills / f"scene{i}_mid.png")
        return
    out = ROOT / "docs" / "walkthrough.mp4"
    with tempfile.TemporaryDirectory() as td:
        parts = []
        for i, (fn, dur) in enumerate(SCENES):
            part = Path(td) / f"s{i}.mp4"
            render_scene(fn, dur, c, part)
            parts.append(part)
        lst = Path(td) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        "-movflags", "+faststart", str(out)], check=True)
    gif = ROOT / "docs" / "img" / "walkthrough.gif"
    vf = "fps=10,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-vf", vf, str(gif)], check=True)
    print(f"wrote {out.relative_to(ROOT)} and {gif.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
