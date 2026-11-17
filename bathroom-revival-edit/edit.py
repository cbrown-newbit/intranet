#!/usr/bin/env python3
"""Bathroom Revival Weekend - programmatic edit pipeline (Python + ffmpeg).

    python3 edit.py scan     --media DIR      # pair .LRF proxies with .MP4 masters -> build/manifest.json
    python3 edit.py analyze  --media DIR      # LRF-only analysis (motion, framing, WB, speech) -> build/analysis.json
    python3 edit.py edl                       # beat-locked cut -> edl.json, EDL.md, bathroom_revival.edl (CMX3600)
    python3 edit.py render   --media DIR --aspect both   # conform against the MP4 masters
    python3 edit.py all      --media DIR --aspect both   # everything above

Analysis only ever opens the .LRF proxies; render only ever opens the .MP4 masters
(unless --proxy is given for a fast preview). Requires: ffmpeg/ffprobe, numpy, Pillow.
"""
import argparse, datetime as dt, hashlib, json, math, os, re, subprocess, sys, wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- config
FPS = 30
BPM = 90.0                      # 90 BPM @ 30 fps -> one beat is exactly 20 frames
SR = 48000
TITLE, SUBTITLE = "Bathroom Revival Weekend", "Before & After"
ACT_TITLES = {1: "The Before", 2: "The Deep Clean", 3: "The Reveal"}
ACT_BEATS = {1: 24, 2: 40, 3: 44}            # 108 beats = 72.0 s at 90 BPM
# Story window: Friday 11:25 AM (clip 0195) -> Sunday 5:54 AM (clip 0220).
SINCE, UNTIL = dt.datetime(2025, 5, 23, 11, 25), dt.datetime(2025, 5, 25, 6, 0)
ACT2_FROM, ACT3_FROM = dt.datetime(2025, 5, 23, 17, 0), dt.datetime(2025, 5, 25, 0, 0)
TRANS_FRAMES = {"cut": 0, "whip": 6, "dissolve": 12, "dip": 10}
XFADE = {"whip": "slideleft", "dissolve": "fade", "dip": "fadewhite"}
ACCENT = (232, 176, 74)
CLIP_RE = re.compile(r"^DJI_(\d{14})_(\d{4})_D\.(MP4|LRF)$", re.I)


def bf(beats):
    """Beat count -> frame index on the timeline."""
    return int(round(beats * 60.0 / BPM * FPS))


def run(cmd, **kw):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, **kw)
    if p.returncode:
        sys.exit("command failed: %s\n%s" % (" ".join(map(str, cmd[:12])), p.stderr.decode(errors="replace")[-2000:]))
    return p.stdout


def clip_start(stem):
    return dt.datetime.strptime(stem.split("_")[1], "%Y%m%d%H%M%S")


def clock(t):
    return t.strftime("%A ") + t.strftime("%I:%M %p").lstrip("0")


def tc(frames):
    s, f = divmod(int(frames), FPS)
    return "%02d:%02d:%02d:%02d" % (s // 3600, s // 60 % 60, s % 60, f)


def act_of(start):
    return 1 if start < ACT2_FROM else 2 if start < ACT3_FROM else 3


# ----------------------------------------------------------------------------- scan
def probe(path):
    j = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    n, d = v.get("avg_frame_rate", "30/1").split("/")
    return {"path": str(path), "duration": float(j["format"]["duration"]), "w": v["width"], "h": v["height"],
            "fps": round(float(n) / max(float(d), 1), 3), "transfer": v.get("color_transfer", ""),
            "audio": any(s["codec_type"] == "audio" for s in j["streams"])}


def cmd_scan(a):
    drop = [x.strip() for x in (a.exclude or "").split(",") if x.strip()]
    keep = lambda stem: SINCE <= clip_start(stem) <= UNTIL and not any(x in stem for x in drop)
    clips = {}
    media = Path(a.media)
    if media.is_dir():
        for p in sorted(media.rglob("*")):
            m = CLIP_RE.match(p.name)      # exact names only: "..._D (1).MP4" Drive duplicates are skipped
            if m and keep(p.stem):
                c = clips.setdefault(p.stem, {"stem": p.stem, "num": m[2], "start": clip_start(p.stem).isoformat()})
                c[m[3].lower()] = probe(p)
    if clips:
        for c in clips.values():
            c["duration"] = (c.get("mp4") or c["lrf"])["duration"]
            c["fps"] = (c.get("mp4") or c["lrf"])["fps"]
        src = "scanned " + str(media)
    else:  # no media on this machine: fall back to the Drive listing so an EDL can still be drafted
        dm = json.loads((HERE / "drive_manifest.json").read_text())
        clips = {c["stem"]: c for c in dm["clips"] if keep(c["stem"])}
        src = "drive_manifest.json (no media found in %s; durations from Drive timestamps)" % media
    out = sorted(clips.values(), key=lambda c: c["start"])
    for c in out:
        c["act"] = act_of(clip_start(c["stem"]))
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / "manifest.json").write_text(json.dumps({"source": src, "clips": out}, indent=1))
    print("manifest: %d clips (%s)" % (len(out), src))
    for c in out:
        print("  act %d  %s  %7.1fs  %s%s" % (c["act"], c["stem"], c["duration"],
              "LRF " if "lrf" in c else "--- ", "MP4" if "mp4" in c else "---"))


def load_manifest(a):
    if not (a.work / "manifest.json").exists():
        cmd_scan(a)
    return json.loads((a.work / "manifest.json").read_text())["clips"]


# ----------------------------------------------------------------------------- analyze (LRF proxies only)
AW, AH, AFPS = 64, 36, 4


def analyze_clip(lrf):
    raw = run(["ffmpeg", "-v", "error", "-i", lrf, "-an", "-vf", "fps=%d,scale=%d:%d" % (AFPS, AW, AH),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    fr = np.frombuffer(raw, np.uint8).reshape(-1, AH, AW, 3).astype(np.float32)
    luma = fr.mean(axis=3)
    diff = np.abs(np.diff(luma, axis=0))                         # per-pixel motion, 4 Hz
    motion = np.concatenate([[0.0], diff.mean(axis=(1, 2))])
    col = diff.sum(axis=1) + 1e-3                                # where on the x axis the action is
    cx = np.concatenate([[0.5], (col * np.linspace(0, 1, AW)).sum(axis=1) / col.sum(axis=1)])
    mid = (luma > 60) & (luma < 200)                             # grey-world WB estimate from midtones
    rgb = fr[mid].mean(axis=0) if mid.sum() > 100 else fr.reshape(-1, 3).mean(axis=0)
    out = {"motion": [round(float(x), 3) for x in motion], "cx": [round(float(x), 3) for x in cx],
           "rgb": [round(float(x), 2) for x in rgb], "luma": round(float(luma.mean()), 2), "speech": [], "hits": []}
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", lrf, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                       capture_output=True)
    au = np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768
    if len(au) > 16000:
        # speech-ish: loud 0.25 s frames whose energy sits in the 300-3400 Hz voice band
        n = 4000
        F = np.abs(np.fft.rfft(au[:len(au) // n * n].reshape(-1, n), axis=1)) ** 2
        f = np.fft.rfftfreq(n, 1 / 16000)
        tot = F.sum(axis=1) + 1e-9
        ratio = F[:, (f > 300) & (f < 3400)].sum(axis=1) / tot
        db = 10 * np.log10(tot)
        voiced = (ratio > 0.6) & (db > np.median(db) + 8)
        i = 0
        while i < len(voiced):
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            if j - i >= 3:
                out["speech"].append([round(i * .25, 2), round(j * .25, 2)])
            i = max(j, i + 1)
        # hits: sharp transients (a tool set down, a lid, a click) for the pop/snap SFX
        m = 320
        r = np.sqrt((au[:len(au) // m * m].reshape(-1, m) ** 2).mean(axis=1))
        thr = max(np.percentile(r, 97), 5 * np.median(r) + 1e-4)
        last = -1
        for k in np.where((r[1:] > thr) & (r[1:] > 3 * r[:-1]))[0] + 1:
            if k * .02 - last > 0.4:
                out["hits"].append(round(k * .02, 2))
                last = k * .02
    return out


def cmd_analyze(a):
    clips = load_manifest(a)
    cache = a.work / "analysis.json"
    res = json.loads(cache.read_text()) if cache.exists() else {}
    for c in clips:
        if "lrf" not in c or "path" not in c["lrf"]:
            continue
        if c["stem"] not in res:
            print("analyze (LRF)", c["stem"], flush=True)
            res[c["stem"]] = analyze_clip(c["lrf"]["path"])
            cache.write_text(json.dumps(res))
    print("analysis: %d/%d clips have LRF analysis" % (len(res), len(clips)))
    if not res:
        print("  (no .LRF files found - the EDL will use evenly spaced blind picks)")


# ----------------------------------------------------------------------------- EDL builder
def src_time(t, s0, s1, D):
    """Source seconds consumed after t timeline seconds of a linear s0->s1 speed ramp lasting D."""
    return s0 * t + (s1 - s0) * t * t / (2 * D) if D > 0 else 0.0


def share(n, weights):
    """Split n slots across items proportionally to weights (largest remainder, >=1 each when possible)."""
    if n <= len(weights):
        idx = {int(round(i * (len(weights) - 1) / max(n - 1, 1))) for i in range(n)}
        return [1 if i in idx else 0 for i in range(len(weights))]
    tot = sum(weights)
    raw = [(n - len(weights)) * w / tot for w in weights]
    out = [1 + int(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)[:n - sum(out)]:
        out[i] += 1
    return out


def pick(clip, an, k, n, span, mode, hi=None, prefer=None):
    """Choose the in-point for the k-th of n picks from a clip (chronological zones)."""
    lo, hi = 0.4, (hi or clip["duration"]) - 0.4
    z0 = lo + (hi - lo) * k / n
    z1 = lo + (hi - lo) * (k + 1) / n
    last = max(lo, min(hi, z1) - span)
    first = min(max(lo, z0), last)
    if prefer is not None and first <= prefer <= last:
        return prefer, 0.5
    start = (first + last) / 2
    if an:
        m, best = np.array(an["motion"]), -1e9
        for s in np.arange(first, last + 1e-6, 0.25):
            w = m[int(s * AFPS):max(int((s + span) * AFPS), int(s * AFPS) + 1)]
            if not len(w):
                continue
            if mode == "action":      # cut on action: put the motion peak ~40% into the shot
                p = int((s + 0.4 * span) * AFPS)
                score = m[max(p - 1, 0):p + 2].max() if p < len(m) else 0
            else:                     # glide: steady, sustained camera movement, no jolts
                score = w.mean() - 2.0 * w.std()
            if score > best:
                best, start = score, float(s)
        cx = np.array(an["cx"])[int(start * AFPS):int((start + span) * AFPS) + 1]
        fx = float(np.clip(np.median(cx), 0.25, 0.75)) if len(cx) else 0.5
        return start, round(fx, 3)
    return start, 0.5


def plan_slots(clips):
    """Beat-level slot plan. Timeline stays strictly chronological so the on-screen clock only moves forward."""
    acts = {k: [c for c in clips if c["act"] == k] for k in (1, 2, 3)}
    for k in (1, 2, 3):
        if not acts[k]:
            sys.exit("no clips for act %d - check the media folder / story window" % k)
    slots = []

    def add(act, clip, beats, speed, trans, mode, note="", kind="clip"):
        slots.append({"act": act, "clip": clip, "beats": beats, "speed": list(speed), "trans": trans,
                      "mode": mode, "note": note, "kind": kind})

    # Act 1 - before: 4-beat establishing shot under the title, then 2-beat quick cuts
    a1 = acts[1]
    add(1, a1[0], 4, (1, 1), "cut", "glide", "establishing shot under main title")
    n = (ACT_BEATS[1] - 4) // 2
    per = share(n, [math.sqrt(c["duration"]) for c in a1])
    j = 0
    for c, m in zip(a1, per):
        for _ in range(m):
            add(1, c, 2, (1, 2) if j % 3 == 2 else (1, 1), "cut", "action",
                "quick cut on action" + (", ramp 1x>2x" if j % 3 == 2 else ""))
            j += 1

    # Act 2 - the grind: jump cuts in a 2-1-1 beat rhythm, per shooting session (gap > 2 h = new session)
    sessions = [[acts[2][0]]]
    for c in acts[2][1:]:
        prev = sessions[-1][-1]
        gap = clip_start(c["stem"]) - clip_start(prev["stem"]) - dt.timedelta(seconds=prev["duration"])
        sessions.append([c]) if gap > dt.timedelta(hours=2) else sessions[-1].append(c)
    sb = [4 * b for b in share(ACT_BEATS[2] // 4, [len(s) + 2 for s in sessions])]
    for s_i, (sess, beats) in enumerate(zip(sessions, sb)):
        lens, pat = [], [2, 1, 1]
        while sum(lens) < beats:
            lens.append(min(pat[len(lens) % 3], beats - sum(lens)))
        per = share(len(lens), [math.sqrt(c["duration"]) for c in sess])
        order = [c for c, m in zip(sess, per) for _ in range(m)]
        for j, (c, b) in enumerate(zip(order, lens)):
            long_ = c["duration"] > 240
            if long_:                 # long locked-off takes become time-lapse with speed ramps
                sp = [(4, 16), (16, 16), (16, 4)][j % 3] if b > 1 else (8, 8)
            else:
                sp = [(2, 2), (1.5, 3)][j % 2] if b > 1 else (1, 1)
            note = ("time-lapse" if long_ else "jump cut") + (", snap" if b == 1 else "")
            add(2, c, b, sp, "whip" if j == 0 else "cut", "action", note)

    # Act 3 - reveal: 4-beat glides on dissolves, before/after split, outro
    a3 = acts[3]
    n = (ACT_BEATS[3] - 12) // 4
    per = share(n, [1.0] * len(a3))
    j = 0
    for c, m in zip(a3, per):
        slow = c.get("fps", 30) >= 48          # only slow down when the master has the frames for it
        for _ in range(m):
            add(3, c, 4, (0.8, 0.8) if slow else (1, 1), "dip" if j == 0 else "dissolve", "glide",
                "cinematic glide" + (" (80% slow-mo)" if slow else ""))
            j += 1
    add(3, a3[-1], 8, (1, 1), "dissolve", "glide", "before/after split: %s vs %s" % (a1[0]["num"], a3[-1]["num"]),
        kind="split")
    add(3, a3[-1], 4, (1, 1), "dissolve", "glide", "outro hero shot, fade to black")
    return slots, a1[0]


def cmd_edl(a):
    clips = load_manifest(a)
    by = {c["stem"]: c for c in clips}
    an_all = json.loads((a.work / "analysis.json").read_text()) if (a.work / "analysis.json").exists() else {}
    slots, first_clip = plan_slots(clips)
    count = {}
    for s in slots:
        count[s["clip"]["stem"]] = count.get(s["clip"]["stem"], 0) + 1
    used, segs, rec, beat = {}, [], 0, 0
    reaction_done = set()
    for i, s in enumerate(slots):
        c, an = s["clip"], an_all.get(s["clip"]["stem"])
        beat += s["beats"]
        frames = bf(beat) - rec
        head = TRANS_FRAMES[s["trans"]] // 2 if i else 0
        tail = TRANS_FRAMES[slots[i + 1]["trans"]] // 2 if i + 1 < len(slots) else 0
        D = (frames + head + tail) / FPS
        s0, s1 = s["speed"]
        k, n = used.get(c["stem"], 0), count[c["stem"]]
        used[c["stem"]] = k + 1
        zone = (c["duration"] - 0.8) / n
        if (s0 + s1) / 2 * D > zone:                       # not enough footage in this zone: ease the speed off
            f = max(zone / ((s0 + s1) / 2 * D), 1 / max(s0, s1))
            s0, s1 = max(round(s0 * f, 2), 0.5), max(round(s1 * f, 2), 0.5)
        span = src_time(D, s0, s1, D)
        # keep the last clip inside its starting clock minute so the story ends on "Sunday 5:54 AM"
        hi = None
        if c is clips[-1]:
            left = 60 - clip_start(c["stem"]).second
            hi = left if left >= 15 and left < c["duration"] else None
        # spoken reactions: one per act (1 and 3), played at 1x with music ducked under it
        prefer, audio = None, "none"
        if an and s["act"] in (1, 3) and s["act"] not in reaction_done and s["kind"] == "clip" and s["beats"] >= 2:
            sp = [x for x in an["speech"] if x[1] - x[0] >= 1.0]
            if sp:
                prefer, s0, s1 = max(0.4, sp[0][0] - 0.2), 1, 1
                span = D
        rin, fx = pick(c, an, k, n, span, s["mode"], hi, prefer)
        rin = max(0.0, min(rin, c["duration"] - span))
        if s0 == s1 == 1 and s["kind"] == "clip":
            audio = "ambient"
            if an and any(min(x[1], rin + span) - max(x[0], rin) >= 0.8 for x in an["speech"]) \
                    and s["act"] not in reaction_done:
                audio = "reaction"
                reaction_done.add(s["act"])
        seg = {"n": i + 1, "act": s["act"], "kind": s["kind"], "clip": c["stem"],
               "rec_in": rec, "frames": frames, "head": head, "tail": tail,
               "render_in": round(rin, 3), "speed": [s0, s1], "fx": fx,
               "src_in": round(rin + src_time(head / FPS, s0, s1, D), 3),
               "src_out": round(rin + src_time((head + frames) / FPS, s0, s1, D), 3),
               "transition": s["trans"] if i else "fade-in", "audio": audio, "note": s["note"], "pops": []}
        if s["kind"] == "split":
            b = next(x for x in segs if x["clip"] == first_clip["stem"])
            seg["before"] = {"clip": b["clip"], "src_in": b["src_in"], "fx": b["fx"]}
        # pop/snap SFX on tool placements: audio transients when analysed, else the 1-beat snap cuts
        if s["act"] == 2:
            if an and an["hits"]:
                for h in an["hits"]:
                    if seg["src_in"] <= h < seg["src_out"]:
                        seg["pops"].append(int((h - seg["src_in"]) / max((s0 + s1) / 2, 1e-6) * FPS))
                seg["pops"] = seg["pops"][:1]
            elif s["beats"] == 1:
                seg["pops"] = [0]
        segs.append(seg)
        rec += frames
    # white balance: one correction per shooting day-part so sessions match each other
    groups = {}
    for c in clips:
        if c["stem"] in an_all:
            groups.setdefault((c["start"][:10], clip_start(c["stem"]).hour // 6), []).append(an_all[c["stem"]]["rgb"])
    for s in segs:
        st = clip_start(s["clip"])
        g = groups.get((by[s["clip"]]["start"][:10], st.hour // 6))
        s["wb"] = [1.0, 1.0]
        if g:
            r, gg, b = np.mean(g, axis=0)
            s["wb"] = [round(float(np.clip(1 + 0.6 * (gg / r - 1), 0.8, 1.25)), 3),
                       round(float(np.clip(1 + 0.6 * (gg / b - 1), 0.8, 1.25)), 3)]
    edl = {"title": TITLE, "fps": FPS, "bpm": BPM, "frames": rec, "duration": round(rec / FPS, 3),
           "analysed": sorted(an_all), "act_titles": ACT_TITLES, "segments": segs}
    (HERE / "edl.json").write_text(json.dumps(edl, indent=1))
    write_edl_docs(edl, by)
    print("EDL: %d events, %.2fs (%d frames) -> edl.json, EDL.md, bathroom_revival.edl" % (len(segs), rec / FPS, rec))
    print("     picks: %s" % ("content-aware from LRF analysis" if an_all else "BLIND (no LRF analysis yet)"))


def seg_clock(seg, f):
    D = (seg["frames"] + seg["head"] + seg["tail"]) / FPS
    t = seg["render_in"] + src_time((seg["head"] + f) / FPS, seg["speed"][0], seg["speed"][1], D)
    return clip_start(seg["clip"]) + dt.timedelta(seconds=t)


def write_edl_docs(edl, by):
    segs = edl["segments"]
    L = ["# Edit decision list - %s" % edl["title"], "",
         "%d events - %.2f s (%s) - %d fps - cuts locked to a %g BPM grid (1 beat = %d frames)" %
         (len(segs), edl["duration"], tc(edl["frames"]), FPS, BPM, bf(1)), "",
         "Picks: " + ("content-aware, from LRF proxy analysis of %d clips." % len(edl["analysed"]) if edl["analysed"]
                      else "**blind** - no footage was analysed; in-points are evenly spaced inside each clip. "
                           "Run `edit.py analyze` + `edit.py edl` with the LRFs to replace them."), "",
         "Transitions land centred on the record-in time (dissolve 12 f, whip 6 f, dip-to-white 10 f). "
         "Source times are seconds from the head of each clip; speed is a linear ramp start>end.", ""]
    cm = ["TITLE: %s" % edl["title"].upper(), "FCM: NON-DROP FRAME", ""]
    act = 0
    for s in segs:
        if s["act"] != act:
            act = s["act"]
            L += ["", "## Act %d - %s" % (act, edl["act_titles"][str(act)] if str(act) in edl["act_titles"]
                                           else edl["act_titles"][act]), "",
                  "| # | Rec in | Rec out | Dur | Clip | Src in | Src out | Speed | Transition in | Audio | Clock | Note |",
                  "|--:|---|---|--:|---|--:|--:|---|---|---|---|---|"]
        s0, s1 = s["speed"]
        sp = "%gx" % s0 if s0 == s1 else "%gx > %gx ramp" % (s0, s1)
        num = by[s["clip"]]["num"]
        day = clip_start(s["clip"]).strftime("%a")
        tr = s["transition"] + (" (%d f)" % TRANS_FRAMES[s["transition"]] if TRANS_FRAMES.get(s["transition"]) else "")
        au = {"none": "music", "ambient": "music + nat", "reaction": "nat, music ducked"}[s["audio"]]
        if s["pops"]:
            au += " + pop"
        if s["transition"] in ("whip", "dip", "dissolve"):
            au += " + whoosh"
        L.append("| %d | %s | %s | %.2f | %s %s | %.2f | %.2f | %s | %s | %s | %s | %s |" % (
            s["n"], tc(s["rec_in"]), tc(s["rec_in"] + s["frames"]), s["frames"] / FPS, day, num,
            s["src_in"], s["src_out"], sp, tr, au, clock(seg_clock(s, 0)), s["note"]))
        reel = (day[0] + num).ljust(8)
        ttype = "C        " if s["transition"] in ("cut", "fade-in") else "D    %03d " % TRANS_FRAMES[s["transition"]]
        cm += ["%03d  %s V     %s %s %s %s %s" % (s["n"], reel, ttype, tc(round(s["src_in"] * FPS)),
               tc(round(s["src_out"] * FPS)), tc(s["rec_in"]), tc(s["rec_in"] + s["frames"])),
               "* FROM CLIP NAME: %s.MP4" % s["clip"]]
        if (s0, s1) != (1, 1):
            cm.append("M2   %s       %05.1f                %s" % (reel, FPS * (s0 + s1) / 2, tc(round(s["src_in"] * FPS))))
            cm.append("* SPEED RAMP: %gX TO %gX (LINEAR)" % (s0, s1))
        if s["transition"] not in ("cut", "fade-in"):
            cm.append("* TRANSITION: %s" % s["transition"].upper())
        if s["kind"] == "split":
            cm.append("* SPLIT SCREEN: BEFORE %s.MP4 @ %s" % (s["before"]["clip"], tc(round(s["before"]["src_in"] * FPS))))
    L += ["", "## Source clips", "", "| Clip | File | Starts | Duration | Act |", "|---|---|---|--:|--:|"]
    for c in sorted(by.values(), key=lambda c: c["start"]):
        L.append("| %s %s | `%s.MP4` | %s | %.0f s | %d |" % (clip_start(c["stem"]).strftime("%a"), c["num"], c["stem"],
                 clock(clip_start(c["stem"])), c["duration"], c["act"]))
    (HERE / "EDL.md").write_text("\n".join(L) + "\n")
    (HERE / "bathroom_revival.edl").write_text("\n".join(cm) + "\n")


# ----------------------------------------------------------------------------- generated assets (LUT, music, SFX)
def write_lut(path, n=33):
    """Warm interior grade: lifted shadows, soft shoulder, warm mids/highs, slightly cooler blacks."""
    g = np.linspace(0, 1, n)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")              # .cube order: red varies fastest
    rgb = np.stack([r, gg, b], -1)
    rgb = rgb + 0.045 * (1 - rgb) ** 3                           # lift shadows
    rgb = rgb + 0.06 * np.sin(2 * np.pi * rgb) * -0.5 * 0.5      # gentle S-curve
    y = (rgb * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True)
    rgb = rgb * (1 + (np.array([0.05, 0.005, -0.06]) * (0.35 + 0.65 * y)))   # warmth grows with luminance
    rgb = y + (rgb - y) * 1.06                                   # a touch of saturation
    rgb = np.clip(rgb, 0, 1).reshape(-1, 3)
    with open(path, "w") as f:
        f.write('TITLE "Bathroom Revival warm interior"\nLUT_3D_SIZE %d\n' % n)
        f.writelines("%.5f %.5f %.5f\n" % tuple(x) for x in rgb)


def lowpass(x, hz, hp=False):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    H = 1 / np.sqrt(1 + (f / hz) ** 4)
    return np.fft.irfft(X * (np.sqrt(1 - H ** 2) if hp else H), len(x))


def synth_music(frames, act_frames):
    """Royalty-free placeholder: a procedurally synthesised lo-fi beat on the exact cut grid."""
    rng = np.random.default_rng(7)
    n = int(frames / FPS * SR) + SR
    beat = 60 / BPM
    drums, keys, bass = np.zeros(n), np.zeros(n), np.zeros(n)

    def put(buf, t, x, g=1.0):
        i = int(t * SR)
        if i < n:
            buf[i:i + len(x)] += g * x[:n - i]
    t = np.arange(int(0.3 * SR)) / SR
    kick = np.sin(2 * np.pi * (45 * t + 75 / 18 * (1 - np.exp(-18 * t)))) * np.exp(-9 * t)
    ts = np.arange(int(0.18 * SR)) / SR
    snare = lowpass(rng.standard_normal(len(ts)), 1800, hp=True) * np.exp(-28 * ts) * 0.5 + np.sin(2 * np.pi * 185 * ts) * np.exp(-35 * ts) * 0.5
    th = np.arange(int(0.05 * SR)) / SR
    hat = lowpass(rng.standard_normal(len(th)), 7000, hp=True) * np.exp(-90 * th)
    chords = [(174.61, 220.0, 261.63, 329.63), (164.81, 196.0, 246.94, 293.66),      # Fmaj7  Em7
              (146.83, 174.61, 220.0, 329.63), (130.81, 164.81, 196.0, 246.94)]      # Dm9    Cmaj7
    a2, a3 = act_frames[2] / FPS, act_frames[3] / FPS
    total = frames / FPS
    for bar in range(int(total / (4 * beat)) + 1):
        t0 = bar * 4 * beat
        ch = chords[bar % 4]
        tk = np.arange(int(4.2 * beat * SR)) / SR
        note = sum(np.sin(2 * np.pi * f0 * tk) + 0.3 * np.sin(4 * np.pi * f0 * tk) for f0 in ch)
        put(keys, t0, note * np.exp(-0.9 * tk) * (1 + 0.15 * np.sin(2 * np.pi * 4.5 * tk)) * np.minimum(tk / 0.02, 1), 0.07)
        for step in range(16):
            ts_ = t0 + step * beat / 4 + (0.035 if step % 2 else 0)      # swing the off 16ths
            quiet = ts_ < 4 * beat or a3 <= ts_ < a3 + 4 * beat           # intro and reveal breathe: no drums
            if ts_ >= total - 2 * beat or quiet:
                continue
            soft = 0.75 if ts_ >= a3 else 1.0
            if step in (0, 8, 10) or (step == 7 and ts_ >= a2):
                put(drums, ts_, kick, 0.9 * soft * (0.6 if step == 7 else 1))
            if step in (4, 12):
                put(drums, ts_, snare, 0.55 * soft)
            if step % 2 == 0 or ts_ >= a2:
                put(drums, ts_, hat, (0.16 if step % 4 == 0 else 0.09) * (1.3 if a2 <= ts_ < a3 else 1))
            if step in (0, 10):
                tb = np.arange(int(1.2 * beat * SR)) / SR
                put(bass, ts_, np.sin(2 * np.pi * ch[0] / 2 * tb) * np.exp(-3 * tb) * np.minimum(tb / 0.01, 1), 0.35)
    crackle = (rng.random(n) > 0.9993) * rng.standard_normal(n) * 0.05 + rng.standard_normal(n) * 0.0015
    mix = lowpass(keys, 2200) + lowpass(drums, 9000) + lowpass(bass, 400) + lowpass(crackle, 5000)
    tt = np.arange(n) / SR
    mix *= np.clip((total - tt) / (2 * beat), 0, 1) * np.clip(tt / 0.3, 0, 1)
    return np.stack([mix, mix], 1) * 0.8


def sfx_whoosh(rng):
    n = int(0.42 * SR)
    x = rng.standard_normal(n)
    w = np.linspace(0, 1, n)[:, None]
    lo, hi = lowpass(x, 500), lowpass(x, 2500)
    mono = (lo * (1 - w[:, 0]) + hi * w[:, 0]) * np.hanning(n) ** 2
    return np.stack([mono * (1 - 0.6 * w[:, 0]), mono * (0.4 + 0.6 * w[:, 0])], 1) * 0.25   # pans L>R


def sfx_pop():
    t = np.arange(int(0.07 * SR)) / SR
    x = np.sin(2 * np.pi * (300 * t + 600 / 60 * (1 - np.exp(-60 * t)))) * np.exp(-55 * t)
    x[:96] += np.random.default_rng(3).standard_normal(96) * np.exp(-np.arange(96) / 20) * 0.3   # snap click
    return np.stack([x, x], 1) * 0.3


def nat_audio(path, start, dur):
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.3f" % start, "-t", "%.3f" % dur, "-i", path, "-vn",
                        "-ac", "2", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
    x = np.frombuffer(p.stdout, np.int16).astype(np.float32).reshape(-1, 2) / 32768
    return x


def build_audio(edl, by, music_path, music_offset, out_wav):
    frames = edl["frames"]
    n = int(frames / FPS * SR)
    act_frames = {s["act"]: s["rec_in"] for s in reversed(edl["segments"])}
    if music_path:
        raw = run(["ffmpeg", "-v", "error", "-ss", "%.3f" % music_offset, "-i", music_path, "-vn", "-ac", "2",
                   "-ar", str(SR), "-f", "s16le", "-"])
        music = np.frombuffer(raw, np.int16).astype(np.float32).reshape(-1, 2) / 32768 * 0.7
        music = np.pad(music, ((0, max(0, n - len(music))), (0, 0)))[:n]
        music *= np.clip((n - np.arange(n)) / (1.3 * SR), 0, 1)[:, None]
    else:
        music = synth_music(frames, act_frames)[:n]
    fx, nat, duck = np.zeros((n, 2)), np.zeros((n, 2)), np.ones(n)
    rng = np.random.default_rng(11)
    pop = sfx_pop()

    def put(buf, frame, x, g=1.0):
        i = max(0, int(frame / FPS * SR))
        x = x[:max(0, n - i)]
        buf[i:i + len(x)] += g * x
    for s in edl["segments"]:
        if s["transition"] in ("whip", "dip", "dissolve"):       # whoosh peaks on the cut
            put(fx, s["rec_in"] - 0.22 * FPS, sfx_whoosh(rng), 1.0 if s["transition"] != "dissolve" else 0.45)
        for p in s["pops"]:
            put(fx, s["rec_in"] + p, pop)
        if s["audio"] != "none":
            src = by[s["clip"]].get("mp4") or by[s["clip"]].get("lrf") or {}
            if src.get("path") and src.get("audio"):
                x = nat_audio(src["path"], s["src_in"], s["frames"] / FPS)
                if len(x):
                    e = np.minimum(1, np.minimum(np.arange(len(x)), len(x) - np.arange(len(x))) / (0.04 * SR))
                    put(nat, s["rec_in"], x * e[:, None], 1.0 if s["audio"] == "reaction" else 0.12)
            if s["audio"] == "reaction":                         # duck the music -12 dB under the voice
                i0, i1 = int(s["rec_in"] / FPS * SR), int((s["rec_in"] + s["frames"]) / FPS * SR)
                duck[i0:i1] = 0.25
    k = np.hanning(int(0.35 * SR))
    duck = np.convolve(duck, k / k.sum(), "same")
    mix = music * duck[:, None] + fx + nat
    mix = np.tanh(mix * 1.1) / 1.1                               # soft limiter
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())


# ----------------------------------------------------------------------------- overlays (titles, lower-thirds, clock)
FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts", "~/.local/share/fonts", "/Library/Fonts",
             "/System/Library/Fonts", "~/Library/Fonts", "C:/Windows/Fonts"]
FONT_PREFS = {"bold": ["Montserrat-Bold", "Inter-Bold", "InterDisplay-Bold", "HelveticaNeue", "Arial Bold", "arialbd",
                       "DejaVuSans-Bold"],
              "regular": ["Montserrat-Medium", "Montserrat-Regular", "Inter-Medium", "Inter-Regular", "Helvetica",
                          "Arial", "arial", "DejaVuSans"]}


def find_font(kind, override=None):
    if override:
        return override
    files = [p for d in FONT_DIRS if Path(d).expanduser().is_dir() for p in Path(d).expanduser().rglob("*.[ot]t[fc]")]
    for name in FONT_PREFS[kind]:
        for p in files:
            if p.stem.lower() == name.lower():
                return str(p)
    sys.exit("no usable %s font found - pass --font-bold / --font-regular" % kind)


class Overlay:
    """Renders the motion-graphics layer as RGBA frames; identical frames are cached."""

    def __init__(self, edl, by, W, H, fb, fr):
        self.edl, self.W, self.H, self.vert = edl, W, H, H > W
        self.u = min(W, H) / 1080.0
        self.f = lambda kind, px: ImageFont.truetype(fb if kind == "b" else fr, max(8, int(px * self.u)))
        self.segs = edl["segments"]
        self.acts = {}
        for s in self.segs:
            self.acts.setdefault(s["act"], s["rec_in"])
        self.split = next((s for s in self.segs if s["kind"] == "split"), None)
        self.cache = {}
        self.margin = int(64 * self.u)
        self.top = int((0.085 * H) if self.vert else 54 * self.u)       # keep clear of reel/shorts UI
        self.lower_y = int(H * (0.70 if self.vert else 0.80))

    def state(self, f):
        seg = next(s for s in reversed(self.segs) if s["rec_in"] <= f)
        st = {"clock": clock(seg_clock(seg, f - seg["rec_in"]))}
        q = lambda x: round(max(0.0, min(1.0, x)) * 12) / 12
        st["black"] = q(max(1 - f / 12, (f - (self.edl["frames"] - 24)) / 20))
        st["title"] = q(min(f / 10, (bf(4) - 6 - f) / 10))
        for act, a0 in self.acts.items():
            a = a0 + (bf(4) if act == 1 else 0)
            card, low = q(min((f - a) / 6, (a + 46 - f) / 8)), q(min((f - a - 48) / 8, (a + 130 - f) / 10))
            if card > 0:
                st["card"] = (act, card)
            if low > 0:
                st["lower"] = (act, low)
        if self.split and self.split["rec_in"] <= f < self.split["rec_in"] + self.split["frames"]:
            st["split"] = q(min((f - self.split["rec_in"]) / 10, (self.split["rec_in"] + self.split["frames"] - f) / 10))
        return st

    def text(self, img, xy, s, font, alpha=1.0, fill=(255, 255, 255), anchor="la", shadow=True):
        d = ImageDraw.Draw(img)
        if shadow:
            o = max(1, int(2 * self.u))
            d.text((xy[0] + o, xy[1] + o), s, font=font, fill=(0, 0, 0, int(110 * alpha)), anchor=anchor)
        d.text(xy, s, font=font, fill=fill + (int(255 * alpha),), anchor=anchor)

    def render(self, st):
        W, H, u = self.W, self.H, self.u
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if st.get("title", 0) > 0 or "card" in st:               # soft scrim so centred titles stay readable
            a = st.get("title", 0) or st["card"][1]
            img = Image.alpha_composite(img, Image.new("RGBA", (W, H), (12, 10, 8, int(95 * a))))
            d = ImageDraw.Draw(img)
        if st.get("title", 0) > 0:
            a = st["title"]
            self.text(img, (W / 2, H * 0.46), TITLE.upper(), self.f("b", 64 if self.vert else 84), a, anchor="mm")
            d.line([(W / 2 - 60 * u, H * 0.46 + 70 * u), (W / 2 + 60 * u, H * 0.46 + 70 * u)],
                   fill=ACCENT + (int(255 * a),), width=max(2, int(4 * u)))
            self.text(img, (W / 2, H * 0.46 + 112 * u), SUBTITLE.upper(), self.f("r", 34), a, anchor="mm")
        if "card" in st:
            act, a = st["card"]
            self.text(img, (W / 2, H * 0.47), ACT_TITLES[act], self.f("b", 78 if self.vert else 96), a, anchor="mm")
            d.line([(W / 2 - 50 * u, H * 0.47 + 74 * u), (W / 2 + 50 * u, H * 0.47 + 74 * u)],
                   fill=ACCENT + (int(255 * a),), width=max(2, int(4 * u)))
        if "lower" in st:
            act, a = st["lower"]
            x, y = self.margin - int((1 - a) * 24 * u), self.lower_y
            day = clip_start(next(s for s in self.segs if s["act"] == act)["clip"]).strftime("%A, %B %-d")
            f1, f2 = self.f("b", 44), self.f("r", 28)
            w = max(d.textlength(ACT_TITLES[act], font=f1), d.textlength(day, font=f2))
            d.rounded_rectangle([x - 22 * u, y - 18 * u, x + w + 46 * u, y + 104 * u], radius=int(10 * u),
                                fill=(14, 12, 10, int(150 * a)))
            d.rectangle([x, y, x + 6 * u, y + 86 * u], fill=ACCENT + (int(255 * a),))
            self.text(img, (x + 24 * u, y - 4 * u), ACT_TITLES[act], f1, a, shadow=False)
            self.text(img, (x + 24 * u, y + 52 * u), day, f2, a, fill=(235, 228, 216), shadow=False)
        if "split" in st:
            a, f1, f2 = st["split"], self.f("b", 40), self.f("r", 26)
            first, last = self.segs[0], self.split
            labels = [("BEFORE", clock(clip_start(self.split["before"]["clip"]) +
                                 dt.timedelta(seconds=self.split["before"]["src_in"]))),
                      ("AFTER", clock(seg_clock(last, 0)))]
            for k, (lab, when) in enumerate(labels):
                if self.vert:
                    x, y = self.margin, int(H * (0.13 + 0.5 * k)) if k == 0 else int(H * 0.53)
                    d.line([(0, H / 2), (W, H / 2)], fill=(255, 255, 255, int(255 * a)), width=max(2, int(5 * u)))
                else:
                    x, y = self.margin + k * W // 2, int(H * 0.80)
                    d.line([(W / 2, 0), (W / 2, H)], fill=(255, 255, 255, int(255 * a)), width=max(2, int(5 * u)))
                tw = max(d.textlength(lab, font=f1), d.textlength(when, font=f2))
                d.rounded_rectangle([x - 18 * u, y - 12 * u, x + tw + 22 * u, y + 92 * u], radius=int(10 * u),
                                    fill=(14, 12, 10, int(150 * a)))
                self.text(img, (x, y), lab, f1, a, fill=ACCENT if k else (255, 255, 255), shadow=False)
                self.text(img, (x, y + 50 * u), when, f2, a, fill=(235, 228, 216), shadow=False)
        # persistent elapsed-time indicator, upper right: "Friday 11:40 AM"
        day, when = st["clock"].split(" ", 1)
        fd, ft = self.f("b", 30), self.f("r", 30)
        wd, wt = d.textlength(day + "  ", font=fd), d.textlength(when, font=ft)
        widest = d.textlength("Wednesday  ", font=fd) + d.textlength("12:00 AM", font=ft)   # pill never jitters
        x1, y0 = W - self.margin, self.top
        x0 = x1 - widest - 76 * u
        d.rounded_rectangle([x0, y0, x1, y0 + 58 * u], radius=int(29 * u), fill=(14, 12, 10, 150))
        d.ellipse([x0 + 22 * u, y0 + 22 * u, x0 + 36 * u, y0 + 36 * u], fill=ACCENT + (255,))
        tx = x1 - 24 * u - wd - wt
        self.text(img, (tx, y0 + 29 * u), day, fd, anchor="lm", shadow=False)
        self.text(img, (tx + wd, y0 + 29 * u), when, ft, fill=(235, 228, 216), anchor="lm", shadow=False)
        if st["black"] > 0:
            img = Image.alpha_composite(img, Image.new("RGBA", (W, H), (0, 0, 0, int(255 * st["black"]))))
        return img.tobytes()

    def frame(self, f):
        st = self.state(f)
        key = json.dumps(st, sort_keys=True)
        if key not in self.cache:
            if len(self.cache) > 24:
                self.cache.clear()
            self.cache[key] = self.render(st)
        return self.cache[key]


# ----------------------------------------------------------------------------- render (conform to masters)
def source(clip, proxy):
    key = "lrf" if proxy else "mp4"
    if not clip.get(key, {}).get("path"):
        sys.exit("missing %s for %s - put the file in the media folder and re-run `scan`%s" % (
            key.upper(), clip["stem"], "" if proxy else " (or use --proxy to preview from the LRF)"))
    return clip[key]["path"]


def seg_filter(seg, W, H, lut, input_lut, speed_ramp=True, fx=None):
    D = (seg["frames"] + seg["head"] + seg["tail"]) / FPS
    s0, s1 = seg["speed"] if speed_ramp else (1, 1)
    if abs(s1 - s0) < 1e-6:
        pts = "setpts=(PTS-STARTPTS)/%g" % s0
    else:   # invert the ramp: output time for each source timestamp
        k = (s1 - s0) / D
        pts = "setpts=((sqrt(%g+%g*(PTS-STARTPTS)*TB)-%g)/%g)/TB" % (s0 * s0, 2 * k, s0, k)
    fx = seg["fx"] if fx is None else fx
    wr, wb = seg.get("wb", [1, 1])
    f = [pts, "fps=%d" % FPS, "scale=%d:%d:force_original_aspect_ratio=increase:flags=lanczos" % (W, H),
         "crop=%d:%d:(iw-%d)*%g:(ih-%d)/2" % (W, H, W, fx, H)]
    if input_lut:                                # e.g. DJI D-Log M -> Rec.709 conversion LUT
        f.append("lut3d='%s'" % input_lut)
    if (wr, wb) != (1, 1):
        f.append("colorchannelmixer=rr=%g:bb=%g" % (wr, wb))
    f += ["lut3d='%s'" % lut, "format=yuv420p"]
    return ",".join(f)


def render_segments(edl, by, a, W, H, tag, lut):
    seg_dir = a.work / ("seg_" + tag)
    seg_dir.mkdir(parents=True, exist_ok=True)
    enc = ["-c:v", "libx264", "-preset", "veryfast" if a.proxy else "fast", "-crf", "20" if a.proxy else "14",
           "-pix_fmt", "yuv420p", "-an"]
    files = []
    for s in edl["segments"]:
        nfr = s["frames"] + s["head"] + s["tail"]
        D = nfr / FPS
        span = src_time(D, s["speed"][0], s["speed"][1], D)
        src = source(by[s["clip"]], a.proxy)
        key = hashlib.md5(json.dumps([s, W, H, src, a.input_lut, os.path.getmtime(src)], sort_keys=True).encode()).hexdigest()[:10]
        out = seg_dir / ("%03d_%s.mp4" % (s["n"], key))
        files.append(out)
        if out.exists():
            continue
        print("  segment %02d/%d  %s  %s" % (s["n"], len(edl["segments"]), by[s["clip"]]["num"], s["note"]), flush=True)
        if s["kind"] == "split":
            b = s["before"]
            vert = H > W
            hw, hh = (W, H // 2) if vert else (W // 2, H)
            bseg = dict(s, wb=next(x["wb"] for x in edl["segments"] if x["clip"] == b["clip"]))
            fc = "[0:v]%s[b];[1:v]%s[a];[b][a]%s=inputs=2[v]" % (
                seg_filter(bseg, hw, hh, lut, a.input_lut, fx=b["fx"]), seg_filter(s, hw, hh, lut, a.input_lut),
                "vstack" if vert else "hstack")
            run(["ffmpeg", "-v", "error", "-y", "-ss", "%.3f" % b["src_in"], "-t", "%.3f" % (D + 1), "-i",
                 source(by[b["clip"]], a.proxy), "-ss", "%.3f" % s["render_in"], "-t", "%.3f" % (D + 1), "-i", src,
                 "-filter_complex", fc, "-map", "[v]", "-frames:v", nfr] + enc + [out])
        else:
            run(["ffmpeg", "-v", "error", "-y", "-ss", "%.3f" % s["render_in"], "-t", "%.3f" % (span + 1), "-i", src,
                 "-vf", seg_filter(s, W, H, lut, a.input_lut), "-frames:v", nfr] + enc + [out])
    return files


def assemble(edl, by, files, a, W, H, tag, wav):
    segs = edl["segments"]
    runs = [[0]]                                    # hard cuts concat; everything else becomes an xfade between runs
    for i in range(1, len(segs)):
        runs.append([i]) if TRANS_FRAMES.get(segs[i]["transition"]) else runs[-1].append(i)
    fc, labels, lens = [], [], []
    for r, idx in enumerate(runs):
        ins = "".join("[%d:v]" % i for i in idx)
        fc.append("%sconcat=n=%d:v=1:a=0,fps=%d,format=yuv420p[r%d]" % (ins, len(idx), FPS, r))
        labels.append("[r%d]" % r)
        lens.append(sum(segs[i]["frames"] + segs[i]["head"] + segs[i]["tail"] for i in idx))
    cur, acc = labels[0], lens[0]
    for r in range(1, len(runs)):
        t = segs[runs[r][0]]["transition"]
        d = TRANS_FRAMES[t]
        fc.append("%s%sxfade=transition=%s:duration=%.5f:offset=%.5f[x%d]" % (cur, labels[r], XFADE[t], d / FPS,
                                                                           (acc - d) / FPS, r))
        cur, acc = "[x%d]" % r, acc + lens[r] - d
    assert acc == edl["frames"], (acc, edl["frames"])
    ov, au = len(files), len(files) + 1
    fc.append("%s[%d:v]overlay=format=auto,format=yuv420p[v]" % (cur, ov))
    out = a.out / ("bathroom_revival_%s%s.mp4" % (tag, "_proxy" if a.proxy else ""))
    a.out.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for f in files:
        cmd += ["-i", str(f)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "pipe:0",
            "-i", str(wav), "-filter_complex", ";".join(fc), "-map", "[v]", "-map", "%d:a" % au,
            "-frames:v", str(edl["frames"]), "-c:v", "libx264", "-preset", "veryfast" if a.proxy else "slow",
            "-crf", "21" if a.proxy else "17", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    o = Overlay(edl, by, W, H, find_font("bold", a.font_bold), find_font("regular", a.font_regular))
    try:
        for f in range(edl["frames"]):
            p.stdin.write(o.frame(f))
        p.stdin.close()
    except BrokenPipeError:
        pass
    err = p.stderr.read().decode(errors="replace")
    if p.wait():
        sys.exit("final assembly failed:\n" + err[-3000:])
    print("wrote %s  (%dx%d, %.2fs)" % (out, W, H, edl["frames"] / FPS))


def cmd_render(a):
    if not (HERE / "edl.json").exists():
        cmd_edl(a)
    edl = json.loads((HERE / "edl.json").read_text())
    by = {c["stem"]: c for c in load_manifest(a)}
    a.work.mkdir(parents=True, exist_ok=True)
    lut = a.work / "warm_interior.cube"
    write_lut(lut)
    wav = a.work / "mix.wav"
    build_audio(edl, by, a.music, a.music_offset, wav)
    for c in by.values():
        if (c.get("mp4") or {}).get("transfer") in ("arib-std-b67", "smpte2084") and not a.input_lut:
            print("note: %s is HDR (%s) - pass --input-lut with a to-Rec.709 LUT" % (c["stem"], c["mp4"]["transfer"]))
    short = 540 if a.proxy else a.height
    for asp in (["16:9", "9:16"] if a.aspect == "both" else [a.aspect]):
        W, H = (short * 16 // 9, short) if asp == "16:9" else (short, short * 16 // 9)
        tag = asp.replace(":", "x")
        print("render %s  %dx%d  from %s" % (asp, W, H, "LRF proxies (preview)" if a.proxy else "MP4 masters"))
        assemble(edl, by, render_segments(edl, by, a, W, H, tag, lut), a, W, H, tag, wav)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "analyze", "edl", "render", "all"])
    ap.add_argument("--media", default=str(HERE / "media"), help="folder holding the DJI .MP4 and .LRF files")
    ap.add_argument("--work", type=Path, default=HERE / "build", help="cache / intermediates")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--aspect", choices=["16:9", "9:16", "both"], default="16:9")
    ap.add_argument("--height", type=int, default=1080, help="short-side pixels: 1080 (HD) or 2160 (4K)")
    ap.add_argument("--proxy", action="store_true", help="fast 540p preview rendered from the LRFs")
    ap.add_argument("--music", help="licensed beat track to use instead of the built-in synth")
    ap.add_argument("--music-offset", type=float, default=0.0, help="seconds into --music where beat 1 falls")
    ap.add_argument("--bpm", type=float, help="tempo of --music (re-run `edl` so cuts follow it)")
    ap.add_argument("--exclude", help="comma-separated clips to leave out, matched against the filename, "
                    "e.g. 112633_0196,0198 (applies at `scan`; re-run `edl` afterwards)")
    ap.add_argument("--input-lut", help="optional .cube applied before the grade (e.g. D-Log M to Rec.709)")
    ap.add_argument("--font-bold")
    ap.add_argument("--font-regular")
    a = ap.parse_args()
    if a.bpm:
        global BPM
        BPM = a.bpm
    if a.cmd in ("scan", "all"):
        cmd_scan(a)
    if a.cmd in ("analyze", "all"):
        cmd_analyze(a)
    if a.cmd in ("edl", "all"):
        cmd_edl(a)
    if a.cmd in ("render", "all"):
        cmd_render(a)


if __name__ == "__main__":
    main()
