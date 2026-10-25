# -*- coding: utf-8 -*-
"""Privacy Camera Studio: local Gradio face registration + privacy camera.
Run: python app.py   (rename this downloaded file to app.py)
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

BASE = Path(__file__).resolve().parent
KNOWN = BASE / "known_faces"
MODELS = BASE / "insightface_data"
EMOJI_DIR = BASE / "images"
LANDMARKER = BASE / "models" / "face_landmarker.task"
KNOWN.mkdir(exist_ok=True)
MODEL_LOCK = threading.RLock()
FACE_APP = None
LANDMARKER_MODEL = None
EMOJIS = {}
FACE_DATABASE = []
DIRECTIONS = ("front", "right", "left")
FACE_SIZE = 35


def app_model():
    global FACE_APP
    with MODEL_LOCK:
        if FACE_APP is None:
            # CoreMLの初期化失敗によるフォールバック警告を防ぐため、CPUExecutionProviderに固定します
            FACE_APP = FaceAnalysis(
                name="buffalo_s", 
                root=str(MODELS), 
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"]
            )
            FACE_APP.prepare(ctx_id=-1, det_size=(320, 320))
    return FACE_APP


def get_faces(image_bgr):
    with MODEL_LOCK:
        return app_model().get(image_bgr)


def read_image(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def refresh_faces():
    global FACE_DATABASE
    entries = []
    for person in sorted(KNOWN.iterdir()):
        paths = (
            sorted(
                p
                for p in person.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if person.is_dir()
            else (
                [person] if person.suffix.lower() in (".jpg", ".jpeg", ".png") else []
            )
        )
        for p in paths:
            im = read_image(p)
            if im is None:
                continue
            faces = get_faces(im)
            if len(faces) != 1 or getattr(faces[0], "normed_embedding", None) is None:
                continue
            entries.append(
                {
                    "name": person.name if person.is_dir() else p.stem,
                    "embedding": faces[0].normed_embedding.copy(),
                    "image": im,
                }
            )
    FACE_DATABASE = entries
    return entries


def names_text():
    counts = {}
    for entry in FACE_DATABASE:
        counts[entry["name"]] = counts.get(entry["name"], 0) + 1
    if not counts:
        return "登録者なし。先に顔登録を行ってください。"
    return "\n".join(f"● {name} — {n}枚の顔画像" for name, n in counts.items())


def quality(frame):
    return float(
        cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    )


def nose_offset(face):
    pts = getattr(face, "kps", None)
    if pts is None or len(pts) < 3:
        return None
    eye_mid = (float(pts[0][0]) + float(pts[1][0])) / 2
    distance = abs(float(pts[1][0] - pts[0][0]))
    return (float(pts[2][0]) - eye_mid) / distance if distance >= 3 else None


def crop_face(frame, box, pad=0.20):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(x) for x in box]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    a, b, c, d = (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(w, int(x2 + px)),
        min(h, int(y2 + py)),
    )
    return frame[b:d, a:c].copy() if a < c and b < d else None


def inspect_photo(photo, direction):
    labels = {"front": "正面", "right": "右向き", "left": "左向き"}
    label = labels[direction]
    if photo is None:
        return None, f"⚠️ {label}の写真がありません。撮影してください。"
    arr = np.asarray(photo)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None, f"⚠️ {label}の画像形式を確認してください。"
    frame = cv2.cvtColor(
        np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2BGR
    )
    h, w = frame.shape[:2]
    candidates = []
    for f in get_faces(frame):
        score = float(getattr(f, "det_score", 0.0))
        x1, y1, x2, y2 = map(float, f.bbox)
        fw, fh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ellipse_distance = ((cx - w / 2) / (w * 0.27)) ** 2 + (
            (cy - h / 2) / (h * 0.46)
        ) ** 2
        if (
            score >= 0.60
            and fw >= FACE_SIZE
            and fh >= FACE_SIZE
            and ellipse_distance <= 1.0
        ):
            candidates.append((fw * fh, f))
    if not candidates:
        return (
            None,
            f"⚠️ {label}：楕円内に顔を検出できません。正面を向き、明るい場所で少し近づいてください。",
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] >= candidates[0][0] * 0.45:
        return (
            None,
            f"⚠️ {label}：楕円内に複数の顔を検出しました。ほかの人が映らないようにしてください。",
        )
    face = candidates[0][1]
    x1, y1, x2, y2 = face.bbox
    if min(x2 - x1, y2 - y1) < FACE_SIZE:
        return None, f"⚠️ {label}：顔が小さすぎます。カメラに近づいてください。"
    crop = crop_face(frame, face.bbox)
    if crop is None or quality(crop) < 8:
        return None, f"⚠️ {label}：写真がぼけています。明るい場所で撮り直してください。"
    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        return None, f"⚠️ {label}：顔の特徴を抽出できませんでした。"
    return {
        "image": crop,
        "embedding": embedding.copy(),
        "offset": nose_offset(face),
    }, None


def register(name, front, right, left, progress=gr.Progress()):
    global FACE_DATABASE
    name = (name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        return (
            "⚠️ 登録名は半角英数字・_・- の1〜40文字にしてください。",
            [],
            names_text(),
        )

    photos = {"front": front, "right": right, "left": left}
    samples = {}
    for i, (direction, photo) in enumerate(photos.items()):
        progress(i / 4, desc=f"{direction} の写真を確認中")
        sample, error = inspect_photo(photo, direction)
        if error:
            return error + " 他の撮影済み写真はそのまま使えます。", [], names_text()
        samples[direction] = sample

    offsets = [samples[d]["offset"] for d in DIRECTIONS]
    guidance = ""
    if all(v is not None for v in offsets):
        f, r, l = offsets
        if abs(r - f) < 0.055 or abs(l - f) < 0.055 or (r - f) * (l - f) >= 0:
            guidance = " ⚠️ 顔の左右の向きの差が小さい可能性があります。写真を確認し、必要なら撮り直してください。"

    folder = KNOWN / name
    folder.mkdir(parents=True, exist_ok=True)
    gallery = []
    new_entries = []
    labels = {"front": "① 正面", "right": "② 右向き", "left": "③ 左向き"}
    for i, direction in enumerate(DIRECTIONS):
        sample = samples[direction]
        target = folder / f"{direction}.png"
        Image.fromarray(cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB)).save(target)
        gallery.append((str(target), labels[direction]))
        new_entries.append(
            {"name": name, "embedding": sample["embedding"], "image": sample["image"]}
        )
        progress(0.75 + 0.25 * (i + 1) / 3, desc="保存中")
    with MODEL_LOCK:
        FACE_DATABASE = [e for e in FACE_DATABASE if e["name"] != name] + new_entries
    return (
        f"✅ {name} の登録が完了しました（3/3枚）。" + guidance,
        gallery,
        names_text(),
    )


def capture_state():
    return {"index": 0, "samples": {}, "photos": {}, "name": ""}


def capture_once(name, raw, state):
    global FACE_DATABASE
    if not isinstance(state, dict):
        state = capture_state()
    name = (name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        return (
            state,
            "登録名は半角英数字・_・- の1〜40文字にしてください。",
            [],
            names_text(),
            gr.update(value="📸 正面を撮影"),
        )
    if state.get("name") and state["name"] != name and state["index"]:
        return (
            state,
            "撮影途中で名前を変更できません。「最初から撮影」を押してください。",
            [(v, CAPTURE_LABELS[i]) for i, v in sorted(state["photos"].items())],
            names_text(),
            gr.update(),
        )
    direction_index = int(state.get("index", 0))
    if direction_index >= 3:
        return (
            state,
            "登録済みです。別の登録には「最初から撮影」を押してください。",
            [(v, CAPTURE_LABELS[i]) for i, v in sorted(state["photos"].items())],
            names_text(),
            gr.update(value="✅ 登録完了", interactive=False),
        )
    if raw is None:
        return (
            state,
            "カメラ映像がありません。カメラをオンにしてください。",
            [],
            names_text(),
            gr.update(),
        )
    direction = DIRECTIONS[direction_index]
    sample, error = inspect_photo(raw, direction)
    if error:
        return (
            state,
            error + " 同じ方向でもう一度撮影してください。",
            [(v, CAPTURE_LABELS[i]) for i, v in sorted(state["photos"].items())],
            names_text(),
            gr.update(),
        )
    state["name"] = name
    state["samples"][direction] = sample
    state["photos"][direction_index] = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB)
    state["index"] = direction_index + 1
    gallery = [(v, CAPTURE_LABELS[i]) for i, v in sorted(state["photos"].items())]
    if state["index"] < 3:
        next_label = CAPTURE_LABELS[state["index"]]
        return (
            state,
            f"✅ {CAPTURE_LABELS[direction_index]}を取得（{state['index']}/3）。次は **{next_label}** を撮影してください。",
            gallery,
            names_text(),
            gr.update(value=f"📸 {next_label}を撮影"),
        )
    folder = KNOWN / name
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for d in DIRECTIONS:
        sample = state["samples"][d]
        target = folder / f"{d}.png"
        Image.fromarray(cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB)).save(target)
        entries.append(
            {"name": name, "embedding": sample["embedding"], "image": sample["image"]}
        )
    with MODEL_LOCK:
        FACE_DATABASE = [e for e in FACE_DATABASE if e["name"] != name] + entries
    return (
        state,
        f"✅ **{name} の顔登録が完了しました！** 3回の撮影だけで3方向を保存しました。",
        gallery,
        names_text(),
        gr.update(value="✅ 登録完了", interactive=False),
    )


CAPTURE_LABELS = ("正面", "右向き", "左向き")


def reset_capture():
    return (
        capture_state(),
        "最初から撮影できます。正面を楕円に合わせてください。",
        [],
        gr.update(value="📸 正面を撮影", interactive=True),
    )


def process_demo_frame(
    frame_rgb, state, threshold, auto_expression, consent, strict_privacy
):
    rgb, clean_rgb, state, stat = process_frame(
        frame_rgb,
        state,
        threshold,
        auto_expression,
        consent,
        strict_privacy,
        include_clean=True,
    )
    if clean_rgb is not None:
        cv2.imshow("Clean Live Stream (TargetShield)", cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    return rgb, clean_rgb, state, stat


def clip_box(box, shape, margin=0.18):
    h, w = shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    px, py = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(w, int(x2 + px)),
        min(h, int(y2 + py)),
    )


def mosaic(frame, box):
    x1, y1, x2, y2 = clip_box(box, frame.shape, 0.24)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    tiny = cv2.resize(roi, (max(1, (x2 - x1) // 18), max(1, (y2 - y1) // 18)))
    roi[:] = cv2.resize(tiny, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def obscure(frame, box):
    x1, y1, x2, y2 = clip_box(box, frame.shape, 0.25)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = (18, 23, 34)


def emoji_art(expression):
    s = 180
    a = np.zeros((s, s, 4), dtype=np.uint8)
    cv2.circle(a, (90, 90), 84, (24, 210, 250, 255), -1, cv2.LINE_AA)
    cv2.circle(a, (59, 70), 9, (28, 48, 72, 255), -1, cv2.LINE_AA)
    cv2.circle(a, (121, 70), 9, (28, 48, 72, 255), -1, cv2.LINE_AA)
    if expression == "smile":
        cv2.ellipse(
            a, (90, 104), (37, 25), 0, 0, 180, (28, 48, 72, 255), 6, cv2.LINE_AA
        )
    elif expression == "tired":
        cv2.ellipse(a, (59, 70), (18, 3), 0, 0, 180, (28, 48, 72, 255), 5, cv2.LINE_AA)
        cv2.ellipse(a, (121, 70), (18, 3), 0, 0, 180, (28, 48, 72, 255), 5, cv2.LINE_AA)
        cv2.ellipse(
            a, (90, 113), (14, 20), 0, 0, 360, (28, 48, 72, 255), 5, cv2.LINE_AA
        )
    else:
        cv2.line(a, (65, 115), (115, 115), (28, 48, 72, 255), 5, cv2.LINE_AA)
    return a


def emoji_image(expression):
    if expression not in EMOJIS:
        paths = {
            "smile": "emoji_1-circle.png",
            "tired": "emoji_2-circle.png",
            "natural": "emoji_3-circle.png",
        }
        p = EMOJI_DIR / paths[expression]
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.is_file() else None
        EMOJIS[expression] = (
            img
            if img is not None and img.ndim == 3 and img.shape[2] == 4
            else emoji_art(expression)
        )
    return EMOJIS[expression]


def overlay(frame, rgba, box):
    x1, y1, x2, y2 = clip_box(box, frame.shape, 0.24)
    if x1 >= x2 or y1 >= y2:
        return
    roi = frame[y1:y2, x1:x2]
    resized = cv2.resize(rgba, (x2 - x1, y2 - y1))
    alpha = resized[:, :, 3:4].astype(np.float32) / 255
    roi[:] = (alpha * resized[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)


def replace_with_registered_face(frame, box, registered):
    if not registered:
        mosaic(frame, box)
        return
    source = registered[0].get("image")
    if source is None or source.size == 0:
        mosaic(frame, box)
        return
    x1, y1, x2, y2 = clip_box(box, frame.shape, 0.10)
    if x2 <= x1 or y2 <= y1:
        return
    frame[y1:y2, x1:x2] = cv2.resize(
        source, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR
    )


def expression_model():
    global LANDMARKER_MODEL
    if not LANDMARKER.is_file():
        return None
    if LANDMARKER_MODEL is None:
        import mediapipe as mp

        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=5,
            output_face_blendshapes=True,
        )
        LANDMARKER_MODEL = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    return LANDMARKER_MODEL


def expression_faces(frame):
    try:
        lm = expression_model()
        if lm is None:
            return []
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        with MODEL_LOCK:
            result = lm.detect(image)
        h, w = frame.shape[:2]
        out = []
        for pts, shapes in zip(result.face_landmarks, result.face_blendshapes):
            values = {x.category_name: float(x.score) for x in shapes}
            smile = (
                values.get("mouthSmileLeft", 0) + values.get("mouthSmileRight", 0)
            ) / 2
            tired = (
                min(values.get("eyeBlinkLeft", 0), values.get("eyeBlinkRight", 0))
                >= 0.65
                or values.get("jawOpen", 0) >= 0.65
            )
            expr = "smile" if smile >= 0.45 else "tired" if tired else "natural"
            xs = [pt.x * w for pt in pts]
            ys = [pt.y * h for pt in pts]
            out.append(((min(xs), min(ys), max(xs), max(ys)), expr))
        return out
    except (ImportError, RuntimeError, ValueError) as e:
        print("Expression analysis unavailable:", e)
        return []


def overlap(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area2 = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1, area1 + area2 - inter)


def initial_state():
    return {
        "tracks": {},
        "next_id": 1,
        "frame": 0,
        "commands": {},
        "global_mode": "mosaic",
        "hitboxes": [],
        "show_global_controls": True,
    }


def update_tracking(detections, state):
    tracks = state["tracks"]
    frame_no = state["frame"]
    candidates = []
    for j, det in enumerate(detections):
        for tid, t in tracks.items():
            if frame_no - t["last"] > 20:
                continue
            sim = float(np.dot(det["emb"], t["emb"]))
            ov = overlap(det["box"], t["box"])
            cx = (det["box"][0] + det["box"][2]) / 2
            cy = (det["box"][1] + det["box"][3]) / 2
            tx = (t["box"][0] + t["box"][2]) / 2
            ty = (t["box"][1] + t["box"][3]) / 2
            d = np.hypot(cx - tx, cy - ty)
            size = max(det["box"][2] - det["box"][0], det["box"][3] - det["box"][1], 1)
            if sim > 0.48 and (ov > 0.05 or d < size * 1.5):
                candidates.append((sim + 0.15 * ov - 0.03 * d / size, tid, j))
    candidates.sort(reverse=True)
    assigned = {}
    used_tracks = set()
    for score, tid, j in candidates:
        if j not in assigned and tid not in used_tracks:
            assigned[j] = tid
            used_tracks.add(tid)
    for j, det in enumerate(detections):
        if j not in assigned:
            tid = state["next_id"]
            state["next_id"] += 1
            assigned[j] = tid
            tracks[tid] = {
                "mode": state["global_mode"],
                "expr": "natural",
                "candidate": "natural",
                "stable": 0,
            }
        tid = assigned[j]
        t = tracks[tid]
        t["emb"] = det["emb"].copy()
        t["box"] = det["box"]
        t["last"] = frame_no
        if tid in state["commands"]:
            t["mode"] = state["commands"][tid]
        det["id"] = tid
        det["track"] = t
    for tid in list(tracks):
        if frame_no - tracks[tid]["last"] > 30:
            del tracks[tid]
            state["commands"].pop(tid, None)
    return detections


def process_frame(
    frame_rgb,
    state,
    threshold,
    auto_expression,
    consent,
    strict_privacy,
    include_clean=False,
):
    if frame_rgb is None:
        if include_clean:
            return None, None, state, "カメラ映像待ち"
        return None, state, "カメラ映像待ち"
    if not isinstance(state, dict):
        state = initial_state()
    state["frame"] += 1
    
    raw = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    frame = raw.copy()
    if strict_privacy:
        frame = cv2.GaussianBlur(frame, (61, 61), 0)
    faces = get_faces(raw)
    known = list(FACE_DATABASE)
    detections = []
    presenter_count = 0
    owner_annotations = []
    for f in faces:
        box = tuple(float(x) for x in f.bbox)
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            obscure(frame, box)
            continue
        best_name = None
        score = -1.0
        for entry in known:
            value = float(np.dot(emb, entry["embedding"]))
            if value > score:
                score = value
                best_name = entry["name"]
        if known and score >= float(threshold):
            if strict_privacy:
                ax, ay, bx, by = clip_box(box, raw.shape, 0.08)
                if bx > ax and by > ay:
                    frame[ay:by, ax:bx] = raw[ay:by, ax:bx]
            presenter_count += 1
            owner_annotations.append((box, best_name, score))
        else:
            detections.append({"box": box, "emb": emb})
    update_tracking(detections, state)
    
    expression_list = []
    if auto_expression and any(d["track"]["mode"] == "emoji" for d in detections) and state["frame"] % 2 == 0:
        expression_list = expression_faces(raw)

    counts = {"mosaic": 0, "emoji": 0, "hide": 0, "show": 0, "replace": 0}
    for det in detections:
        tid = det["id"]
        t = det["track"]
        mode = t["mode"]
        box = det["box"]
        if mode == "emoji":
            expr = "natural"
            if expression_list:
                chosen = max(expression_list, key=lambda it: overlap(it[0], box))
                if overlap(chosen[0], box) > 0.05:
                    expr = chosen[1]
            if expr == t["candidate"]:
                t["stable"] += 1
            else:
                t["candidate"] = expr
                t["stable"] = 1
            if t["stable"] >= 3:
                t["expr"] = expr
            mosaic(frame, box)
            overlay(frame, emoji_image(t["expr"]), box)
        elif mode == "hide":
            obscure(frame, box)
        elif mode == "replace":
            replace_with_registered_face(frame, box, known)
        elif mode == "show":
            if strict_privacy:
                ax, ay, bx, by = clip_box(box, raw.shape, 0.24)
                if bx > ax and by > ay:
                    frame[ay:by, ax:bx] = raw[ay:by, ax:bx]
        else:
            mosaic(frame, box)
        counts[mode if mode in counts else "mosaic"] += 1

    clean_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preview = frame.copy()
    for box, best_name, score in owner_annotations:
        x1, y1, x2, y2 = clip_box(box, preview.shape, 0)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (58, 205, 122), 2)
        label = f"{best_name} {score:.2f}"
        ty = max(23, y1 - 9)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(
            preview,
            (x1, max(0, ty - th - 6)),
            (min(preview.shape[1] - 1, x1 + tw + 8), ty + 4),
            (20, 77, 50),
            -1,
        )
        cv2.putText(
            preview,
            label,
            (x1 + 4, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for det in detections:
        tid = det["id"]
        mode = det["track"]["mode"]
        box = det["box"]
        x1, y1, x2, y2 = clip_box(box, preview.shape, 0)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (75, 155, 255), 2)
        label = f"ID {tid} | {mode}"
        ty = max(23, y1 - 10)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(
            preview,
            (x1, max(0, ty - th - 6)),
            (min(preview.shape[1] - 1, x1 + tw + 8), ty + 4),
            (19, 31, 49),
            -1,
        )
        cv2.putText(
            preview,
            label,
            (x1 + 4, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    state["hitboxes"] = []
    width, height = preview.shape[1], preview.shape[0]
    show_global = state.get("show_global_controls", True)
    toggle_rect = (8, 8, 94, 44)
    cv2.rectangle(preview, toggle_rect[:2], toggle_rect[2:], (38, 52, 80), -1)
    cv2.rectangle(preview, toggle_rect[:2], toggle_rect[2:], (170, 188, 222), 1)
    cv2.putText(
        preview,
        "CLOSE" if show_global else "OPEN",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if show_global:
        cv2.line(preview, (78, 20), (83, 26), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(preview, (83, 26), (88, 20), (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.line(preview, (78, 28), (83, 22), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(preview, (83, 22), (88, 28), (255, 255, 255), 2, cv2.LINE_AA)
    state["hitboxes"].append((toggle_rect, "__ui__", "__toggle_global__"))
    if show_global:
        global_controls = [
            ("M", "mosaic", (32, 64, 198)),
            ("E", "emoji", (177, 95, 30)),
            ("R", "replace", (42, 136, 64)),
            ("C", "show", (95, 95, 95)),
        ]
        bx = toggle_rect[2] + 8
        for label, mode, color in global_controls:
            rect = (bx, 8, bx + 78, 42)
            if rect[2] > width - 8:
                break
            cv2.rectangle(preview, rect[:2], rect[2:], color, -1)
            cv2.rectangle(preview, rect[:2], rect[2:], (245, 245, 245), 1)
            cv2.putText(
                preview,
                "ALL " + label,
                (bx + 8, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            state["hitboxes"].append((rect, None, mode))
            bx += 84
    for det in detections:
        tid = det["id"]
        x1, y1, x2, y2 = clip_box(det["box"], preview.shape, 0)
        btn_h = 28
        btn_w = 36
        gap = 4
        btn_y = y1 - (btn_h + 6) if y1 >= 56 else min(height - (btn_h + 6), y2 + 6)
        total_w = btn_w * 4 + gap * 3
        start_x = max(0, min(x1, width - total_w - 4))
        for j, (label, mode, color) in enumerate(
            [
                ("M", "mosaic", (32, 64, 198)),
                ("E", "emoji", (177, 95, 30)),
                ("R", "replace", (42, 136, 64)),
                ("C", "show", (95, 95, 95)),
            ]
        ):
            ax = start_x + j * (btn_w + gap)
            rect = (ax, btn_y, ax + btn_w, btn_y + btn_h)
            cv2.rectangle(preview, rect[:2], rect[2:], color, -1)
            cv2.rectangle(preview, rect[:2], rect[2:], (245, 245, 245), 1)
            cv2.putText(
                preview,
                label,
                (ax + 10, btn_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            state["hitboxes"].append((rect, tid, mode))
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    stat = (
        (
            "全画面ぼかし保護 ON / "
            if strict_privacy
            else "顔領域のみ加工（検出漏れのリスクあり） / "
        )
        + f"検出 {len(faces)}人  /  登録者 {presenter_count}人  /  非登録 {len(detections)}人  |  "
        + "  ".join(f"{k}:{v}" for k, v in counts.items() if v)
    )
    stat += "\n操作対象ID: " + (
        ", ".join(str(d["id"]) for d in detections) if detections else "なし"
    )
    if include_clean:
        return rgb, clean_rgb, state, stat
    return rgb, state, stat


def parse_command(command):
    s = str(command or "").strip().lower().replace("_", " ")
    aliases = {
        "show": ("c", "x", "show", "clear", "解除", "外す", "通常表示"),
        "replace": ("r", "replace", "置換", "登録顔"),
        "emoji": ("e", "emoji", "絵文字"),
        "mosaic": ("m", "mosaic", "モザイク", "ぼかし"),
        "hide": ("hide", "隠す", "黒塗り", "非表示"),
    }
    words = set(re.findall(r"[a-z]+|[0-9]+|[ぁ-んァ-ヶ一-龠]+", s))
    mode = next(
        (
            mode
            for mode, names in aliases.items()
            if any(name in words or (len(name) > 1 and name in s) for name in names)
        ),
        None,
    )
    found = re.search(r"(?:id|顔|番号)?\s*([0-9]+)", s)
    all_faces = bool(words & {"all", "全員", "すべて"})
    return mode, int(found.group(1)) if found else None, all_faces


def apply_command(command, state, consent):
    if not isinstance(state, dict):
        state = initial_state()
    mode, tid, all_faces = parse_command(command)
    if mode is None:
        return state, "操作を認識できません。例：2 C / ALL C"
    if all_faces:
        state["global_mode"] = mode
        state["commands"].clear()
        for t in state["tracks"].values():
            t["mode"] = mode
        return (
            state,
            "全員の加工を解除しました"
            if mode == "show"
            else f"全員を {mode} にしました",
        )
    if tid is None:
        return state, "対象の顔IDを指定してください。例：2 C"
    if tid not in state["tracks"]:
        return state, f"ID {tid} は現在検出されていません"
    state["tracks"][tid]["mode"] = mode
    state["commands"][tid] = mode
    return (
        state,
        f"ID {tid} の加工を解除しました"
        if mode == "show"
        else f"ID {tid} を {mode} にしました",
    )


def click_preview(state, consent, evt: gr.SelectData):
    if not isinstance(state, dict):
        state = initial_state()
    point = evt.index
    if not isinstance(point, (tuple, list)) or len(point) < 2:
        return state, "映像上の操作ボタンをクリックしてください"
    x, y = map(int, point[:2])
    for (x1, y1, x2, y2), tid, mode in reversed(state.get("hitboxes", [])):
        if x1 <= x <= x2 and y1 <= y <= y2:
            if mode == "__toggle_global__":
                state["show_global_controls"] = not state.get(
                    "show_global_controls", True
                )
                return state, "一括操作を" + (
                    "表示" if state["show_global_controls"] else "収納"
                ) + "しました"
            return apply_command(
                f"all {mode}" if tid is None else f"{tid} {mode}", state, consent
            )
    return state, "映像上の操作ボタンをクリックしてください"


def apply_selected(tid, mode, state, consent):
    try:
        i = int(tid)
    except (TypeError, ValueError):
        return state, "⚠️ 顔IDを入力してください。"
    return apply_command(f"{i} {mode}", state, consent)


def reset_controls(state):
    if not isinstance(state, dict):
        state = initial_state()
    state["global_mode"] = "mosaic"
    state["commands"].clear()
    for t in state["tracks"].values():
        t["mode"] = "mosaic"
    return state, "✅ 全員を初期状態のモザイクに戻しました。"


def transcribe(audio):
    if not audio:
        return "⚠️ マイクで録音してください。", ""
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.AudioFile(str(audio)) as source:
            data = recognizer.record(source)
        text = recognizer.recognize_google(data, language="ja-JP")
        return "認識結果：" + text, text
    except ImportError:
        return "⚠️ SpeechRecognition がありません。pip install SpeechRecognition", ""
    except Exception as e:
        return f"⚠️ 音声認識失敗: {e}", ""


def face_library_data():
    counts = {}
    files = {}
    for person in sorted(KNOWN.iterdir()):
        if person.is_dir():
            image_paths = sorted(
                p
                for p in person.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if image_paths:
                label = f"{person.name} ({len(image_paths)}枚)"
                counts[label] = person.name
                files[label] = "\n".join(f"・{p.name}" for p in image_paths)
        elif person.suffix.lower() in (".jpg", ".jpeg", ".png"):
            label = f"{person.stem} (1枚)"
            counts[label] = person.stem
            files[label] = f"・{person.name}"
    labels = list(counts.keys())
    default = labels[0] if labels else None
    return labels, default, files.get(default, "登録済みファイルはありません。"), files


def all_files_text():
    parts = []
    for person in sorted(KNOWN.iterdir()):
        if person.is_dir():
            photos = sorted(
                p
                for p in person.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
            for photo in photos:
                parts.append(f"{person.name} / {photo.name}")
        elif person.suffix.lower() in (".png", ".jpg", ".jpeg"):
            parts.append(person.name)
    return "\n".join(parts) if parts else "登録済みファイルはありません。"


def select_face_library(choice):
    _, _, _, files = face_library_data()
    if not choice:
        return "登録済みファイルはありません。"
    return files.get(choice, "登録済みファイルはありません。")


CSS = r"""
:root{--bg:#122036;--panel:#203756;--border:#6b8eb8;--text:#ffffff;--muted:#d4e3fc}
body,.gradio-container{background:var(--bg)!important;color:var(--text)!important}
.gradio-container{max-width:1540px!important;margin:auto!important;font-size:16px!important}
.simple-brand{color:#fff;background:#1e3557;border-left:5px solid #85aff8;border-radius:8px;padding:14px 18px;font-size:18px;font-weight:800;margin:10px 0 28px}
#studio-navigation .tab-nav,#studio-navigation [role="tablist"]{display:flex!important;position:relative!important;gap:12px!important;align-items:stretch!important;flex-wrap:nowrap!important;margin:0 0 32px!important;padding:0 0 18px!important;min-height:83px!important;height:auto!important;overflow:visible!important;border:0!important}
#studio-navigation .tab-nav button,#studio-navigation [role="tablist"] [role="tab"]{display:flex!important;align-items:center!important;justify-content:center!important;flex:1 1 0!important;height:78px!important;min-height:78px!important;max-height:78px!important;min-width:0!important;position:relative!important;margin:0!important;padding:10px 14px!important;background:#2b466d!important;border:2px solid #a4c0e3!important;border-radius:14px!important;color:#fff!important;-webkit-text-fill-color:#fff!important;font-size:22px!important;font-weight:800!important;box-sizing:border-box!important;cursor:pointer!important;transform:none!important;overflow:hidden!important}
#studio-navigation .tab-nav button:hover{background:#3b5f93!important}
#studio-navigation .tab-nav button.selected,#studio-navigation .tab-nav button[aria-selected="true"],#studio-navigation [role="tablist"] [role="tab"][aria-selected="true"]{background:#366bce!important;border-color:#ffffff!important;color:#fff!important}
#studio-navigation .tab-nav button:focus-visible{outline:3px solid #f9cf70!important;outline-offset:3px!important}
#studio-navigation .tabitem,#studio-navigation [role="tabpanel"]{position:relative!important;margin-top:0!important;padding-top:14px!important}
.section-tag{color:#b3d1ff;font-size:13px;letter-spacing:.12em;font-weight:750}
.subline{color:#f0f6ff;font-size:16px;line-height:1.8;margin:14px 0}
.feature-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:23px 0}
.feature-tile{border:0!important;border-left:4px solid #6b9bd6!important;border-radius:0!important;background:transparent!important;padding:14px 18px!important;box-shadow:none!important;display:block!important;min-height:0!important}
.feature-icon{width:42px!important;height:42px!important;background:#32537a!important;border-radius:10px;padding:8px;box-sizing:border-box;margin-bottom:12px}
.feature-icon svg{width:100%;height:100%;display:block}
.feature-tile strong{display:block!important;color:#fff!important;font-size:19px!important;margin:0 0 7px!important}
.feature-tile p{color:#eaf2ff!important;font-size:14px!important;line-height:1.7!important;margin:0!important}
.pane{background:#203756!important;border:1px solid #6b8eb8!important;border-radius:17px!important;padding:18px!important;margin-bottom:12px!important}
.pane h3,.gradio-container h3,.gradio-container h2{color:#fff!important}
.gradio-container .gr-button{border-radius:10px!important;min-height:44px!important;font-weight:700!important;font-size:15px!important}
.gradio-container .gr-button-primary{background:#3b71eb!important;color:#fff!important}
.gradio-container .gr-button-secondary{background:#30527e!important;color:#fff!important}
.gradio-container input,.gradio-container textarea{background:#182e4d!important;color:#fff!important;-webkit-text-fill-color:#fff!important;border:1px solid #6b8eb8!important}
.gradio-container input::placeholder,.gradio-container textarea::placeholder{color:#cce1ff!important}
.warning-note{margin-top:20px;background:#4a3235;border:1px solid #d49a9a;color:#fff5f2;border-radius:12px;padding:17px;font-size:15px;line-height:1.8}
#output-control-panel{background:#2b486d;border:1px solid #91b5e2;border-radius:12px;padding:12px;margin-top:14px}
.live-panel-title{color:#fff;font-size:16px;font-weight:800}.live-panel-title span{color:#e6f0ff;font-size:13px;display:block}
.mini,.enroll-hint{color:#edf4ff!important;font-size:14px!important;line-height:1.7!important}
.enroll-center{max-width:790px!important;margin:auto!important}
#enrollment-camera{position:relative;border:1px solid #7eaadb;border-radius:14px;overflow:hidden}
#enrollment-camera::after{content:"";position:absolute;z-index:10;left:50%;top:53%;transform:translate(-50%,-50%);width:min(39%,210px);height:65%;border:3px dashed #86efac;border-radius:50%;pointer-events:none}
#take-photo{min-height:56px!important;font-size:18px!important}
.enroll-thumbs-title{color:#ffffff!important;font-size:17px!important;font-weight:800!important;margin:16px 0 8px!important}
.enroll-thumbs-title span{color:#ffffff!important;font-weight:normal!important;}
footer{border-top:1px solid #6b8eb8;color:#eaf2ff;padding:18px 0;margin-top:25px;font-size:13px}
#live-camera-input .image-container,#live-camera-input .upload-container,#live-camera-input .empty,#live-camera-input .image-frame{background:#e7edf6!important;color:#15263d!important}
#live-camera-input .image-container *,#live-camera-input .upload-container *,#live-camera-input .empty *,#live-camera-input .image-frame *{color:#15263d!important;-webkit-text-fill-color:#15263d!important}
#live-camera-input .image-container button,#live-camera-input .upload-container button{color:#15263d!important}
#live-camera-input .block-label{background:#1e3557!important;color:#fff!important;border-radius:6px;padding:6px!important}
#live-camera-input .block-label *{color:#fff!important}
@media(max-width:650px){#studio-navigation .tab-nav button,#studio-navigation [role="tablist"] [role="tab"]{height:62px!important;min-height:62px!important;max-height:62px!important;font-size:17px!important;padding:6px!important}.feature-grid{grid-template-columns:1fr!important}}

.product-intro{display:flex;flex-direction:column;gap:0;margin:18px 0 26px;padding:16px 4px 22px;border-bottom:1px solid #5a7ca8}
.product-eyebrow{font-size:15px;color:#c2d8ff;font-weight:800;letter-spacing:.13em;margin-bottom:10px}
.product-intro h1{font-size:clamp(40px,4.8vw,66px)!important;line-height:1.2!important;color:#fff!important;font-weight:850!important;margin:0 0 18px!important;letter-spacing:-.025em}
.product-intro p{font-size:clamp(17px,1.5vw,20px)!important;line-height:1.8!important;color:#ffffff!important;max-width:980px;margin:0!important}
#studio-navigation .tab-nav,#studio-navigation [role="tablist"]{position:relative!important;min-height:106px!important;margin-bottom:28px!important;padding:0 0 18px!important;z-index:1!important}
#studio-navigation .tab-nav button,#studio-navigation [role="tablist"] [role="tab"]{height:88px!important;min-height:88px!important;max-height:88px!important;font-size:23px!important;padding:12px 22px!important}
#studio-navigation .tabitem,#studio-navigation [role="tabpanel"]{position:relative!important;z-index:0!important;margin-top:12px!important;padding-top:20px!important}
.home-details-title{padding-top:12px;margin-bottom:14px}
.feature-tile{background:transparent!important;border-left:3px solid #73a3df!important;padding:14px 22px!important}
@media(max-width:650px){#studio-navigation .tab-nav button,#studio-navigation [role="tablist"] [role="tab"]{height:68px!important;min-height:68px!important;max-height:68px!important;font-size:17px!important;padding:8px!important}}

#preview-mic-toolbar {
  display:flex!important;align-items:center!important;justify-content:space-between!important;
  gap:12px!important;min-height:64px!important;padding:10px 16px!important;
  margin:0 0 10px!important;background:#1a2f4d!important;border:1px solid #739ac8!important;
  border-radius:14px!important;flex-wrap:nowrap!important;
}
#preview-mic-toolbar .mic-toolbar-label{font-weight:750!important;font-size:16px!important;color:#ffffff!important;white-space:nowrap!important;display:flex!important;align-items:center!important;}
#preview-mic-toolbar #preview-mic-audio{
  flex:1 1 auto!important;min-width:180px!important;max-width:100%!important;
  min-height:46px!important;max-height:60px!important;
  background:#264166!important;border:1px solid #6b95cc!important;border-radius:10px!important;padding:2px 8px!important;
}
#preview-mic-toolbar #preview-mic-audio *{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;}
#preview-mic-toolbar #preview-mic-audio button{background:#355887!important;color:#ffffff!important;border-radius:8px!important;cursor:pointer!important;}
#preview-mic-toolbar #preview-mic-audio svg{color:#ffffff!important;stroke:#ffffff!important;fill:#ffffff!important;}

#preview-mic-toolbar .mic-toolbar-label::before{
 content:"";display:inline-block;vertical-align:middle;margin-right:8px;width:22px;height:22px;
 background-repeat:no-repeat;background-size:contain;background-position:center;
 background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%23ffffff%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%3E%3Crect%20x%3D%229%22%20y%3D%222%22%20width%3D%226%22%20height%3D%2213%22%20rx%3D%223%22/%3E%3Cpath%20d%3D%22M5%2011a7%207%200%200%200%2014%200M12%2018v4m-4%200h8%22/%3E%3C/svg%3E");
}

#output-control-panel, #output-control-panel > button, #output-control-panel > button *,
#output-control-panel summary, #output-control-panel summary *,
#output-control-panel .label-wrap, #output-control-panel .label-wrap *{
 color:#fff!important;-webkit-text-fill-color:#fff!important;
}
#output-control-panel svg {color:#fff!important;stroke:#fff!important}

/* 秒数カウント（処理時間表示タイマー）を完全に非表示にするスタイル */
#annotated-preview .eta-bar, #clean-live-preview .eta-bar,
#annotated-preview .progress-text, #clean-live-preview .progress-text,
#annotated-preview .counter, #clean-live-preview .counter,
#annotated-preview .toast, #clean-live-preview .toast,
.image-container .eta-bar, div[class*="eta"], div[class*="timer"], div[class*="counter"],
.gr-image span[style*="position: absolute"] {
  display: none !important;
  visibility: hidden !important;
  opacity: 0 !important;
}
/* Gradioの右下に現れる秒数要素（テキストコンポーネント等）を非表示 */
#annotated-preview .wrap > div > div:last-child,
#clean-live-preview .wrap > div > div:last-child {
  display: none !important;
}
"""


with gr.Blocks(
    title="Privacy Camera Studio",
    css=CSS,
    theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
) as demo:
    gr.HTML(
        '<div class="product-intro"><div class="product-eyebrow">PRIVACY CAMERA STUDIO</div><h1>Privacy Camera Studio</h1></div>'
    )
    with gr.Tabs(elem_id="studio-navigation"):
        with gr.Tab("ホーム"):
            gr.HTML("""<div class="section-tag home-details-title">その他の機能</div>
                <div class="feature-grid">
                  <div class="feature-tile"><div class="feature-icon" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#F8FAFC" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21v-2a8 8 0 0 1 16 0v2"/></svg></div><strong>顔登録</strong><p>正面・右向き・左向きの3回撮影で登録します。</p></div>
                  <div class="feature-tile"><div class="feature-icon" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#F8FAFC" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="3"/><circle cx="12" cy="12" r="4"/><path d="M8 5 9 3h6l1 2"/></svg></div><strong>ライブカメラ</strong><p>未加工映像・操作用映像・完成映像を並べて確認できます。</p></div>
                  <div class="feature-tile"><div class="feature-icon" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#F8FAFC" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M4 12h16M4 17h16"/><circle cx="9" cy="7" r="2" fill="#1D3557"/><circle cx="15" cy="12" r="2" fill="#1D3557"/><circle cx="10" cy="17" r="2" fill="#1D3557"/></svg></div><strong>映像上で操作</strong><p>ALL M / ALL E / ALL R / ALL C と、各顔の M / E / R / C を映像上から切り替えます。</p></div>
                </div>""")
            with gr.Row():
                with gr.Column(elem_classes=["pane"]):
                    gr.Markdown("### 登録済みの顔")
                    with gr.Group(elem_classes=["menu-card"]):
                        home_face_files = gr.Textbox(
                            label="登録済みファイル",
                            interactive=False,
                            lines=9,
                            elem_id="home-file-list",
                        )
                        home_refresh = gr.Button("↻ 登録情報を更新")
                with gr.Column(elem_classes=["pane"]):
                    gr.Markdown("### 使い方のポイント")
                    gr.Markdown("""- 顔登録は **登録** タブで行います。
- 実演は **カメラ** タブで行います。
- C は現在の状態を解除して通常表示に戻します。
- 背景ぼかしは必要なときだけオンにしてください。""")
            gr.HTML(
                '<div class="warning-note">※ 公開配信前に必ず加工後映像を確認してください。顔検出と顔認識には誤りがあり、試作版はプライバシーを完全には保証しません。</div>'
            )
        with gr.Tab("登録"):
            gr.HTML(
                '<div class="section-tag">THREE-CLICK FACE ENROLLMENT</div><h2 style="color:#f1f5ff;margin:8px 0">3回の撮影で、顔登録。</h2><p class="subline">名前を入力したらカメラを起動。正面・右向き・左向きの順に、中央の楕円に顔を合わせて撮影してください。</p>'
            )
            registration_state = gr.State(capture_state)
            latest_capture = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1, min_width=40):
                    pass
                with gr.Column(scale=8, min_width=300, elem_classes=["enroll-center"]):
                    name = gr.Textbox(
                        label="登録名（半角英数字）", placeholder="例：streamer"
                    )
                    reg_status = gr.Markdown(
                        "**STEP 1 / 3：正面** 顔を楕円の中に合わせ、カメラを起動して撮影してください。"
                    )
                    reg_camera = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="FACE CAPTURE / 撮影カメラ",
                        height=410,
                        elem_id="enrollment-camera",
                    )
                    capture_btn = gr.Button(
                        "📸 正面を撮影",
                        variant="primary",
                        size="lg",
                        elem_id="take-photo",
                    )
                    gr.HTML(
                        '<div class="enroll-hint">カメラをON → 楕円の中に顔を合わせる → 撮影（正面・右・左の計3回）<br>うまく撮れなかった場合は同じステップのまま再撮影できます。</div>'
                    )
                    gr.HTML(
                        '<div class="enroll-thumbs-title">撮影した写真 <span style="color: #ffffff !important;">01 正 面 / 02 右向き / 03 左向き</span></div>'
                    )
                    gallery = gr.Gallery(
                        label="登録プレビュー（撮影した写真が左から表示されます）",
                        columns=3,
                        height=135,
                        object_fit="contain",
                        elem_id="enrollment-thumbs",
                    )
                    with gr.Row():
                        reset_photos = gr.Button("↻ 最初から撮影", variant="secondary")
                        reload_btn = gr.Button(
                            "↻ 登録者一覧を更新", variant="secondary"
                        )
                    with gr.Accordion("登録済みの顔ファイル", open=False):
                        face_library_select = gr.Dropdown(
                            label="登録者メニュー",
                            choices=[],
                            value=None,
                            interactive=True,
                            elem_id="face-library-select",
                        )
                        registered_list = gr.Textbox(
                            label="ファイル一覧", lines=6, interactive=False
                        )
                with gr.Column(scale=1, min_width=40):
                    pass
        with gr.Tab("カメラ"):
            state = gr.State(initial_state)
            with gr.Row(equal_height=False, elem_id="demo-three-views"):
                with gr.Column(scale=1, min_width=260, elem_classes=["pane"]):
                    gr.Markdown("### ① LIVE INPUT")
                    camera = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        show_label=False,
                        height=290,
                        elem_id="live-camera-input",
                    )
                with gr.Column(
                    scale=1,
                    min_width=300,
                    elem_classes=["pane"],
                    elem_id="privacy-output-column",
                ):
                    gr.Markdown("### ② PRIVACY OUTPUT")
                    output = gr.Image(
                        show_label=False,
                        streaming=False,
                        interactive=False,
                        height=320,
                        elem_id="annotated-preview",
                    )
                    with gr.Row(elem_id="preview-mic-toolbar"):
                        gr.HTML('<span class="mic-toolbar-label">音声コマンド</span>')
                        audio = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            show_label=False,
                            elem_id="preview-mic-audio",
                            min_width=180,
                        )
                    stats = gr.State("")
                    with gr.Accordion(
                        "LIVE CONTROL PANEL", open=False, elem_id="output-control-panel"
                    ):
                        gr.HTML(
                            '<div class="live-panel-title">LIVE CONTROL PANEL</div>'
                        )
                        with gr.Row():
                            face_id = gr.Number(
                                label="顔ID",
                                value=1,
                                precision=0,
                                min_width=90,
                                scale=1,
                            )
                            face_mode = gr.Dropdown(
                                choices=[
                                    ("モザイク", "mosaic"),
                                    ("表情絵文字", "emoji"),
                                    ("登録した顔に置換 [R]", "replace"),
                                    ("モザイク解除 [C]", "show"),
                                    ("完全に隠す", "hide"),
                                ],
                                value="mosaic",
                                label="表示モード",
                                scale=2,
                                min_width=145,
                            )
                        apply_btn = gr.Button(
                            "選択した顔に適用", variant="primary", size="sm"
                        )
                        reset_btn = gr.Button("全員を初期モザイクに戻す", size="sm")
                        command_status = gr.Markdown("")
                        with gr.Accordion("詳細設定", open=False):
                            strict_privacy = gr.Checkbox(
                                value=False, label="背景ぼかし保護を使う"
                            )
                            threshold = gr.Slider(
                                0.3,
                                0.8,
                                value=0.60,
                                step=0.01,
                                label="本人照合のしきい値",
                            )
                            auto_expression = gr.Checkbox(
                                value=True, label="表情に応じて絵文字を自動変更"
                            )
                            consent = gr.Checkbox(
                                value=False,
                                label="撮影対象者の同意を確認（Cは元の顔を表示します）",
                            )
                            command = gr.Textbox(
                                label="コマンド入力",
                                placeholder="例：2 emoji / 3 mosaic / all replace",
                            )
                            cmd_btn = gr.Button("コマンド実行")
                        voice_status = gr.Markdown("")
                with gr.Column(scale=1, min_width=260, elem_classes=["pane"]):
                    gr.Markdown("### ③ CLEAN LIVE (別ウィンドウ出力中)")
                    clean_live = gr.Image(
                        show_label=False,
                        interactive=False,
                        height=290,
                        elem_id="clean-live-preview",
                    )
    gr.HTML(
        "<footer>PRIVACY CAMERA STUDIO · 顔の登録と通常表示は必ず本人の同意を得てください。顔検出漏れや誤認識があり得るため、公開配信では未加工フレームを出さない構成・配信前の検証が別途必要です。起動はローカル（127.0.0.1）限定です。</footer>"
    )

    reg_camera.stream(
        lambda rgb: rgb,
        inputs=reg_camera,
        outputs=latest_capture,
        stream_every=0.033,
        concurrency_limit=1,
    )
    capture_btn.click(
        capture_once,
        [name, latest_capture, registration_state],
        [registration_state, reg_status, gallery, registered_list, capture_btn],
    )
    reset_photos.click(
        reset_capture, outputs=[registration_state, reg_status, gallery, capture_btn]
    )

    def reload_faces_ui():
        refresh_faces()
        labels, default, file_text, _ = face_library_data()
        return (
            gr.update(choices=labels, value=default),
            file_text,
            all_files_text(),
        )

    reload_btn.click(
        reload_faces_ui, outputs=[face_library_select, registered_list, home_face_files]
    )
    demo.load(
        reload_faces_ui, outputs=[face_library_select, registered_list, home_face_files]
    )
    home_refresh.click(
        reload_faces_ui, outputs=[face_library_select, registered_list, home_face_files]
    )
    face_library_select.change(
        select_face_library, inputs=face_library_select, outputs=registered_list
    )
    camera.stream(
        process_demo_frame,
        [camera, state, threshold, auto_expression, consent, strict_privacy],
        [output, clean_live, state, stats],
        stream_every=0.033,
        concurrency_limit=1,
    )
    output.select(
        click_preview, [state, consent], [state, command_status], show_progress="hidden"
    )
    apply_btn.click(
        apply_selected, [face_id, face_mode, state, consent], [state, command_status]
    )
    cmd_btn.click(apply_command, [command, state, consent], [state, command_status])
    command.submit(apply_command, [command, state, consent], [state, command_status])
    reset_btn.click(reset_controls, [state], [state, command_status])

    def apply_voice(audio, state, consent):
        status, text = transcribe(audio)
        if not text:
            return state, status
        state, response = apply_command(text, state, consent)
        return state, status + "\n\n" + response

    audio.change(apply_voice, [audio, state, consent], [state, voice_status])

if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )