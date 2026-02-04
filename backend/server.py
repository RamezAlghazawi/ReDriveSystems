import asyncio
import json
import os
import time
import zipfile
import concurrent.futures
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
from aiohttp import web, WSMsgType

# =========================
# --- Configuration ---
# =========================
HOST = os.getenv("HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("HTTP_PORT", "8080"))
WS_PORT = int(os.getenv("WS_PORT", "8765"))

MODE = os.getenv("MODE", "synthetic")  # 'synthetic' or 'carla'
CARLA_HOST = os.getenv("CARLA_HOST", "127.0.0.1")
CARLA_PORT = int(os.getenv("CARLA_PORT", "2000"))

LOG_DIR = Path(os.getenv("LOG_DIR", "./sessions")).resolve()

# Global tick base (upper bound); actual compute is client-aware
BASE_FPS = float(os.getenv("FPS", "20"))

# JPEG tuning
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "88"))
JPEG_OPTIMIZE = os.getenv("JPEG_OPTIMIZE", "1") == "1"

# Output sizes (these define what goes to the browser)
PANO_W = int(os.getenv("PANO_W", "1600"))
PANO_H = int(os.getenv("PANO_H", "450"))

MIRROR_W = int(os.getenv("MIRROR_W", "480"))
MIRROR_H = int(os.getenv("MIRROR_H", "360"))

REAR_W = int(os.getenv("REAR_W", "480"))
REAR_H = int(os.getenv("REAR_H", "270"))

MAP_W = int(os.getenv("MAP_W", "340"))
MAP_H = int(os.getenv("MAP_H", "260"))

# Per-stream FPS (big performance lever)
PANO_FPS = float(os.getenv("PANO_FPS", "20"))
MIRROR_FPS = float(os.getenv("MIRROR_FPS", "12"))
REAR_FPS = float(os.getenv("REAR_FPS", "12"))
MAP_FPS = float(os.getenv("MAP_FPS", "5"))
TELEMETRY_FPS = float(os.getenv("TELEMETRY_FPS", "15"))

# When no clients are connected, downshift compute
IDLE_FPS = float(os.getenv("IDLE_FPS", "2"))

STREAMS = ["panorama", "rear_mirror", "mirror_left", "mirror_right", "map"]
WEB_DIR = Path(__file__).parent / "web"


# =========================
# LANE OVERLAY (GROUND PLANE)
# =========================
LANE_OVERLAY_ENABLED = os.getenv("LANE_OVERLAY_ENABLED", "1") == "1"
LANE_OVERLAY_ALPHA = float(os.getenv("LANE_OVERLAY_ALPHA", "0.55"))  # overlay blend strength
LANE_OVERLAY_FPS = float(os.getenv("LANE_OVERLAY_FPS", "10"))        # recompute rate
LANE_STEER_EPS = float(os.getenv("LANE_STEER_EPS", "0.02"))          # min change to recompute

LANE_WIDTH_M = float(os.getenv("LANE_WIDTH_M", "3.6"))
LANE_LOOKAHEAD_M = float(os.getenv("LANE_LOOKAHEAD_M", "50.0"))
LANE_SAMPLES = int(os.getenv("LANE_SAMPLES", "40"))

MAX_STEER_DEG = float(os.getenv("LANE_MAX_STEER_DEG", "35.0"))
WHEELBASE_M = float(os.getenv("LANE_WHEELBASE_M", "2.8"))

BEV_W = int(os.getenv("LANE_BEV_W", "900"))
BEV_H = int(os.getenv("LANE_BEV_H", "650"))

BEV_X_MIN = float(os.getenv("LANE_BEV_X_MIN", "-8.0"))
BEV_X_MAX = float(os.getenv("LANE_BEV_X_MAX", "8.0"))
BEV_Y_MIN = float(os.getenv("LANE_BEV_Y_MIN", "0.0"))
BEV_Y_MAX = float(os.getenv("LANE_BEV_Y_MAX", "55.0"))

ROI_HORIZON_Y_FRAC = float(os.getenv("LANE_ROI_HORIZON_Y_FRAC", "0.56"))
ROI_BOTTOM_Y_FRAC = float(os.getenv("LANE_ROI_BOTTOM_Y_FRAC", "0.98"))
ROI_BOTTOM_LEFT_X_FRAC = float(os.getenv("LANE_ROI_BOTTOM_LEFT_X_FRAC", "0.08"))
ROI_BOTTOM_RIGHT_X_FRAC = float(os.getenv("LANE_ROI_BOTTOM_RIGHT_X_FRAC", "0.92"))
ROI_TOP_LEFT_X_FRAC = float(os.getenv("LANE_ROI_TOP_LEFT_X_FRAC", "0.44"))
ROI_TOP_RIGHT_X_FRAC = float(os.getenv("LANE_ROI_TOP_RIGHT_X_FRAC", "0.56"))

LANE_LINE_THICKNESS = int(os.getenv("LANE_LINE_THICKNESS", "4"))
LANE_LINE_SHADOW_THICKNESS = int(os.getenv("LANE_LINE_SHADOW_THICKNESS", "8"))
LANE_DASHED = os.getenv("LANE_DASHED", "0") == "1"


# =========================
# --- Middleware (CORS) ---
# =========================
@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        resp = await handler(request)
    origin = request.headers.get("Origin", "*")
    resp.headers["Access-Control-Allow-Origin"] = origin if origin else "*"
    resp.headers["Vary"] = "Origin"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


# =========================
# Data Models
# =========================
@dataclass
class Control:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    handbrake: bool = False
    reverse: bool = False
    t: float = 0.0


@dataclass
class Telemetry:
    t: float
    speed_kmh: float
    gear: str
    teleop_active: bool
    lane_assist: bool
    aeb: bool
    obstacle_m: float
    lane_offset_m: float
    pos_x: float
    pos_y: float
    yaw: float
    session_id: Optional[str]
    recording: bool
    ack_ts: float  # Last client message timestamp we processed
    server_ts: float # Current server timestamp
    jitter_ms: float # Estimated jitter in control loop


class VehicleAdapter(ABC):
    @abstractmethod
    def apply_control(self, control: Control): pass

    @abstractmethod
    def get_current_control(self) -> Control: pass

    @abstractmethod
    def set_features(self, lane_assist: bool, aeb: bool): pass

    @abstractmethod
    def set_recording_state(self, is_recording: bool, session_id: Optional[str]): pass

    @abstractmethod
    def tick(self, dt: float) -> Telemetry: pass

    @abstractmethod
    def get_camera_frames(self) -> Dict[str, np.ndarray]: pass


# =========================
# SYNTHETIC ADAPTER
# =========================
class SyntheticAdapter(VehicleAdapter):
    def __init__(self):
        self.control = Control()
        self.lane_assist = True
        self.aeb = True
        self.speed_kmh = 0.0
        self.lane_offset_m = 0.0
        self.obstacle_m = 999.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.yaw = 0.0
        self._obstacle_seed = time.time()
        self.session_id_ref = None
        self.is_recording = False
        print("[Adapter] Initialized Synthetic Vehicle")

    def apply_control(self, control: Control):
        self.control = control

    def get_current_control(self) -> Control:
        return self.control

    def set_features(self, lane_assist: bool, aeb: bool):
        self.lane_assist = lane_assist
        self.aeb = aeb

    def set_recording_state(self, is_recording: bool, session_id: Optional[str]):
        self.is_recording = is_recording
        self.session_id_ref = session_id

    def tick(self, dt: float) -> Telemetry:
        t = time.time()
        gear = "R" if self.control.reverse else "D"

        raw_steer = np.clip(self.control.steer, -1.0, 1.0)
        throttle = np.clip(self.control.throttle, 0.0, 1.0)
        brake = np.clip(self.control.brake, 0.0, 1.0)

        self.lane_offset_m += raw_steer * 0.06
        self.lane_offset_m *= 0.99

        steer_applied = raw_steer
        if self.lane_assist:
            correction = -self.lane_offset_m * 0.35
            steer_applied = np.clip(raw_steer + correction, -1.0, 1.0)

        phase = (t - self._obstacle_seed) % 30.0
        self.obstacle_m = max(3.0, (60.0 - phase * 4.0)) if phase < 12.0 else 999.0

        if self.aeb and self.obstacle_m < 10.0 and self.speed_kmh > 8.0:
            brake = max(brake, 0.85)
            throttle = 0.0

        accel = (throttle * 6.0) - (brake * 10.0)
        if self.control.handbrake:
            accel -= 12.0
        self.speed_kmh = np.clip(self.speed_kmh + accel * (dt * 5.0), 0.0, 120.0)

        v = (self.speed_kmh / 3.6)
        self.yaw += steer_applied * 0.02
        self.pos_x += np.cos(self.yaw) * v * dt
        self.pos_y += np.sin(self.yaw) * v * dt

        return Telemetry(
            t=t, speed_kmh=float(self.speed_kmh), gear=gear,
            teleop_active=True, lane_assist=self.lane_assist, aeb=self.aeb,
            obstacle_m=float(self.obstacle_m), lane_offset_m=float(self.lane_offset_m),
            pos_x=float(self.pos_x), pos_y=float(self.pos_y), yaw=float(self.yaw),
            session_id=self.session_id_ref, recording=self.is_recording,
            ack_ts=0.0, server_ts=t, jitter_ms=0.0
        )

    def get_camera_frames(self) -> Dict[str, np.ndarray]:
        frames = {}
        frames["pano_wide"] = self._draw_wide_front(1920, 540)
        frames["mirror_left"] = self._draw_mirror("mirror_left", 480, 360)
        frames["mirror_right"] = self._draw_mirror("mirror_right", 480, 360)
        frames["rear"] = self._draw_rear(480, 270)
        frames["map"] = self._draw_map()
        return frames

    def _draw_wide_front(self, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        horizon = int(h * 0.42)
        img[:horizon, :] = (40, 30, 30)
        img[horizon:, :] = (30, 30, 30)
        center = w // 2
        cv2.line(img, (center - int(w * 0.30), h), (center - int(w * 0.05), horizon), (200, 200, 200), 2)
        cv2.line(img, (center + int(w * 0.30), h), (center + int(w * 0.05), horizon), (200, 200, 200), 2)
        return img

    def _draw_mirror(self, name, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (25, 25, 30)
        cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        return img

    def _draw_rear(self, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (20, 20, 25)
        cv2.putText(img, "rear", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        return img

    def _draw_map(self):
        img = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
        img[:] = (20, 20, 25)
        return img


# =========================
# CARLA ADAPTER
# =========================
class CarlaAdapter(VehicleAdapter):
    def __init__(self, host=CARLA_HOST, port=CARLA_PORT):
        print(f"[CARLA] Connecting to {host}:{port}...")
        try:
            import carla
        except ImportError:
            raise RuntimeError("CARLA Python API not installed. Run: pip install carla==0.9.15")

        self.carla = carla
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / max(10.0, BASE_FPS)
        self.world.apply_settings(settings)

        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in CARLA map.")

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[0])
        if not self.vehicle:
            actors = self.world.get_actors().filter("vehicle.*")
            if actors:
                self.vehicle = actors[0]
            else:
                raise RuntimeError("Failed to spawn or attach to a vehicle.")

        print(f"[CARLA] Vehicle engaged: {self.vehicle.type_id}")

        self.cams = {}
        self.queues = {}
        self._spawn_cameras(bp_lib)

        self.control = carla.VehicleControl()
        self.lane_assist = False
        self.aeb = False
        self.session_id_ref = None
        self.is_recording = False

        self.world.tick()

    def _spawn_cameras(self, bp_lib):
        carla = self.carla

        def make_rgb(w: int, h: int, fov: float):
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(w))
            cam_bp.set_attribute("image_size_y", str(h))
            cam_bp.set_attribute("fov", str(fov))
            return cam_bp

        pano_bp = make_rgb(1920, 540, 110)
        pano_tf = carla.Transform(
            carla.Location(x=1.8, y=0.0, z=1.55),
            carla.Rotation(pitch=-2.0, yaw=0.0, roll=0.0)
        )
        self._attach_cam("pano_wide", pano_bp, pano_tf)

        mirror_pitch = -12.0
        ml_bp = make_rgb(480, 360, 85)
        mr_bp = make_rgb(480, 360, 85)

        ml_tf = carla.Transform(
            carla.Location(x=0.3, y=-1.25, z=1.25),
            carla.Rotation(pitch=mirror_pitch, yaw=-155.0, roll=0.0)
        )
        mr_tf = carla.Transform(
            carla.Location(x=0.3, y=1.25, z=1.25),
            carla.Rotation(pitch=mirror_pitch, yaw=155.0, roll=0.0)
        )
        self._attach_cam("mirror_left", ml_bp, ml_tf)
        self._attach_cam("mirror_right", mr_bp, mr_tf)

        rear_bp = make_rgb(480, 270, 95)
        rear_tf = carla.Transform(
            carla.Location(x=-1.6, y=0.0, z=1.55),
            carla.Rotation(pitch=-6.0, yaw=180.0, roll=0.0)
        )
        self._attach_cam("rear", rear_bp, rear_tf)

    def _attach_cam(self, name, bp, tf):
        cam = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)
        q = queue.Queue(maxsize=2)  # keep queue small to avoid latency buildup
        def _cb(img):
            try:
                if q.full():
                    q.get_nowait()
                q.put_nowait(img)
            except Exception:
                pass
        cam.listen(_cb)
        self.cams[name] = cam
        self.queues[name] = q

    def apply_control(self, ctrl: Control):
        self.control.steer = float(ctrl.steer)
        self.control.throttle = float(ctrl.throttle)
        self.control.brake = float(ctrl.brake)
        self.control.hand_brake = bool(ctrl.handbrake)
        self.control.reverse = bool(ctrl.reverse)
        self.vehicle.apply_control(self.control)

    def get_current_control(self) -> Control:
        c = self.control
        return Control(
            steer=float(c.steer),
            throttle=float(c.throttle),
            brake=float(c.brake),
            handbrake=bool(c.hand_brake),
            reverse=bool(c.reverse),
            t=time.time()
        )

    def set_features(self, lane_assist: bool, aeb: bool):
        self.lane_assist = bool(lane_assist)
        self.aeb = bool(aeb)

    def set_recording_state(self, is_recording: bool, session_id: Optional[str]):
        self.is_recording = bool(is_recording)
        self.session_id_ref = session_id

    def tick(self, dt: float) -> Telemetry:
        self.world.tick()
        t = time.time()
        v = self.vehicle.get_velocity()
        tr = self.vehicle.get_transform()
        speed_kmh = 3.6 * float(np.sqrt(v.x**2 + v.y**2 + v.z**2))

        return Telemetry(
            t=t,
            speed_kmh=speed_kmh,
            gear="R" if self.control.reverse else "D",
            teleop_active=True,
            lane_assist=self.lane_assist,
            aeb=self.aeb,
            obstacle_m=999.0,
            lane_offset_m=0.0,
            pos_x=float(tr.location.x),
            pos_y=float(tr.location.y),
            yaw=float(tr.rotation.yaw),
            session_id=self.session_id_ref,
            recording=self.is_recording,
            ack_ts=0.0,
            server_ts=t,
            jitter_ms=0.0
        )

    def _carla_img_to_bgr(self, img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((img.height, img.width, 4))  # BGRA
        return arr[:, :, :3]

    def _get_latest(self, name: str, timeout_s: float = 0.05) -> np.ndarray:
        q = self.queues.get(name)
        if not q:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            img = q.get(timeout=timeout_s)
            return self._carla_img_to_bgr(img)
        except queue.Empty:
            return np.zeros((100, 100, 3), dtype=np.uint8)

    def _world_to_local(self, vx: float, vy: float, yaw_deg: float, px: float, py: float) -> Tuple[float, float]:
        dx = px - vx
        dy = py - vy
        yaw = np.deg2rad(yaw_deg)
        c = np.cos(-yaw)
        s = np.sin(-yaw)
        fwd = dx * c - dy * s
        right = dx * s + dy * c
        return fwd, right

    def _local_to_img(self, fwd: float, right: float, cx: int, cy: int, scale: float) -> Tuple[int, int]:
        x = int(cx + right * scale)
        y = int(cy - fwd * scale)
        return x, y

    def _render_local_map(self, size_hw=(MAP_H, MAP_W)) -> np.ndarray:
        h, w = size_hw
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (18, 18, 22)

        tr = self.vehicle.get_transform()
        vx, vy = float(tr.location.x), float(tr.location.y)
        yaw = float(tr.rotation.yaw)

        cx, cy = w // 2, h // 2
        scale = 2.0

        ego_color = (0, 255, 255)
        cv2.circle(img, (cx, cy), 4, ego_color, -1)
        hx, hy = self._local_to_img(12.0, 0.0, cx, cy, scale)
        cv2.arrowedLine(img, (cx, cy), (hx, hy), ego_color, 2, tipLength=0.25)

        wp = self.map.get_waypoint(tr.location, project_to_road=True, lane_type=self.carla.LaneType.Driving)
        if wp is None:
            return img

        def collect_wps(start_wp, direction: str, step_m=2.0, count=60) -> List:
            out = []
            cur = start_wp
            for _ in range(count):
                nxts = cur.next(step_m) if direction == "next" else cur.previous(step_m)
                if not nxts:
                    break
                cur = nxts[0]
                out.append(cur)
            return out

        behind = collect_wps(wp, "prev", step_m=2.0, count=20)
        ahead = collect_wps(wp, "next", step_m=2.0, count=70)
        center_wps = list(reversed(behind)) + [wp] + ahead

        pts = []
        for wpi in center_wps:
            loc = wpi.transform.location
            fwd, right = self._world_to_local(vx, vy, yaw, float(loc.x), float(loc.y))
            pts.append(self._local_to_img(fwd, right, cx, cy, scale))

        if len(pts) > 2:
            arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [arr], False, (140, 140, 160), 2, lineType=cv2.LINE_AA)
        return img

    def get_camera_frames(self) -> Dict[str, np.ndarray]:
        frames: Dict[str, np.ndarray] = {}
        frames["pano_wide"] = self._get_latest("pano_wide")
        frames["mirror_left"] = self._get_latest("mirror_left")
        frames["mirror_right"] = self._get_latest("mirror_right")
        frames["rear"] = self._get_latest("rear")
        frames["map"] = self._render_local_map((MAP_H, MAP_W))
        return frames


# =========================
# SERVER CORE
# =========================
class TeleopServer:
    def __init__(self, adapter: VehicleAdapter):
        self.adapter = adapter
        self.recording = False
        self.session_dir: Optional[Path] = None
        self.frames_dir: Dict[str, Optional[Path]] = {}
        self.frame_idx = 0
        self.controls_f = None
        self.telemetry_f = None
        self.io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._last_tick = time.time()

        # latest_streams[name] = {"jpg": bytes, "ts": float}
        self.latest_streams: Dict[str, Dict[str, Any]] = {
            "panorama": {"jpg": b"", "ts": 0.0},
            "rear_mirror": {"jpg": b"", "ts": 0.0},
            "mirror_left": {"jpg": b"", "ts": 0.0},
            "mirror_right": {"jpg": b"", "ts": 0.0},
            "map": {"jpg": b"", "ts": 0.0},
        }
        self.latest_telemetry: Optional[Telemetry] = None

        # per-stream encode scheduling
        self._last_encoded: Dict[str, float] = {k: 0.0 for k in self.latest_streams.keys()}

        # client tracking (stream + websocket)
        self._active_stream_clients: Dict[str, int] = {k: 0 for k in self.latest_streams.keys()}
        self._active_ws_clients: int = 0

        # lane overlay cache
        self._lane_cache = {
            "last_ts": 0.0,
            "last_steer": None,
            "H_bev2img": None,
            "warped": None,
            "mask_f": None,
            "w": None,
            "h": None,
        }

        # Watchdog & Jitter
        self._last_control_ts = time.time()
        self._last_client_ts = 0.0
        self._jitter_accum = 0.0
        self._last_msg_arrival = time.time()

    # --------------------------
    # CLIENT TRACKING
    # --------------------------
    def any_clients(self) -> bool:
        return self._active_ws_clients > 0 or any(v > 0 for v in self._active_stream_clients.values())

    def inc_stream_client(self, stream_name: str):
        if stream_name in self._active_stream_clients:
            self._active_stream_clients[stream_name] += 1

    def dec_stream_client(self, stream_name: str):
        if stream_name in self._active_stream_clients:
            self._active_stream_clients[stream_name] = max(0, self._active_stream_clients[stream_name] - 1)

    def inc_ws_client(self):
        self._active_ws_clients += 1

    def dec_ws_client(self):
        self._active_ws_clients = max(0, self._active_ws_clients - 1)

    # --------------------------
    # BEV LANE OVERLAY
    # --------------------------
    @staticmethod
    def _meter_to_bev(x_m: float, y_m: float) -> Tuple[int, int]:
        x_norm = (x_m - BEV_X_MIN) / (BEV_X_MAX - BEV_X_MIN)
        y_norm = (y_m - BEV_Y_MIN) / (BEV_Y_MAX - BEV_Y_MIN)
        px = int(np.clip(x_norm * (BEV_W - 1), 0, BEV_W - 1))
        py = int(np.clip((1.0 - y_norm) * (BEV_H - 1), 0, BEV_H - 1))
        return px, py

    @staticmethod
    def _get_bev_to_img_homography(w: int, h: int) -> np.ndarray:
        y_top = int(h * ROI_HORIZON_Y_FRAC)
        y_bot = int(h * ROI_BOTTOM_Y_FRAC)

        dst_img = np.float32([
            [w * ROI_BOTTOM_LEFT_X_FRAC,  y_bot],
            [w * ROI_BOTTOM_RIGHT_X_FRAC, y_bot],
            [w * ROI_TOP_RIGHT_X_FRAC,    y_top],
            [w * ROI_TOP_LEFT_X_FRAC,     y_top],
        ])

        src_bev = np.float32([
            [0,        BEV_H - 1],
            [BEV_W-1,  BEV_H - 1],
            [BEV_W-1,  0],
            [0,        0],
        ])

        return cv2.getPerspectiveTransform(src_bev, dst_img)

    @staticmethod
    def _predict_center_path(steer: float) -> List[Tuple[float, float, float]]:
        steer = float(np.clip(steer, -1.0, 1.0))
        delta = np.deg2rad(MAX_STEER_DEG) * steer
        delta = float(np.clip(delta, -np.deg2rad(60.0), np.deg2rad(60.0)))

        look = float(LANE_LOOKAHEAD_M)
        n = int(max(12, LANE_SAMPLES))
        s_vals = np.linspace(0.0, look, n)

        if abs(delta) < 1e-5:
            return [(0.0, float(s), 0.0) for s in s_vals]

        kappa = np.tan(delta) / float(WHEELBASE_M)
        R = 1.0 / kappa

        out = []
        for s in s_vals:
            theta = s / R
            y = R * np.sin(theta)
            x = R * (1.0 - np.cos(theta))
            out.append((float(x), float(y), float(theta)))
        return out

    @staticmethod
    def _draw_dashed_poly(img: np.ndarray, pts: List[Tuple[int, int]], color, thickness, dash_px=28, gap_px=22):
        if len(pts) < 2:
            return
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            dist = float(np.hypot(dx, dy))
            if dist < 1.0:
                continue
            ux = dx / dist
            uy = dy / dist
            t = 0.0
            while t < dist:
                t2 = min(dist, t + dash_px)
                p1 = (int(x1 + ux * t), int(y1 + uy * t))
                p2 = (int(x1 + ux * t2), int(y1 + uy * t2))
                cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)
                t += dash_px + gap_px

    def _compute_lane_overlay_cached(self, w: int, h: int, steer: float):
        """
        Computes and caches (warped overlay + mask) at LANE_OVERLAY_FPS and/or steer change.
        """
        now = time.time()
        last_ts = self._lane_cache["last_ts"]
        last_steer = self._lane_cache["last_steer"]

        need_recalc = False
        if self._lane_cache["w"] != w or self._lane_cache["h"] != h:
            need_recalc = True
        elif last_steer is None:
            need_recalc = True
        else:
            if abs(float(steer) - float(last_steer)) >= LANE_STEER_EPS:
                need_recalc = True
            if (now - last_ts) >= (1.0 / max(1.0, LANE_OVERLAY_FPS)):
                need_recalc = True

        if not need_recalc:
            return

        bev = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
        path = self._predict_center_path(steer)
        half = float(LANE_WIDTH_M) * 0.5

        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []

        for (x, y, heading) in path:
            cs = float(np.cos(heading))
            sn = float(np.sin(heading))
            nx = cs
            ny = -sn

            lx = x - nx * half
            ly = y - ny * half
            rx = x + nx * half
            ry = y + ny * half

            left_pts.append(self._meter_to_bev(lx, ly))
            right_pts.append(self._meter_to_bev(rx, ry))

        shadow = (0, 0, 0)
        main = (245, 245, 245)

        if LANE_DASHED:
            self._draw_dashed_poly(bev, left_pts, shadow, LANE_LINE_SHADOW_THICKNESS)
            self._draw_dashed_poly(bev, right_pts, shadow, LANE_LINE_SHADOW_THICKNESS)
            self._draw_dashed_poly(bev, left_pts, main, LANE_LINE_THICKNESS)
            self._draw_dashed_poly(bev, right_pts, main, LANE_LINE_THICKNESS)
        else:
            if len(left_pts) > 2:
                arrL = np.array(left_pts, dtype=np.int32).reshape((-1, 1, 2))
                arrR = np.array(right_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(bev, [arrL], False, shadow, LANE_LINE_SHADOW_THICKNESS, lineType=cv2.LINE_AA)
                cv2.polylines(bev, [arrR], False, shadow, LANE_LINE_SHADOW_THICKNESS, lineType=cv2.LINE_AA)
                cv2.polylines(bev, [arrL], False, main, LANE_LINE_THICKNESS, lineType=cv2.LINE_AA)
                cv2.polylines(bev, [arrR], False, main, LANE_LINE_THICKNESS, lineType=cv2.LINE_AA)

        H_bev2img = self._get_bev_to_img_homography(w, h)
        warped = cv2.warpPerspective(bev, H_bev2img, (w, h), flags=cv2.INTER_LINEAR)

        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]

        self._lane_cache.update({
            "last_ts": now,
            "last_steer": float(steer),
            "H_bev2img": H_bev2img,
            "warped": warped,
            "mask_f": mask_f,
            "w": w,
            "h": h,
        })

    def _overlay_lane_guides_ground_cached(self, img_bgr: np.ndarray, steer: float) -> np.ndarray:
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr

        h, w = img_bgr.shape[:2]
        self._compute_lane_overlay_cached(w, h, steer)

        warped = self._lane_cache.get("warped", None)
        mask_f = self._lane_cache.get("mask_f", None)
        if warped is None or mask_f is None:
            return img_bgr

        alpha = float(LANE_OVERLAY_ALPHA)
        out = img_bgr.astype(np.float32)
        out = out * (1.0 - mask_f * alpha) + warped.astype(np.float32) * (mask_f * alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    # --------------------------
    # SESSION / RECORDING
    # --------------------------
    def start_session(self) -> Dict[str, Any]:
        if self.recording:
            return {"ok": False}

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        sid = datetime.utcnow().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = LOG_DIR / sid
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.frames_dir = {}
        for name in STREAMS:
            p = self.session_dir / "frames" / name
            p.mkdir(parents=True, exist_ok=True)
            self.frames_dir[name] = p

        self.controls_f = open(self.session_dir / "controls.csv", "w", encoding="utf-8", buffering=1)
        self.controls_f.write("t,steer,throttle,brake,handbrake,reverse\n")

        self.telemetry_f = open(self.session_dir / "telemetry.csv", "w", encoding="utf-8", buffering=1)
        self.telemetry_f.write("t,speed_kmh,gear,teleop_active,lane_assist,aeb,obstacle_m,lane_offset_m,pos_x,pos_y,yaw\n")

        meta = {"created_utc": datetime.utcnow().isoformat() + "Z", "fps": BASE_FPS, "platform": MODE}
        with open(self.session_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.recording = True
        self.frame_idx = 0
        self.adapter.set_recording_state(True, sid)
        print(f"[Server] Session Started: {sid}")
        return {"ok": True, "session_id": sid}

    def stop_session(self) -> Dict[str, Any]:
        if not self.recording:
            return {"ok": True, "session_id": None}

        self.recording = False
        sid = self.session_dir.name
        print(f"[Server] Session Stopping: {sid}")

        if self.controls_f:
            self.controls_f.close()
        if self.telemetry_f:
            self.telemetry_f.close()
        self.controls_f = None
        self.telemetry_f = None
        self.adapter.set_recording_state(False, None)

        try:
            zip_path = self.session_dir.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(self.session_dir):
                    for file in files:
                        p = Path(root) / file
                        zf.write(p, p.relative_to(self.session_dir))
            return {"ok": True, "session_id": sid, "zip_path": str(zip_path)}
        except Exception as e:
            return {"ok": True, "session_id": sid, "error": str(e)}

    # --------------------------
    # STREAMING / ENCODING
    # --------------------------
    @staticmethod
    def _resize_if_needed(img: np.ndarray, w: int, h: int) -> np.ndarray:
        if img is None or img.size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
        ih, iw = img.shape[:2]
        if iw == w and ih == h:
            return img
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if (w < iw or h < ih) else cv2.INTER_LINEAR)

    @staticmethod
    def _jpeg_params():
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(JPEG_QUALITY, 30, 100))]
        if JPEG_OPTIMIZE:
            params += [int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
        return params

    def _enc_jpg(self, img: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", img, self._jpeg_params())
        return buf.tobytes() if ok else b""

    def _due(self, stream_name: str, fps: float) -> bool:
        now = time.time()
        dt = now - self._last_encoded.get(stream_name, 0.0)
        return dt >= (1.0 / max(1.0, fps))

    def tick(self):
        now = time.time()
        dt = max(0.001, min(0.05, now - self._last_tick))
        self._last_tick = now

        tel = self.adapter.tick(dt)
        
        # --- WATCHDOG CHECK ---
        # If no control message for > 100ms, trigger SAFE STOP
        now = time.time()
        time_since_last_ctrl = now - self._last_control_ts
        if time_since_last_ctrl > 0.1: # 100ms strict timeout
             # Force Safe Stop
             stop_ctrl = Control(steer=0.0, throttle=0.0, brake=1.0, handbrake=True, reverse=False, t=now)
             self.adapter.apply_control(stop_ctrl)
             # Update telemetry to reflect 'teleop_active' might be technically true but we are intervening
             # (In this simple MVP we just override the control application)
        
        # Inject Metris into Telemetry
        tel.ack_ts = self._last_client_ts
        tel.server_ts = now
        tel.jitter_ms = self._jitter_accum
        
        self.latest_telemetry = tel

        raw_frames = self.adapter.get_camera_frames()
        ctrl_now = self.adapter.get_current_control()

        # Panorama
        if self._active_stream_clients["panorama"] > 0 and self._due("panorama", PANO_FPS):
            pano_src = raw_frames.get("pano_wide")
            pano = self._resize_if_needed(pano_src, PANO_W, PANO_H)

            if LANE_OVERLAY_ENABLED:
                pano = self._overlay_lane_guides_ground_cached(pano, ctrl_now.steer)

            jpg = self._enc_jpg(pano)
            self.latest_streams["panorama"] = {"jpg": jpg, "ts": time.time()}
            self._last_encoded["panorama"] = time.time()

        # Rear
        if self._active_stream_clients["rear_mirror"] > 0 and self._due("rear_mirror", REAR_FPS):
            rear = self._resize_if_needed(raw_frames.get("rear"), REAR_W, REAR_H)
            jpg = self._enc_jpg(rear)
            self.latest_streams["rear_mirror"] = {"jpg": jpg, "ts": time.time()}
            self._last_encoded["rear_mirror"] = time.time()

        # Mirrors
        if self._active_stream_clients["mirror_left"] > 0 and self._due("mirror_left", MIRROR_FPS):
            ml = self._resize_if_needed(raw_frames.get("mirror_left"), MIRROR_W, MIRROR_H)
            jpg = self._enc_jpg(ml)
            self.latest_streams["mirror_left"] = {"jpg": jpg, "ts": time.time()}
            self._last_encoded["mirror_left"] = time.time()

        if self._active_stream_clients["mirror_right"] > 0 and self._due("mirror_right", MIRROR_FPS):
            mr = self._resize_if_needed(raw_frames.get("mirror_right"), MIRROR_W, MIRROR_H)
            jpg = self._enc_jpg(mr)
            self.latest_streams["mirror_right"] = {"jpg": jpg, "ts": time.time()}
            self._last_encoded["mirror_right"] = time.time()

        # Map
        if self._active_stream_clients["map"] > 0 and self._due("map", MAP_FPS):
            mp = self._resize_if_needed(raw_frames.get("map"), MAP_W, MAP_H)
            jpg = self._enc_jpg(mp)
            self.latest_streams["map"] = {"jpg": jpg, "ts": time.time()}
            self._last_encoded["map"] = time.time()

        # Recording (save what was encoded)
        if self.recording:
            t = time.time()
            c = ctrl_now
            if self.controls_f:
                self.controls_f.write(
                    f"{t:.6f},{c.steer:.4f},{c.throttle:.4f},{c.brake:.4f},{int(c.handbrake)},{int(c.reverse)}\n"
                )
            if self.telemetry_f and tel:
                self.telemetry_f.write(
                    f"{t:.6f},{tel.speed_kmh:.3f},{tel.gear},{int(tel.teleop_active)},{int(tel.lane_assist)},{int(tel.aeb)},"
                    f"{tel.obstacle_m:.3f},{tel.lane_offset_m:.4f},{tel.pos_x:.3f},{tel.pos_y:.3f},{tel.yaw:.4f}\n"
                )

            self.frame_idx += 1
            to_save = {k: self.latest_streams.get(k, {}).get("jpg", b"") for k in self.frames_dir.keys()}
            self.io_pool.submit(self._save_worker, self.frame_idx, to_save, self.frames_dir)

    @staticmethod
    def _save_worker(idx, frames, dirs):
        for name, data in frames.items():
            if name in dirs and data:
                with open(dirs[name] / f"frame_{idx:06d}.jpg", "wb") as f:
                    f.write(data)


# =========================
# --- SELECT ADAPTER ---
# =========================
if MODE == "carla":
    try:
        adapter = CarlaAdapter()
    except Exception as e:
        print(f"[ERROR] Failed to load CARLA: {e}")
        print("[WARN] Falling back to Synthetic")
        adapter = SyntheticAdapter()
else:
    adapter = SyntheticAdapter()

server = TeleopServer(adapter)


# =========================
# --- WEB HANDLERS ---
# =========================
async def index(_):
    return web.FileResponse(WEB_DIR / "index.html")


async def health(_):
    return web.json_response({"ok": True})


def _stream_fps(name: str) -> float:
    if name == "panorama":
        return PANO_FPS
    if name in ("mirror_left", "mirror_right"):
        return MIRROR_FPS
    if name == "rear_mirror":
        return REAR_FPS
    if name == "map":
        return MAP_FPS
    return BASE_FPS


async def stream_handler(req):
    name = req.match_info["name"]
    if name not in server.latest_streams:
        return web.Response(status=404, text="Unknown stream")

    server.inc_stream_client(name)

    boundary = "frame"
    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": f"multipart/x-mixed-replace; boundary={boundary}"}
    )
    await resp.prepare(req)

    last_sent_ts = 0.0
    target_fps = _stream_fps(name)
    sleep_dt = 1.0 / max(1.0, target_fps)

    try:
        while True:
            item = server.latest_streams.get(name, {})
            frame = item.get("jpg", b"")
            ts = float(item.get("ts", 0.0))

            if frame and ts > last_sent_ts:
                last_sent_ts = ts
                await resp.write(
                    b"--" + boundary.encode() +
                    b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() +
                    b"\r\n\r\n" + frame + b"\r\n"
                )

            await asyncio.sleep(sleep_dt)
    except asyncio.CancelledError:
        raise
    except Exception:
        pass
    finally:
        server.dec_stream_client(name)

    return resp


async def ws_handler(req):
    ws = web.WebSocketResponse()
    await ws.prepare(req)
    server.inc_ws_client()

    async def send_loop():
        dt = 1.0 / max(1.0, TELEMETRY_FPS)
        while not ws.closed:
            if server.latest_telemetry:
                await ws.send_str(json.dumps({"type": "telemetry", "data": asdict(server.latest_telemetry)}))
            await asyncio.sleep(dt)

    task = asyncio.create_task(send_loop())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                d = json.loads(msg.data)
                if d.get("type") == "control":
                    c_data = d.get("data", {})
                    c = Control(
                        t=time.time(),
                        steer=float(c_data.get("steer", 0)),
                        throttle=float(c_data.get("throttle", 0)),
                        brake=float(c_data.get("brake", 0)),
                        handbrake=bool(c_data.get("handbrake", False)),
                        reverse=bool(c_data.get("reverse", False)),
                    )
                    
                    # Watchdog Update
                    now = time.time()
                    client_ts = float(c_data.get("t", 0.0)) # Client sent timestamp (seconds)
                    
                    # Jitter Calculation (Exponential Moving Average of inter-arrival diff)
                    # Expected interval is approx 20ms (50Hz)
                    dt_arrival = now - server._last_msg_arrival
                    server._last_msg_arrival = now
                    
                    # deviations from expected ~20ms
                    diff = abs(dt_arrival - 0.020)
                    server._jitter_accum = (server._jitter_accum * 0.9) + (diff * 1000.0 * 0.1)
                    
                    server._last_control_ts = now
                    server._last_client_ts = client_ts * 1000.0 # Convert back to ms for echo if client sent sec
                    if client_ts > 10000000000: # if client sent ms
                         server._last_client_ts = client_ts
                    else:
                         server._last_client_ts = client_ts * 1000.0

                    server.adapter.apply_control(c)
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        task.cancel()
        server.dec_ws_client()

    return ws


async def session_start(_):
    return web.json_response(server.start_session())


async def session_stop(_):
    return web.json_response(server.stop_session())


async def set_features(req):
    d = await req.json()
    server.adapter.set_features(d.get("lane_assist", True), d.get("aeb", True))
    return web.json_response({"ok": True})


# =========================
# --- MAIN LOOP ---
# =========================
async def run_loop():
    while True:
        # Client-aware compute throttling
        target = BASE_FPS if server.any_clients() else IDLE_FPS
        server.tick()
        await asyncio.sleep(1.0 / max(1.0, target))


async def main():
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/stream/{name}", stream_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/api/session/start", session_start)
    app.router.add_post("/api/session/stop", session_stop)
    app.router.add_post("/api/features", set_features)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, HOST, HTTP_PORT).start()

    # Keep WS_PORT server for backwards compatibility (optional),
    # but we already serve /ws on HTTP_PORT too.
    ws_app = web.Application()
    ws_app.router.add_get("/ws", ws_handler)
    ws_runner = web.AppRunner(ws_app)
    await ws_runner.setup()
    await web.TCPSite(ws_runner, HOST, WS_PORT).start()

    print(f"UI:  http://localhost:{HTTP_PORT}")
    print(f"WS:  ws://localhost:{WS_PORT}/ws  (also available on http://localhost:{HTTP_PORT}/ws)")
    print(f"MODE: {MODE}")
    print(f"JPEG_QUALITY: {JPEG_QUALITY}  JPEG_OPTIMIZE: {JPEG_OPTIMIZE}")
    print(f"PANO: {PANO_W}x{PANO_H}@{PANO_FPS}  MIRROR: {MIRROR_W}x{MIRROR_H}@{MIRROR_FPS}  REAR: {REAR_W}x{REAR_H}@{REAR_FPS}  MAP: {MAP_W}x{MAP_H}@{MAP_FPS}")
    print(f"LANE_OVERLAY_ENABLED: {LANE_OVERLAY_ENABLED}  LANE_OVERLAY_FPS: {LANE_OVERLAY_FPS}  LANE_STEER_EPS: {LANE_STEER_EPS}")

    await run_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        server.stop_session()
