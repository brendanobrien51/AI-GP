"""
AirSim Drone Racing Lab - Autonomous Gate Racer v15
====================================================
COMPLETE TARGETING REBUILD.

The fundamental problem with v8-v14 was that the racing line lookahead
at high speed pointed PAST the gate toward the next segment.  No amount
of cushion/lateral force can overcome a primary velocity vector aimed at
the wrong place.

v15 ARCHITECTURE:
  APPROACH: target blends from racing-line to gate center as drone
            approaches.  By 6m out the target IS the gate center.
            The drone is guaranteed to aim through the opening.
  CENTER:   target = gate_center + forward*2m.  No racing line.
            Just punch straight through.
  EXIT:     racing-line lookahead toward next gate.  Full speed,
            smooth arcs between gates.

Speed: turn-angle adaptive (fast on straights, slow on sharp turns)
       + proximity taper near gates.

No cushion needed because the target itself is at gate center.
Backup/retry on collision.  Smooth guideline projection.
"""
import math
import re
import time
import airsimdroneracinglab as airsim
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------
TAKEOFF_HEIGHT        = -1.5

# Speed
MAX_APPROACH_SPEED    = 14.0
MAX_CENTER_SPEED      = 10.0
MAX_EXIT_SPEED        = 16.0
MIN_SPEED_FRACTION    = 0.28
MIN_CENTER_SPEED      = 5.5
YAW_SCALE_EXPONENT    = 1.1
BRAKE_DIST            = 2.0
BRAKE_MIN_FRACTION    = 0.40
VEL_FF_ALPHA          = 0.04

# Gate geometry
APPROACH_TOL          = 5.0
EXIT_TOL              = 12.0
FIRST_APPROACH_DIST   = 8.0
APPROACH_DIST         = 3.5
EXIT_DIST             = 3.5
CENTER_THROUGH_DIST   = 2.5     # target past gate center in CENTER
GATE_CROSS_EPS        = 0.15
DYN_APPROACH_FRAC     = 0.40
AUTO_SKIP_PROGRESS    = 3.5

# Commit zone
COMMIT_DIST           = 2.0
COMMIT_LAT_DAMP       = 0.15
COMMIT_MIN_FWD_SPEED  = 6.0
COMMIT_ALT_TOL        = 1.5

# Collision recovery
COLLISION_SPEED_THRESH = 0.8
COLLISION_DIST_THRESH  = 4.0
COLLISION_TIME_THRESH  = 1.2
BACKUP_DIST           = 6.0
BACKUP_SPEED          = 3.5
MAX_RETRIES           = 2
RETRY_SPEED_MULT      = 0.50

# Turn-angle adaptive speed
TURN_SPEED_STRAIGHT   = 1.0
TURN_SPEED_90DEG      = 0.48
TURN_SPEED_180DEG     = 0.28
TURN_TAPER_DIST       = 14.0    # start slowing this far before gate

# Lookahead (EXIT phase only at full length)
LOOKAHEAD_MIN         = 6.0
LOOKAHEAD_MAX         = 20.0
LOOKAHEAD_SPEED_SCALE = 1.4

# Gate-centric targeting (APPROACH phase)
GATE_BLEND_FAR        = 16.0    # at this dist, 100% racing line
GATE_BLEND_NEAR       = 5.0     # at this dist, 100% gate center
# Between FAR and NEAR the target smoothly transitions.

# Lateral centering (small residual correction in APPROACH)
LATERAL_GAIN_APPROACH = 2.0
LATERAL_GAIN_CENTER   = 3.0
LATERAL_MAX           = 5.0

# CV-guided correction
CV_STEER_GAIN_APPROACH = 1.5
CV_STEER_GAIN_CENTER   = 3.0
CV_STEER_GAIN_EXIT     = 0.5
CV_STEER_MAX           = 5.0
CV_STEER_BLEND         = 0.5

# Yaw alignment
YAW_ALIGN_START_DIST   = 14.0   # start aligning early
YAW_ALIGN_FULL_DIST    = 3.5

# Altitude
ALT_KP                = 6.0
ALT_KP_CENTER         = 7.0
ALT_MAX_Z_VEL         = 10.0

# Stuck detection
STUCK_WINDOW          = 3.0
STUCK_MIN_PROGRESS    = 0.25
TRACKER_STALE_S       = 0.8
MAX_GATE_TIME         = 25.0
CONTROL_DT            = 0.05

# CV
CONTOUR_MIN_AREA      = 250
CV_SCORE_THRESHOLD    = 0.30
CANNY_LOW             = 50
CANNY_HIGH            = 130
HSV_RANGES = [
    (np.array([  5,  80,  80], dtype=np.uint8), np.array([ 25, 255, 255], dtype=np.uint8)),
    (np.array([ 18, 100, 100], dtype=np.uint8), np.array([ 38, 255, 255], dtype=np.uint8)),
    (np.array([  0,  80,  80], dtype=np.uint8), np.array([  8, 255, 255], dtype=np.uint8)),
    (np.array([172,  80,  80], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
]

# Visualisation
CAMERA_FOV_H_DEG = 90.0
PATH_VIS_DIST    = 45.0
PATH_VIS_SAMPLES = 60

# Dashboard
DASH_W, DASH_H = 1280, 720
CAM_W   = 854
RIGHT_W = DASH_W - CAM_W
MAP_H   = 380
BRAIN_H = DASH_H - MAP_H
C_WHITE=(255,255,255); C_BLACK=(0,0,0); C_GREEN=(0,220,80); C_YELLOW=(0,220,220)
C_ORANGE=(0,165,255); C_RED=(50,50,240); C_CYAN=(230,220,0); C_GRAY=(120,120,120)
C_DGRAY=(40,40,40); C_BLUE=(230,100,0); C_LIME=(50,255,100); C_MAGENTA=(200,50,200)
PHASE_COLORS = {"APPROACH": C_YELLOW, "CENTER": C_GREEN, "EXIT": C_ORANGE}

# ---------------------------------------------------------------------------
# MATH
# ---------------------------------------------------------------------------
def clamp(v, lo, hi): return max(lo, min(hi, v))
def vec3(x, y, z): return np.array([float(x), float(y), float(z)], dtype=np.float64)
def norm(v): return float(np.linalg.norm(v))
def unit(v):
    n = norm(v)
    return v / n if n > 1e-7 else np.zeros(3, dtype=np.float64)
def lerp(a, b, t):
    t = clamp(t, 0.0, 1.0)
    return a * (1.0 - t) + b * t
def angle_lerp(a, b, t):
    t = clamp(t, 0.0, 1.0)
    return a + math.atan2(math.sin(b - a), math.cos(b - a)) * t
def quat_to_yaw(q):
    return math.atan2(2.0*(q.w_val*q.z_val + q.x_val*q.y_val),
                      1.0 - 2.0*(q.y_val**2 + q.z_val**2))
def quat_to_rot_matrix(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]], dtype=np.float64)
def pose_to_np(p): return vec3(p.position.x_val, p.position.y_val, p.position.z_val)
def state_pos(s):
    p = s.kinematics_estimated.position; return vec3(p.x_val, p.y_val, p.z_val)
def state_vel(s):
    v = s.kinematics_estimated.linear_velocity; return vec3(v.x_val, v.y_val, v.z_val)

class SmoothQuat:
    """Low-pass filter on quaternion for stable guideline projection."""
    def __init__(self, alpha=0.12):
        self._q = [1.0, 0.0, 0.0, 0.0]; self._alpha = alpha; self._init = False
    def update(self, q):
        nq = [q.w_val, q.x_val, q.y_val, q.z_val]
        if not self._init:
            self._q = list(nq); self._init = True; return
        if sum(a*b for a, b in zip(self._q, nq)) < 0:
            nq = [-v for v in nq]
        a = self._alpha
        self._q = [o*(1-a) + n*a for o, n in zip(self._q, nq)]
        mag = math.sqrt(sum(v**2 for v in self._q))
        if mag > 1e-8:
            self._q = [v/mag for v in self._q]
    @property
    def w_val(self): return self._q[0]
    @property
    def x_val(self): return self._q[1]
    @property
    def y_val(self): return self._q[2]
    @property
    def z_val(self): return self._q[3]

# ---------------------------------------------------------------------------
# GATE DISCOVERY & PATH
# ---------------------------------------------------------------------------
def get_gate_names(client):
    names = client.simListSceneObjects(".*[Gg]ate.*")
    cleaned = [n for n in names if isinstance(n, str) and n.strip()]
    def sk(name):
        nums = re.findall(r"\d+", name)
        return [int(n) for n in nums] if nums else [10**9]
    cleaned.sort(key=sk); return cleaned

def get_object_pose_safe(client, name):
    if not hasattr(client, "race_tier"): client.race_tier = None
    if not hasattr(client, "level_name"): client.level_name = ""
    try: return client.simGetObjectPose(name)
    except Exception:
        internal = getattr(client, "_VehicleClient__internalGetObjectPose", None)
        if internal: return internal(name)
        raise

def build_path(client, gate_names):
    gates = []
    for name in gate_names:
        pose = get_object_pose_safe(client, name)
        pos = pose_to_np(pose)
        if np.isfinite(pos).all():
            gates.append((name, pos.copy()))
    path = []; total = len(gates)
    for i, (name, center) in enumerate(gates):
        if i == 0:
            fwd = unit(gates[1][1] - center) if total > 1 else vec3(1,0,0)
        elif i == total - 1:
            fwd = unit(center - gates[i-1][1])
        else:
            fwd = unit(gates[i+1][1] - gates[i-1][1])
        if norm(fwd) < 1e-6: fwd = vec3(1,0,0)
        if i == 0: ba = FIRST_APPROACH_DIST
        else:
            gap = norm(center - gates[i-1][1])
            ba = max(2.0, min(APPROACH_DIST, gap * DYN_APPROACH_FRAC))
        path.append({"index": i, "name": name, "center": center.copy(),
                      "approach": center - fwd * ba,
                      "exit": center + fwd * EXIT_DIST, "forward": fwd})
    # Turn angles
    for i, g in enumerate(path):
        if i == 0 or i == len(path) - 1:
            g["turn_angle"] = 0.0; g["turn_speed_mult"] = TURN_SPEED_STRAIGHT
        else:
            v_in = unit(g["center"] - path[i-1]["center"])
            v_out = unit(path[i+1]["center"] - g["center"])
            dot = clamp(float(np.dot(v_in, v_out)), -1.0, 1.0)
            angle = math.degrees(math.acos(dot))
            g["turn_angle"] = angle
            if angle <= 0.1:
                g["turn_speed_mult"] = TURN_SPEED_STRAIGHT
            elif angle <= 90:
                t = angle / 90.0
                g["turn_speed_mult"] = TURN_SPEED_STRAIGHT + t * (TURN_SPEED_90DEG - TURN_SPEED_STRAIGHT)
            else:
                t = (angle - 90) / 90.0
                g["turn_speed_mult"] = TURN_SPEED_90DEG + t * (TURN_SPEED_180DEG - TURN_SPEED_90DEG)
    for g in path:
        print(f"  Gate {g['index']+1:>2}: {g['name']:<14} "
              f"turn={g['turn_angle']:5.1f}d  spd_mult={g['turn_speed_mult']:.2f}")
    return path

# ---------------------------------------------------------------------------
# RACING LINE (used for EXIT phase and visualisation)
# ---------------------------------------------------------------------------
def build_racing_line(path):
    line = []
    for g in path:
        line.append(g["approach"].copy())
        line.append(g["center"].copy())
        line.append(g["exit"].copy())
    return line

def find_closest_segment(line, pos, min_seg=0):
    bd = float("inf"); bi = min_seg; bt = 0.0
    for i in range(min_seg, len(line)-1):
        a = line[i]; ab = line[i+1] - a; absq = float(np.dot(ab, ab))
        t = 0.0 if absq < 1e-8 else clamp(float(np.dot(pos-a, ab))/absq, 0, 1)
        d = norm(pos - (a + ab*t))
        if d < bd: bd = d; bi = i; bt = t
    return bi, bt, bd

def get_lookahead_point(line, si, st, ld):
    if si >= len(line)-1: return line[-1].copy()
    a, b = line[si], line[si+1]; sl = norm(b-a); rm = sl*(1.0-st)
    if ld <= rm and sl > 1e-7: return lerp(a, b, st + ld/sl)
    dl = ld - rm
    for i in range(si+1, len(line)-1):
        s = norm(line[i+1] - line[i])
        if dl <= s and s > 1e-7: return lerp(line[i], line[i+1], dl/s)
        dl -= s
    return line[-1].copy()

def sample_path_ahead(line, si, st, td, np_):
    pts = []; step = td / max(np_, 1)
    for i in range(np_+1): pts.append(get_lookahead_point(line, si, st, step*i))
    return pts

# ---------------------------------------------------------------------------
# CAMERA PROJECTION
# ---------------------------------------------------------------------------
def project_points(world_pts, drone_pos, quat, dw, dh, fov=CAMERA_FOV_H_DEG):
    R = quat_to_rot_matrix(quat); Rt = R.T
    fx = dw / (2.0 * math.tan(math.radians(fov/2.0))); fy = fx
    cx = dw/2.0; cy = dh/2.0; out = []
    for pt in world_pts:
        lc = Rt @ (pt - drone_pos)
        if lc[0] < 0.3: out.append(None); continue
        out.append((int(cx + lc[1]/lc[0]*fx), int(cy + lc[2]/lc[0]*fy)))
    return out

# ---------------------------------------------------------------------------
# COMPUTER VISION
# ---------------------------------------------------------------------------
def multi_hsv_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    return combined

def edge_mask(frame):
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0)
    return cv2.dilate(cv2.Canny(g, CANNY_LOW, CANNY_HIGH), None, iterations=1)

def score_contour(c, fs):
    area = cv2.contourArea(c)
    if area < CONTOUR_MIN_AREA: return None
    fh, fw = fs[:2]; frac = area/(fh*fw)
    if frac < 0.001 or frac > 0.65: return None
    rect = cv2.minAreaRect(c); rw, rh = rect[1]
    if min(rw, rh) < 5: return None
    ra = rw*rh; rectangularity = area/ra if ra > 1 else 0
    aspect = max(rw, rh)/(min(rw, rh)+1e-6)
    if aspect > 5: return None
    ascore = 1.0 if aspect < 1.8 else max(0.25, 1.0-(aspect-1.8)*0.18)
    hull = cv2.convexHull(c); ha = cv2.contourArea(hull)
    solidity = area/ha if ha > 1 else 0
    score = rectangularity * ascore * (0.5 + 0.5*solidity)
    if score < CV_SCORE_THRESHOLD: return None
    cx, cy = int(rect[0][0]), int(rect[0][1])
    return {"score": score, "rect_score": rectangularity, "aspect_score": ascore,
            "solidity": solidity, "area_frac": frac, "aspect": aspect,
            "centroid_px": (cx, cy), "contour": c, "rect": rect}

def find_best_gate_contour(hm, em, fs):
    merged = cv2.bitwise_or(hm, em)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        r = score_contour(c, fs)
        if r and (best is None or r["score"] > best["score"]): best = r
    return best

class GateTracker:
    def __init__(self): self._best = None; self._t = 0; self.frames_lost = 0
    def update(self, det):
        if det: self._best = det; self._t = time.time(); self.frames_lost = 0
        else: self.frames_lost += 1
    @property
    def current(self):
        if self._best is None or time.time()-self._t > TRACKER_STALE_S: return None
        return self._best
    def reset(self): self._best = None; self._t = 0; self.frames_lost = 0

# ---------------------------------------------------------------------------
# CV STEERING
# ---------------------------------------------------------------------------
def cv_steer_correction(tracker, fs, gfwd, phase, commit):
    det = tracker.current
    if det is None: return np.zeros(3, dtype=np.float64), 0.0, 0.0
    fh, fw = fs[:2]; cx, cy = det["centroid_px"]
    nx = (cx - fw/2.0)/(fw/2.0); ny = (cy - fh/2.0)/(fh/2.0)
    gain = {"CENTER": CV_STEER_GAIN_CENTER, "APPROACH": CV_STEER_GAIN_APPROACH
            }.get(phase, CV_STEER_GAIN_EXIT)
    if commit: gain *= COMMIT_LAT_DAMP
    down = vec3(0,0,1); right = np.cross(gfwd, down)
    rn = norm(right)
    right = right/rn if rn > 1e-6 else vec3(0,1,0)
    corr = right*nx*gain + vec3(0, 0, ny*gain*0.5)
    m = norm(corr)
    if m > CV_STEER_MAX: corr = corr/m*CV_STEER_MAX
    return corr, nx, ny

# ---------------------------------------------------------------------------
# STUCK DETECTOR
# ---------------------------------------------------------------------------
class StuckDetector:
    def __init__(self): self._h = []
    def update(self, p):
        now = time.time(); self._h.append((now, p))
        self._h = [(t,v) for t,v in self._h if t >= now-STUCK_WINDOW]
    def is_stuck(self):
        if len(self._h) < 10: return False
        if self._h[-1][0]-self._h[0][0] < STUCK_WINDOW*0.75: return False
        return (self._h[-1][1]-self._h[0][1]) < STUCK_MIN_PROGRESS
    def reset(self): self._h = []

# ---------------------------------------------------------------------------
# FLIGHT HELPERS
# ---------------------------------------------------------------------------
def lateral_correction_vec(dp, gc, gf, gain, mx):
    tg = gc-dp; fp = float(np.dot(tg, gf)); le = tg - gf*fp
    c = le*gain; m = norm(c)
    if m > mx: c = c/m*mx
    return c

def compute_speed_scale(ye, dt, phase):
    ys = max(MIN_SPEED_FRACTION, math.cos(clamp(abs(ye), 0, math.pi/2))**YAW_SCALE_EXPONENT)
    if phase == "CENTER": return ys
    bs = max(BRAKE_MIN_FRACTION, dt/BRAKE_DIST) if dt < BRAKE_DIST else 1.0
    return min(ys, bs)

def signed_progress(dp, gc, gf): return float(np.dot(dp-gc, gf))

def altitude_cmd(dz, tz, phase):
    ze = tz-dz; kp = ALT_KP_CENTER if phase == "CENTER" else ALT_KP
    return clamp(ze*kp, -ALT_MAX_Z_VEL, ALT_MAX_Z_VEL)

# ---------------------------------------------------------------------------
# DASHBOARD (compact)
# ---------------------------------------------------------------------------
class TopDownMap:
    def __init__(self, gcs, pw, ph, m=28):
        xs=[g[0] for g in gcs]; ys=[g[1] for g in gcs]
        sx=max(max(xs)-min(xs),5); sy=max(max(ys)-min(ys),5)
        uw=pw-2*m; uh=ph-2*m; self.scale=min(uw/sx,uh/sy)
        self.ox=m+(uw-sx*self.scale)/2-min(xs)*self.scale
        self.oy=m+(uh-sy*self.scale)/2-min(ys)*self.scale
        self.pw,self.ph=pw,ph
    def to_px(self,wx,wy):
        return(clamp(int(wx*self.scale+self.ox),0,self.pw-1),
               clamp(int(wy*self.scale+self.oy),0,self.ph-1))

def draw_top_down(path, gi, dp, dy, tgt, rl, rsi):
    p = np.full((MAP_H,RIGHT_W,3),(18,18,28),dtype=np.uint8)
    if not path: return p
    tdm = TopDownMap([g["center"] for g in path], RIGHT_W, MAP_H)
    for i in range(max(0,rsi), len(rl)-1):
        cv2.line(p,tdm.to_px(rl[i][0],rl[i][1]),tdm.to_px(rl[i+1][0],rl[i+1][1]),(40,80,40),1,cv2.LINE_AA)
    for i,g in enumerate(path):
        px,py = tdm.to_px(g["center"][0],g["center"][1]); ic=(i==gi)
        col = C_CYAN if ic else ((60,160,60) if i<gi else C_GRAY)
        fwd=g["forward"]; perp=np.array([-fwd[1],fwd[0]])
        hl=max(4,int((10 if ic else 6)/tdm.scale))
        a=tdm.to_px(g["center"][0]+perp[0]*hl/tdm.scale,g["center"][1]+perp[1]*hl/tdm.scale)
        b=tdm.to_px(g["center"][0]-perp[0]*hl/tdm.scale,g["center"][1]-perp[1]*hl/tdm.scale)
        cv2.line(p,a,b,col,3 if ic else 2,cv2.LINE_AA)
        cv2.putText(p,str(i+1),(px+5,py-4),cv2.FONT_HERSHEY_SIMPLEX,0.36,col,1)
    tx,ty=tdm.to_px(tgt[0],tgt[1]); cv2.drawMarker(p,(tx,ty),C_ORANGE,cv2.MARKER_DIAMOND,10,2)
    dx,dy2=tdm.to_px(dp[0],dp[1]); tl=11
    tip=(int(dx+math.cos(dy)*tl),int(dy2+math.sin(dy)*tl))
    lft=(int(dx+math.cos(dy+2.4)*6),int(dy2+math.sin(dy+2.4)*6))
    rgt=(int(dx+math.cos(dy-2.4)*6),int(dy2+math.sin(dy-2.4)*6))
    cv2.fillPoly(p,[np.array([tip,lft,rgt])],C_RED)
    cv2.polylines(p,[np.array([tip,lft,rgt])],True,C_WHITE,1)
    cv2.putText(p,"TOP-DOWN MAP",(6,14),cv2.FONT_HERSHEY_SIMPLEX,0.42,C_GRAY,1)
    return p

def _bar(p,x,y,w,h,f,fg,lb="",vl=""):
    f=clamp(f,0,1); cv2.rectangle(p,(x,y),(x+w,y+h),C_DGRAY,-1)
    cv2.rectangle(p,(x,y),(x+max(2,int(w*f)),y+h),fg,-1)
    cv2.rectangle(p,(x,y),(x+w,y+h),C_GRAY,1)
    if lb: cv2.putText(p,lb,(x-2,y+h-2),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_GRAY,1)
    if vl: cv2.putText(p,vl,(x+w+4,y+h-2),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_WHITE,1)
def _cbar(p,x,y,w,h,f,fg,lb="",vl=""):
    f=clamp(f,-1,1); mid=x+w//2; cv2.rectangle(p,(x,y),(x+w,y+h),C_DGRAY,-1)
    fw=int(w/2*abs(f))
    if f>=0: cv2.rectangle(p,(mid,y),(mid+fw,y+h),fg,-1)
    else: cv2.rectangle(p,(mid-fw,y),(mid,y+h),fg,-1)
    cv2.line(p,(mid,y),(mid,y+h),C_GRAY,1); cv2.rectangle(p,(x,y),(x+w,y+h),C_GRAY,1)
    if lb: cv2.putText(p,lb,(x-2,y+h-2),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_GRAY,1)
    if vl: cv2.putText(p,vl,(x+w+4,y+h-2),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_WHITE,1)

def draw_brain(phase,gi,total,gn,spd,mx,yed,lem,gp,dg,det,vel,el,sd,ze,cnx,cny,ic,ld,retries,ta):
    p=np.full((BRAIN_H,RIGHT_W,3),(14,14,22),dtype=np.uint8)
    px=10; bx=92; bw=RIGHT_W-bx-54; row=24; y=24
    cv2.putText(p,"BRAIN STATE",(px,14),cv2.FONT_HERSHEY_SIMPLEX,0.42,C_GRAY,1)
    pcol=PHASE_COLORS.get(phase,C_WHITE); badge=f"  {phase}  "
    if ic: badge=" COMMIT "; pcol=C_MAGENTA
    (tw,th),_=cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,0.54,2)
    rx=RIGHT_W-tw-px-4
    cv2.rectangle(p,(rx-4,2),(rx+tw+2,2+th+6),pcol,-1)
    cv2.putText(p,badge,(rx,2+th+2),cv2.FONT_HERSHEY_SIMPLEX,0.54,C_BLACK,2)
    cv2.putText(p,f"Gate {gi+1}/{total} {gn} r={retries} t={ta:.0f}d",(px,y),cv2.FONT_HERSHEY_SIMPLEX,0.40,C_WHITE,1)
    y+=12; cv2.line(p,(px,y),(RIGHT_W-px,y),C_DGRAY,1); y+=7
    _bar(p,bx,y,bw,12,spd/max(mx,0.1),C_GREEN,"SPEED",f"{spd:.1f}/{mx:.0f}"); y+=row
    _cbar(p,bx,y,bw,12,yed/90,C_ORANGE,"YAW",f"{yed:+.1f}d"); y+=row
    _bar(p,bx,y,bw,12,min(lem/2.5,1),C_RED if lem>1.5 else C_YELLOW,"LAT",f"{lem:.2f}m"); y+=row
    _cbar(p,bx,y,bw,12,clamp(gp/4,-1,1),C_GREEN if gp>0 else C_YELLOW,"PRG",f"{gp:+.2f}m"); y+=row
    _cbar(p,bx,y,bw,12,clamp(ze/3,-1,1),C_BLUE,"ALT",f"{ze:+.2f}m"); y+=row
    _bar(p,bx,y,bw,12,clamp(1-dg/25,0,1),C_BLUE,"DIST",f"{dg:.1f}m"); y+=row
    _bar(p,bx,y,bw,12,clamp(ld/LOOKAHEAD_MAX,0,1),C_LIME,"LOOK",f"{ld:.1f}m"); y+=row
    _cbar(p,bx,y,bw,12,clamp(cnx,-1,1),C_CYAN,"CV-X",f"{cnx:+.2f}"); y+=row
    if det:
        cc=C_GREEN if det["score"]>0.65 else(C_YELLOW if det["score"]>0.45 else C_RED)
        _bar(p,bx,y,bw,12,det["score"],cc,"CV",f"{det['score']:.2f}"); y+=18
    else: cv2.putText(p,"CV: --",(bx,y+10),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_RED,1); y+=18
    cv2.line(p,(px,y),(RIGHT_W-px,y),C_DGRAY,1); y+=6
    sc=C_RED if sd.is_stuck() else C_DGRAY
    cv2.rectangle(p,(px,y),(px+80,y+13),sc,-1)
    cv2.putText(p,"STUCK" if sd.is_stuck() else "FLOW",(px+4,y+10),cv2.FONT_HERSHEY_SIMPLEX,0.36,C_WHITE,1)
    s3=norm(vel)
    cv2.putText(p,f"|V|={s3:.1f} t={el:.1f}s",(px,y+28),cv2.FONT_HERSHEY_SIMPLEX,0.34,C_CYAN,1)
    return p

def draw_cam(frame, tracker, phase, gn, dv, yed, lc, ic, ppx, gcpx, glab, tpx):
    cam = cv2.resize(frame, (CAM_W, DASH_H), interpolation=cv2.INTER_LINEAR)
    h, w = cam.shape[:2]; cw, ch = w//2, h//2; mg = 80; np_ = len(ppx)
    for i in range(np_-1):
        p1, p2 = ppx[i], ppx[i+1]
        if p1 is None or p2 is None: continue
        if not(-mg<p1[0]<w+mg and -mg<p1[1]<h+mg): continue
        if not(-mg<p2[0]<w+mg and -mg<p2[1]<h+mg): continue
        t = i/max(np_-1,1); col=(0,220,int(80+140*t)); th=max(1,3-int(t*2))
        cv2.line(cam,p1,p2,col,th,cv2.LINE_AA)
    for i,pt in enumerate(ppx):
        if pt and 0<=pt[0]<w and 0<=pt[1]<h and i%5==0:
            t=i/max(np_-1,1); cv2.circle(cam,pt,2,(0,220,int(80+140*t)),-1)
    if tpx and 0<=tpx[0]<w and 0<=tpx[1]<h:
        cv2.drawMarker(cam,tpx,C_CYAN,cv2.MARKER_DIAMOND,16,2)
    for i,gpt in enumerate(gcpx):
        if gpt and -20<gpt[0]<w+20 and -20<gpt[1]<h+20:
            cv2.circle(cam,gpt,10,C_CYAN,2,cv2.LINE_AA); cv2.circle(cam,gpt,3,C_CYAN,-1)
            if i<len(glab): cv2.putText(cam,glab[i],(gpt[0]+14,gpt[1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.48,C_CYAN,1)
    det = tracker.current
    if det:
        sx=CAM_W/frame.shape[1]; sy=DASH_H/frame.shape[0]
        box=cv2.boxPoints(det["rect"]); box[:,0]*=sx; box[:,1]*=sy
        t=clamp((det["score"]-CV_SCORE_THRESHOLD)/(1-CV_SCORE_THRESHOLD),0,1)
        bc=(int(50*t),int(200*t),int(50+200*(1-t)))
        cv2.drawContours(cam,[np.int32(box)],0,bc,2)
        dpx=int(det["centroid_px"][0]*sx); dpy=int(det["centroid_px"][1]*sy)
        cv2.circle(cam,(dpx,dpy),7,bc,-1); cv2.line(cam,(cw,ch),(dpx,dpy),(0,200,120),1,cv2.LINE_AA)
        lbl=f"CV {det['score']:.2f}"+(" [C]" if tracker.frames_lost>0 else "")
        cv2.putText(cam,lbl,(dpx+10,dpy-8),cv2.FONT_HERSHEY_SIMPLEX,0.46,bc,1)
    else: cv2.putText(cam,"NO GATE",(cw-70,ch-50),cv2.FONT_HERSHEY_SIMPLEX,0.7,C_RED,2)
    cv2.circle(cam,(cw,ch),32,C_WHITE,1,cv2.LINE_AA)
    cv2.line(cam,(cw-24,ch),(cw+24,ch),C_WHITE,1); cv2.line(cam,(cw,ch-24),(cw,ch+24),C_WHITE,1)
    spd=norm(dv)
    if spd>0.3:
        sc=5.5; ax=int(cw+dv[1]*sc); ay=int(ch-dv[2]*sc)
        cv2.arrowedLine(cam,(cw,ch),(ax,ay),C_CYAN,2,tipLength=0.28,line_type=cv2.LINE_AA)
    yr=math.radians(yed); yt=(int(cw+math.sin(yr)*55),int(ch-math.cos(yr)*55))
    yc=C_GREEN if abs(yed)<10 else(C_YELLOW if abs(yed)<30 else C_RED)
    cv2.line(cam,(cw,ch),yt,yc,2,cv2.LINE_AA)
    pcol=PHASE_COLORS.get(phase,C_WHITE); badge=f" {phase} "
    if ic: badge=" COMMIT "; pcol=C_MAGENTA
    (bw2,bh2),_=cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,0.70,2)
    cv2.rectangle(cam,(8,6),(8+bw2+4,6+bh2+8),pcol,-1)
    cv2.putText(cam,badge,(10,6+bh2+2),cv2.FONT_HERSHEY_SIMPLEX,0.70,C_BLACK,2)
    cv2.putText(cam,f"-> {gn}",(w-210,30),cv2.FONT_HERSHEY_SIMPLEX,0.50,C_WHITE,1)
    if tracker.frames_lost>0: cv2.putText(cam,f"LOST {tracker.frames_lost}f",(w-150,60),cv2.FONT_HERSHEY_SIMPLEX,0.44,C_RED,1)
    return cam

def shutdown(client):
    for fn in [lambda:client.hoverAsync().join(), lambda:client.landAsync().join(),
               lambda:client.disarm(), lambda:client.disableApiControl(), cv2.destroyAllWindows]:
        try: fn()
        except: pass

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Connecting..."); client = airsim.MultirotorClient()
    client.race_tier=None; client.level_name=""
    client.confirmConnection(); client.enableApiControl(); client.arm()
    print("Taking off..."); client.takeoffAsync().join()
    s0 = state_pos(client.getMultirotorState())
    client.moveToPositionAsync(float(s0[0]),float(s0[1]),float(TAKEOFF_HEIGHT),2.0).join()
    gate_names = get_gate_names(client)
    if not gate_names: shutdown(client); raise RuntimeError("No gates found.")
    print(f"Found {len(gate_names)} gates.")
    for i,n in enumerate(gate_names): print(f"  {i+1:>2}. {n}")
    path = build_path(client, gate_names)
    rl = build_racing_line(path)
    if path:
        g1a=path[0]["approach"]; g1z=path[0]["center"][2]
        cur=state_pos(client.getMultirotorState())
        print(f"Pre-race: climbing to z={g1z:.1f}")
        client.moveToPositionAsync(float(cur[0]),float(cur[1]),float(g1z),3.0).join()
        print("Pre-race: approach pt")
        client.moveToPositionAsync(float(g1a[0]),float(g1a[1]),float(g1z),3.0).join()
        client.hoverAsync().join(); time.sleep(0.5); print("GO!")
    cv2.namedWindow("Drone Racing Dashboard",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Racing Dashboard",DASH_W,DASH_H)

    gi=0; phase="APPROACH"; gstart=time.time(); tracker=GateTracker()
    sd=StuckDetector(); cvnx=0.0; cvny=0.0; rsi=0
    retries=0; col_timer=0.0; smooth_q=SmoothQuat(alpha=0.12)

    try:
        while True:
            tl = time.time()
            state = client.getMultirotorState()
            dp = state_pos(state); dyaw = quat_to_yaw(state.kinematics_estimated.orientation)
            dquat = state.kinematics_estimated.orientation
            dvel = state_vel(state); gel = time.time()-gstart; cspd = norm(dvel)

            responses = client.simGetImages([airsim.ImageRequest("0",airsim.ImageType.Scene,False,False)])
            if responses and responses[0].height>0:
                frame=np.frombuffer(responses[0].image_data_uint8,dtype=np.uint8).reshape(
                    responses[0].height,responses[0].width,3).copy()
            else: frame=np.zeros((480,640,3),dtype=np.uint8)
            hsm=multi_hsv_mask(frame); em=edge_mask(frame)
            tracker.update(find_best_gate_contour(hsm,em,frame.shape))

            if gi>=len(path):
                done=np.zeros((DASH_H,DASH_W,3),dtype=np.uint8)
                cv2.putText(done,"COURSE COMPLETE",(DASH_W//2-230,DASH_H//2),cv2.FONT_HERSHEY_SIMPLEX,2.2,C_GREEN,4)
                cv2.imshow("Drone Racing Dashboard",done); cv2.waitKey(2500); break

            gate=path[gi]
            if gel>MAX_GATE_TIME:
                print(f"  [TIMEOUT] gate {gi+1}"); gi+=1; phase="APPROACH"; gstart=time.time()
                tracker.reset(); sd.reset(); retries=0; col_timer=0; continue

            gp = signed_progress(dp, gate["center"], gate["forward"])
            dg = norm(gate["center"]-dp); ze = gate["center"][2]-dp[2]

            # Auto-skip
            if phase in("APPROACH","CENTER") and gp>AUTO_SKIP_PROGRESS:
                print(f"  [AUTO-SKIP] gate {gi+1}"); gi+=1; phase="APPROACH"; gstart=time.time()
                tracker.reset(); sd.reset(); retries=0; col_timer=0; continue

            # Collision backup
            if(cspd<COLLISION_SPEED_THRESH and dg<COLLISION_DIST_THRESH and
               phase in("APPROACH","CENTER") and gel>2.0):
                col_timer += CONTROL_DT
            else: col_timer = max(0, col_timer-CONTROL_DT*0.5)
            if col_timer > COLLISION_TIME_THRESH:
                if retries<MAX_RETRIES:
                    retries+=1; col_timer=0; sd.reset()
                    print(f"  [BACKUP] gate {gi+1}, retry {retries}/{MAX_RETRIES}")
                    client.hoverAsync().join(); time.sleep(0.2)
                    bk=gate["center"]-gate["forward"]*BACKUP_DIST; bk[2]=gate["center"][2]
                    client.moveToPositionAsync(float(bk[0]),float(bk[1]),float(bk[2]),BACKUP_SPEED).join()
                    client.hoverAsync().join(); time.sleep(0.3)
                    phase="APPROACH"; gstart=time.time(); tracker.reset(); continue
                else:
                    print(f"  [SKIP] gate {gi+1}"); gi+=1; phase="APPROACH"; gstart=time.time()
                    tracker.reset(); sd.reset(); retries=0; col_timer=0; continue

            in_commit=(phase=="CENTER" and abs(gp)<COMMIT_DIST and abs(ze)<COMMIT_ALT_TOL)

            # ===========================================================
            # TARGETING - the core of v15
            # ===========================================================
            # Racing line position (for EXIT and blending)
            ms = gi*3
            rsi, rst, _ = find_closest_segment(rl, dp, min_seg=ms)
            rl_lookahead = clamp(cspd*LOOKAHEAD_SPEED_SCALE, LOOKAHEAD_MIN, LOOKAHEAD_MAX)
            rl_target = get_lookahead_point(rl, rsi, rst, rl_lookahead)

            # Gate center target (slightly past center for through-flight)
            gate_target = gate["center"] + gate["forward"] * CENTER_THROUGH_DIST

            if phase == "APPROACH":
                max_speed = MAX_APPROACH_SPEED
                # KEY: blend from racing line to gate center as we approach
                # Far away: follow racing line for smooth path
                # Close: aim directly at gate center
                blend = clamp((GATE_BLEND_FAR - dg) /
                              (GATE_BLEND_FAR - GATE_BLEND_NEAR), 0.0, 1.0)
                target = lerp(rl_target, gate_target, blend)
                lookahead_d = rl_lookahead * (1.0 - blend * 0.5)

            elif phase == "CENTER":
                max_speed = MAX_CENTER_SPEED
                # Aim straight through gate center. No racing line.
                target = gate_target
                lookahead_d = CENTER_THROUGH_DIST

            else:  # EXIT
                max_speed = MAX_EXIT_SPEED
                target = rl_target
                lookahead_d = rl_lookahead

            # Turn-angle adaptive speed
            turn_mult = gate["turn_speed_mult"]
            if phase in ("APPROACH", "CENTER"):
                taper = clamp(1.0 - dg/TURN_TAPER_DIST, 0.0, 1.0)
                max_speed *= 1.0 + taper*(turn_mult-1.0)
            elif phase == "EXIT" and gi+1 < len(path):
                nm = path[gi+1]["turn_speed_mult"]
                nd = norm(path[gi+1]["center"]-dp)
                taper = clamp(1.0 - nd/TURN_TAPER_DIST, 0.0, 1.0)
                max_speed *= 1.0 + taper*(nm-1.0)

            if retries>0 and phase in("APPROACH","CENTER"):
                max_speed *= RETRY_SPEED_MULT

            tt = target-dp; dt = norm(tt)

            # Stuck detection
            if phase=="CENTER":
                sd.update(gp)
                if sd.is_stuck():
                    print(f"  [STUCK] gate {gi+1}"); phase="EXIT"; sd.reset(); continue
            else: sd.reset()

            # Phase transitions
            if phase=="APPROACH" and dg<APPROACH_TOL:
                phase="CENTER"; sd.reset(); continue
            if phase=="CENTER" and gp>GATE_CROSS_EPS:
                phase="EXIT"; sd.reset(); continue
            if phase=="EXIT":
                pe = gp>EXIT_DIST*0.6
                cn = (gi+1<len(path) and norm(path[gi+1]["approach"]-dp)<APPROACH_TOL*1.5)
                if dt<EXIT_TOL or pe or cn:
                    gi+=1; phase="APPROACH"; gstart=time.time()
                    tracker.reset(); sd.reset(); retries=0; col_timer=0; continue

            # Direction & yaw
            direction = unit(tt)
            target_yaw = math.atan2(direction[1], direction[0])
            gate_fwd_yaw = math.atan2(gate["forward"][1], gate["forward"][0])
            if dg < YAW_ALIGN_START_DIST:
                yb = clamp(1.0-(dg-YAW_ALIGN_FULL_DIST)/(YAW_ALIGN_START_DIST-YAW_ALIGN_FULL_DIST),0,0.7)
                desired_yaw = angle_lerp(target_yaw, gate_fwd_yaw, yb)
            else: desired_yaw = target_yaw
            yerr = math.atan2(math.sin(desired_yaw-dyaw), math.cos(desired_yaw-dyaw))
            yed = math.degrees(yerr)

            # Speed
            ss = compute_speed_scale(yerr, dt, phase)
            spd = max_speed*ss
            if phase=="CENTER": spd = max(spd, MIN_CENTER_SPEED)
            vd = direction*spd

            # Lateral corrections (smaller now - targeting does the heavy lifting)
            lg = {"CENTER":LATERAL_GAIN_CENTER,"APPROACH":LATERAL_GAIN_APPROACH}.get(phase,0.0)
            lc = lateral_correction_vec(dp,gate["center"],gate["forward"],lg,LATERAL_MAX)
            lem = norm(lc)/max(lg,0.01)
            cvc,cvnx,cvny = cv_steer_correction(tracker,frame.shape,gate["forward"],phase,in_commit)
            cl = lerp(lc,cvc,CV_STEER_BLEND) if tracker.current else lc

            if in_commit:
                cl = cl*COMMIT_LAT_DAMP
                fc = float(np.dot(vd,gate["forward"]))
                if fc<COMMIT_MIN_FWD_SPEED:
                    vd = vd+gate["forward"]*(COMMIT_MIN_FWD_SPEED-fc)

            vc = vd+cl; vc = lerp(vc,dvel,VEL_FF_ALPHA)

            # Altitude
            atz = gate["center"][2] if phase=="CENTER" else target[2]
            vc[2] = altitude_cmd(dp[2], atz, phase)

            client.moveByVelocityAsync(float(vc[0]),float(vc[1]),float(vc[2]),
                CONTROL_DT*1.5, yaw_mode=airsim.YawMode(is_rate=False,
                yaw_or_rate=float(math.degrees(desired_yaw))))

            # Visualisation
            smooth_q.update(dquat)
            vpts = sample_path_ahead(rl, rsi, rst, PATH_VIS_DIST, PATH_VIS_SAMPLES)
            ppx = project_points(vpts, dp, smooth_q, CAM_W, DASH_H)
            tpx = (project_points([target],dp,smooth_q,CAM_W,DASH_H) or [None])[0]
            gc3=[path[j]["center"] for j in range(gi,min(gi+4,len(path)))]
            glab=[f"G{j+1}" for j in range(gi,min(gi+4,len(path)))]
            gcpx=project_points(gc3,dp,smooth_q,CAM_W,DASH_H)

            dash=np.zeros((DASH_H,DASH_W,3),dtype=np.uint8)
            dash[:,:CAM_W]=draw_cam(frame,tracker,phase,gate["name"],dvel,yed,cl,in_commit,ppx,gcpx,glab,tpx)
            cv2.line(dash,(CAM_W,0),(CAM_W,DASH_H),C_DGRAY,1)
            dash[:MAP_H,CAM_W:]=draw_top_down(path,gi,dp,dyaw,target,rl,rsi)
            cv2.line(dash,(CAM_W,MAP_H),(DASH_W,MAP_H),C_DGRAY,1)
            dash[MAP_H:,CAM_W:]=draw_brain(phase,gi,len(path),gate["name"],spd,max_speed,
                yed,lem,gp,dg,tracker.current,dvel,gel,sd,ze,cvnx,cvny,in_commit,
                lookahead_d,retries,gate["turn_angle"])
            cv2.imshow("Drone Racing Dashboard",dash)

            cvl="%.2f"%tracker.current["score"] if tracker.current else "--"
            ct=" CMT" if in_commit else ""
            bl_pct = int(clamp((GATE_BLEND_FAR-dg)/(GATE_BLEND_FAR-GATE_BLEND_NEAR),0,1)*100) if phase=="APPROACH" else (100 if phase=="CENTER" else 0)
            print(f"G{gi+1:>2}/{len(path)}|{phase:<7}{ct:<4}|{gate['name']:<12}|"
                  f"d={dt:5.2f} p={gp:+5.2f} z={ze:+5.2f}|"
                  f"gate_blend={bl_pct:3d}%|"
                  f"s={spd:5.1f}/{max_speed:5.1f}|"
                  f"trn={gate['turn_angle']:4.0f}d|cv={cvl}|r={retries}")

            if cv2.waitKey(1)&0xFF==ord("q"): break
            el=time.time()-tl
            if CONTROL_DT-el>0: time.sleep(CONTROL_DT-el)
    finally: shutdown(client)

if __name__=="__main__": main()
