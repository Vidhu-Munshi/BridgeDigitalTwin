"""
Smart Bridge Digital Twin — Real-Time Structural & Fire Monitoring
================================================================
Arduino output format:
    Piezo:0.4523,Vibr:1,Fire:0

Sensors:
    Piezoelectric  → Analog  (A0) — normalized 0.0000-0.9999
    Vibration SW420→ Digital (D2) — 0=still, 1=vibrating
    Flame Sensor   → Digital (D3) — 0=fire!, 1=clear

4-State Logic  (V=Vibration, F=Flame):
    V=0 F=0  → GOOD       — all clear
    V=1 F=1  → EARTHQUAKE — vibration detected, no fire
    V=0 F=1  → FIRE       — fire detected, no vibration
    V=1 F=0  → EXTREME    — both vibration and fire

Piezo Health (SHI):
    >= 0.80  → GOOD
    0.60-0.79 → PROBLEM
    < 0.60   → CRITICAL
"""

from flask import Flask, jsonify, render_template_string, request
import numpy as np
import threading
import datetime
import time
import random
import collections
import os
import pickle

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── ML MODEL ──────────────────────────────────────────────────────────────────
ML_AVAILABLE = False
rf_reg = rf_cls = feature_columns = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        rf_reg          = bundle["regressor"]
        rf_cls          = bundle["classifier"]
        feature_columns = bundle["feature_columns"]
        ML_AVAILABLE    = True
        print(f"✅ ML model loaded — {len(feature_columns)} features")
    except Exception as e:
        print(f"⚠️  rf_model.pkl error: {e}")
else:
    print("ℹ️  rf_model.pkl not found — using SignalEngine for SHI")

# ── CONFIG ────────────────────────────────────────────────────────────────────
ARDUINO_PORT    = "COM3"
BAUD_RATE       = 9600
SAMPLE_INTERVAL = 0.5
WINDOW_SIZE     = 60
SMOOTH_N        = 5
MAX_LOG         = 500

app  = Flask(__name__)
lock = threading.Lock()
stop_event   = threading.Event()
arduino_conn = None

# ── STATE ─────────────────────────────────────────────────────────────────────
state = {
    "running":      False,
    "arduino_ok":   False,
    "mode":         "idle",
    "error_msg":    "",
    "session_id":   0,
    "sessions":     [],
    "current_logs": [],
    "latest":       {},
    "ml_active":    ML_AVAILABLE,
    "ts":    collections.deque(maxlen=WINDOW_SIZE),
    "piezo": collections.deque(maxlen=WINDOW_SIZE),
    "shi":   collections.deque(maxlen=WINDOW_SIZE),
}

# ── PIEZO HEALTH ──────────────────────────────────────────────────────────────
def piezo_health(p):
    if p >= 0.80:   return "GOOD",     "#00ff88"
    elif p >= 0.60: return "PROBLEM",  "#ffb800"
    else:           return "CRITICAL", "#ff3c3c"

# ── 4-STATE CLASSIFICATION ────────────────────────────────────────────────────
STATE_MAP = {
    (0, 0): ("GOOD",       "✅", "#00ff88", "All clear — structure normal"),
    (1, 0): ("EARTHQUAKE", "🌍", "#ffb800", "Vibration detected — possible seismic activity"),
    (0, 1): ("FIRE",       "🔥", "#ff6600", "Fire detected — evacuate immediately"),
    (1, 1): ("EXTREME",    "🚨", "#ff0033", "Vibration + Fire — extreme emergency"),
}

# ── SIGNAL ENGINE (fallback SHI when no ML) ───────────────────────────────────
class SignalEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ema   = None
        self.shi   = 0.84
        self.buf   = collections.deque(maxlen=8)
        self.alpha = 2 / (SMOOTH_N + 1)

    def _ema(self, prev, val):
        return float(val) if prev is None else self.alpha * val + (1 - self.alpha) * prev

    def push(self, piezo: float, vib: int, flame: int) -> float:
        self.ema = self._ema(self.ema, piezo)
        p = self.ema
        fire_alert = flame == 1
        vib_alert  = vib == 1
        fire_pen   = 0.35 if fire_alert else 0.0
        vib_pen    = 0.25 if vib_alert  else 0.0
        piezo_pen  = p * 0.15
        noise      = random.uniform(-0.006, 0.006)
        target     = float(np.clip(0.84 - vib_pen - fire_pen - piezo_pen + noise, 0.15, 0.95))
        alpha_shi  = 0.55 if (vib_alert or fire_alert) else 0.18
        self.shi   = float(np.clip(alpha_shi * target + (1 - alpha_shi) * self.shi, 0.15, 0.95))
        self.buf.append(self.shi)
        return self.shi

engine = SignalEngine()

# ── ML PREDICT ────────────────────────────────────────────────────────────────
def ml_predict(piezo: float, vib: int, flame: int, shi_prev: float) -> float:
    row = {col: 0.0 for col in feature_columns}
    for k in ["Piezoelectric_Vibration","Piezo_Vibration","Vibration_Intensity","Piezo","piezo"]:
        if k in row: row[k] = piezo * 1023
    for k in ["Modal_Frequency_Hz","Vibration_Frequency"]:
        if k in row: row[k] = 1.2 if vib else 2.5
    for k in ["SHI_Previous","Last_SHI","SHI_lag1"]:
        if k in row: row[k] = shi_prev
    X = np.array([[row[c] for c in feature_columns]])
    try:
        return float(np.clip(rf_reg.predict(X)[0], 0.15, 0.95))
    except:
        return shi_prev

# ── SIMULATOR ─────────────────────────────────────────────────────────────────
class Simulator:
    SCENARIOS = [
        # dur, vib, flame, piezo_range
        (40, 0, 1, (0.82, 0.96)),   # GOOD
        (25, 0, 1, (0.62, 0.79)),   # GOOD-ish
        (20, 1, 1, (0.70, 0.90)),   # EARTHQUAKE
        (15, 1, 1, (0.50, 0.70)),   # EARTHQUAKE + problem piezo
        (20, 0, 0, (0.80, 0.95)),   # FIRE
        (10, 1, 0, (0.30, 0.55)),   # EXTREME
        (35, 0, 1, (0.83, 0.95)),   # recovery
    ]
    def __init__(self):
        self.idx=0; self.count=0; self.limit=0
        self.vib=0; self.flame=1; self.pr=(0.8,0.95)
        self._next()

    def _next(self):
        dur,v,f,pr = self.SCENARIOS[self.idx % len(self.SCENARIOS)]
        self.limit=dur; self.vib=v; self.flame=f; self.pr=pr
        self.count=0; self.idx+=1

    def read(self):
        if self.count >= self.limit: self._next()
        self.count += 1
        p = float(np.clip(random.uniform(*self.pr)+random.gauss(0,.01), 0.0, 0.9999))
        return p, self.vib, self.flame

sim = Simulator()

# ── ARDUINO PARSER ────────────────────────────────────────────────────────────
def parse_arduino(line: str):
    """Parse 'Piezo:0.4523,Vibr:1,Fire:0' → (piezo, vib, flame)"""
    vals = {}
    for part in line.split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            try: vals[k.strip().upper()] = float(v.strip())
            except: pass
    return (
        float(np.clip(vals.get("PIEZO", 0.5), 0.0, 0.9999)),
        int(vals.get("VIBR", 0)),
        int(vals.get("FIRE", 1)),
    )

# ── ACQUISITION LOOP ──────────────────────────────────────────────────────────
def acquisition_loop(session_id: int, use_arduino: bool):
    global arduino_conn
    last_shi = 0.84
    print(f"[Session {session_id}] {'Arduino' if use_arduino else 'Simulation'} started")

    while not stop_event.is_set():
        try:
            if use_arduino and arduino_conn:
                raw = arduino_conn.readline().decode("utf-8", errors="ignore").strip()
                piezo, vib, flame = parse_arduino(raw)
            else:
                piezo, vib, flame = sim.read()

            # ── SHI ──────────────────────────────────────────────────────
            if ML_AVAILABLE:
                shi     = ml_predict(piezo, vib, flame, last_shi)
                ml_used = True
            else:
                shi     = engine.push(piezo, vib, flame)
                ml_used = False
            last_shi = shi

            # ── 4-State classification ────────────────────────────────────
            key = (int(vib), int(flame))
            state_name, icon, color, desc = STATE_MAP[key]

            # ── Piezo health zone ─────────────────────────────────────────
            p_health, p_color = piezo_health(piezo)

            # ── SHI trend ─────────────────────────────────────────────────
            with lock:
                shi_hist = list(state["shi"])
            if len(shi_hist) >= 4:
                d = np.mean(shi_hist[-2:]) - np.mean(shi_hist[:2])
                trend = "improving" if d > 0.003 else ("deteriorating" if d < -0.003 else "stable")
            else:
                trend = "stable"

            result = {
                "timestamp":    datetime.datetime.now().strftime("%H:%M:%S"),
                "piezo":        round(piezo, 4),
                "vib":          int(vib),
                "flame":        int(flame),
                "state_name":   state_name,
                "state_icon":   icon,
                "state_color":  color,
                "state_desc":   desc,
                "piezo_health": p_health,
                "piezo_color":  p_color,
                "shi":          round(shi, 4),
                "trend":        trend,
                "ml_used":      ml_used,
                "session_id":   session_id,
            }

            with lock:
                if state["running"] and state["session_id"] == session_id:
                    state["latest"] = result
                    state["current_logs"].append(result)
                    if len(state["current_logs"]) > MAX_LOG:
                        state["current_logs"].pop(0)
                    state["ts"].append(result["timestamp"])
                    state["piezo"].append(piezo)
                    state["shi"].append(shi)

        except Exception as e:
            print(f"[Session {session_id}] Error: {e}")

        time.sleep(SAMPLE_INTERVAL)

    print(f"[Session {session_id}] Stopped")

# ── SESSION CONTROL ───────────────────────────────────────────────────────────
def start_session(port, use_sim):
    global arduino_conn
    with lock:
        if state["running"]: return {"ok": False, "msg": "Already running"}

    arduino_ok = False; err_msg = ""
    if not use_sim and SERIAL_AVAILABLE:
        try:
            if arduino_conn:
                try: arduino_conn.close()
                except: pass
            arduino_conn = serial.Serial(port, BAUD_RATE, timeout=2)
            time.sleep(2); arduino_ok = True
        except Exception as e:
            err_msg = f"Cannot connect {port}: {e}. Using simulation."
    elif not SERIAL_AVAILABLE and not use_sim:
        err_msg = "pyserial not installed — using simulation."

    engine.reset(); sim.__init__()
    with lock:
        state["session_id"] += 1; sid = state["session_id"]
        state.update(running=True, arduino_ok=arduino_ok,
                     mode="running", error_msg=err_msg,
                     current_logs=[], latest={})
        for k in ("ts","piezo","shi"): state[k].clear()

    stop_event.clear()
    threading.Thread(target=acquisition_loop, args=(sid, arduino_ok), daemon=True).start()
    return {"ok": True, "session_id": sid, "arduino_ok": arduino_ok,
            "mode": "arduino" if arduino_ok else "simulation",
            "ml_active": ML_AVAILABLE,
            "msg": err_msg or f"Session {sid} started"}

def stop_session():
    with lock:
        if not state["running"]: return {"ok": False, "msg": "Not running"}
        sid  = state["session_id"]
        logs = list(state["current_logs"])
        states = [l["state_name"] for l in logs]
        summary = {
            "session_id":    sid,
            "start_time":    logs[0]["timestamp"] if logs else "—",
            "end_time":      datetime.datetime.now().strftime("%H:%M:%S"),
            "total_reads":   len(logs),
            "good_count":    states.count("GOOD"),
            "eq_count":      states.count("EARTHQUAKE"),
            "fire_count":    states.count("FIRE"),
            "extreme_count": states.count("EXTREME"),
            "avg_shi":       round(np.mean([l["shi"] for l in logs]), 4) if logs else 0,
            "min_shi":       round(min((l["shi"] for l in logs), default=0), 4),
            "ml_used":       logs[0]["ml_used"] if logs else False,
        }
        state["sessions"].append(summary)
        state.update(running=False, mode="stopped", current_logs=[])
    stop_event.set()
    if arduino_conn:
        try: arduino_conn.close()
        except: pass
    return {"ok": True, "session_id": sid, "total_reads": len(logs)}

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Bridge Digital Twin</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;800&display=swap');
:root{
  --bg:#020509; --bg2:#060b12; --bg3:#0a1019;
  --edge:#0e1e3a; --acc:#00c8ff; --acc2:#ff3c3c;
  --acc3:#ffb800; --acc4:#00ff88; --acc5:#bf7aff;
  --text:#c5d5ee; --dim:#324a68;
  --mono:'Share Tech Mono',monospace; --body:'Barlow',sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{background:var(--bg);color:var(--text);font-family:var(--body);min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.06) 2px,rgba(0,0,0,.06) 4px);}

header{background:linear-gradient(90deg,#010407,#050f1e,#010407);
  border-bottom:1px solid var(--edge);padding:10px 28px;
  display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100;
  box-shadow:0 0 40px rgba(0,200,255,.05);}
.logo{width:42px;height:42px;border-radius:8px;
  background:linear-gradient(135deg,#001020,#002550);
  border:1px solid var(--acc);display:flex;align-items:center;justify-content:center;
  font-size:1.3rem;box-shadow:0 0 16px rgba(0,200,255,.25);}
.hname{font-size:1.1rem;font-weight:800;letter-spacing:3px;color:var(--acc);text-transform:uppercase;}
.hsub{font-size:.6rem;color:var(--dim);letter-spacing:2px;margin-top:2px;}
.hright{margin-left:auto;display:flex;align-items:center;gap:10px;}
.pill{font-family:var(--mono);font-size:.68rem;padding:4px 12px;border-radius:3px;
  letter-spacing:2px;border:1px solid var(--dim);color:var(--dim);}
.pill.live{border-color:var(--acc4);color:var(--acc4);box-shadow:0 0 8px rgba(0,255,136,.2);}
.pill.sim{border-color:var(--acc3);color:var(--acc3);}
.pill.ml{border-color:var(--acc5);color:var(--acc5);}
.pulse{width:8px;height:8px;border-radius:50%;background:var(--dim);}
.pulse.on{background:var(--acc4);box-shadow:0 0 8px var(--acc4);animation:throb 1.2s infinite;}
@keyframes throb{0%,100%{opacity:1;}50%{opacity:.2;}}

.wrap{max-width:1440px;margin:0 auto;padding:16px 22px;}

/* ── CONTROL BAR ── */
.ctrl{background:var(--bg2);border:1px solid var(--edge);border-radius:10px;
  padding:14px 20px;margin-bottom:14px;display:flex;align-items:flex-end;gap:12px;flex-wrap:wrap;}
.flabel{font-size:.58rem;color:var(--dim);letter-spacing:2px;margin-bottom:4px;}
.finput{background:#040b15;border:1px solid var(--edge);border-radius:5px;
  color:var(--acc);padding:8px 12px;font-size:.82rem;font-family:var(--mono);width:170px;outline:none;}
.finput:focus{border-color:var(--acc);}
.btn{padding:9px 22px;border-radius:5px;font-size:.78rem;font-weight:700;letter-spacing:2px;
  cursor:pointer;border:none;font-family:var(--body);text-transform:uppercase;transition:all .15s;}
.btn-start{background:rgba(0,255,136,.1);color:var(--acc4);border:1px solid rgba(0,255,136,.5);}
.btn-start:hover:not(:disabled){background:rgba(0,255,136,.2);}
.btn-stop{background:rgba(255,60,60,.1);color:var(--acc2);border:1px solid rgba(255,60,60,.5);}
.btn-stop:hover:not(:disabled){background:rgba(255,60,60,.2);}
.btn-sim{background:rgba(255,184,0,.07);color:var(--acc3);border:1px solid rgba(255,184,0,.4);font-size:.72rem;}
.btn-sim:hover:not(:disabled){background:rgba(255,184,0,.15);}
.btn:disabled{opacity:.25;cursor:not-allowed;}
.cmsg{font-family:var(--mono);font-size:.7rem;color:var(--dim);margin-left:auto;max-width:380px;text-align:right;}
.cmsg.ok{color:var(--acc4);} .cmsg.err{color:var(--acc2);}

/* ── MAIN STATE PANEL ── */
.state-panel{
  display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;
}

/* Big 4-state card */
.state-card{
  background:var(--bg2);border:1px solid var(--edge);border-radius:12px;
  padding:24px 28px;display:flex;flex-direction:column;gap:14px;
  transition:all .35s;
}
.state-card.GOOD     {border-color:#00904a;background:linear-gradient(135deg,#001208,#001a0a);}
.state-card.EARTHQUAKE{border-color:#cc8400;background:linear-gradient(135deg,#170f00,#0d0700);animation:warn 2s infinite;}
.state-card.FIRE     {border-color:#ff4200;background:linear-gradient(135deg,#200400,#120200);animation:firepulse 1s infinite;}
.state-card.EXTREME  {border-color:#ff0030;background:linear-gradient(135deg,#230003,#140002);animation:crit .7s infinite;}
@keyframes warn{0%,100%{box-shadow:0 0 24px rgba(200,125,0,.1);}50%{box-shadow:0 0 60px rgba(200,125,0,.35);}}
@keyframes firepulse{0%,100%{box-shadow:0 0 30px rgba(255,70,0,.12);}50%{box-shadow:0 0 80px rgba(255,70,0,.5);}}
@keyframes crit{0%,100%{box-shadow:0 0 40px rgba(255,0,45,.15);}50%{box-shadow:0 0 110px rgba(255,0,45,.7);}}

.sc-icon{font-size:3rem;line-height:1;}
.sc-name{font-size:2rem;font-weight:800;letter-spacing:4px;margin-top:4px;}
.sc-desc{font-family:var(--mono);font-size:.72rem;color:var(--dim);margin-top:4px;}
.GOOD      .sc-name{color:#00ff88;}
.EARTHQUAKE .sc-name{color:#ffb800;}
.FIRE      .sc-name{color:#ff6800;}
.EXTREME   .sc-name{color:#ff0033;}

/* VF matrix */
.vf-matrix{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px;}
.vf-cell{
  border-radius:8px;padding:10px 14px;border:1px solid var(--edge);
  display:flex;flex-direction:column;gap:4px;cursor:default;
  transition:all .2s;
}
.vf-cell.active{border-color:currentColor;box-shadow:0 0 12px rgba(255,255,255,.1);}
.vf-cell .vf-label{font-size:.58rem;color:var(--dim);letter-spacing:2px;}
.vf-cell .vf-state{font-size:1rem;font-weight:800;font-family:var(--mono);letter-spacing:2px;}
.vf-cell .vf-meaning{font-size:.62rem;margin-top:2px;}

.cell-good  {background:rgba(0,255,136,.06);color:#00ff88;}
.cell-eq    {background:rgba(255,184,0,.06);color:#ffb800;}
.cell-fire  {background:rgba(255,100,0,.08);color:#ff6600;}
.cell-ext   {background:rgba(255,0,48,.08); color:#ff0033;}

/* ── PIEZO + SHI RIGHT PANEL ── */
.right-panel{display:flex;flex-direction:column;gap:12px;}

/* Piezo card */
.piezo-card{
  background:var(--bg2);border:1px solid var(--edge);border-radius:12px;
  padding:20px 22px;flex:1;
}
.pc-head{font-size:.6rem;letter-spacing:2px;color:var(--dim);margin-bottom:14px;text-transform:uppercase;}
.pc-value{font-size:3.2rem;font-weight:800;font-family:var(--mono);line-height:1;transition:color .4s;}
.pc-zone{
  display:inline-block;margin-top:10px;padding:4px 18px;border-radius:20px;
  font-size:.72rem;font-weight:800;letter-spacing:2px;transition:all .3s;
}
.zone-GOOD    {background:rgba(0,255,136,.1);color:#00ff88;border:1px solid rgba(0,255,136,.3);}
.zone-PROBLEM {background:rgba(255,184,0,.1);color:#ffb800;border:1px solid rgba(255,184,0,.3);}
.zone-CRITICAL{background:rgba(255,60,60,.1);color:#ff3c3c;border:1px solid rgba(255,60,60,.3);}
.pc-bar-wrap{margin-top:14px;position:relative;}
.pc-bar-track{height:8px;border-radius:4px;background:#040b15;overflow:hidden;position:relative;}
.pc-bar-fill{height:100%;border-radius:4px;transition:width .6s ease,background .4s;}
.pc-thresholds{display:flex;justify-content:space-between;margin-top:4px;}
.pc-thresh{font-size:.58rem;color:var(--dim);font-family:var(--mono);}
.pc-zone-labels{display:flex;margin-top:10px;gap:2px;}
.pc-zl{flex:1;text-align:center;padding:5px 0;font-size:.58rem;font-weight:700;border-radius:4px;letter-spacing:1px;}
.pc-zl.g{background:rgba(0,255,136,.08);color:#00ff88;}
.pc-zl.p{background:rgba(255,184,0,.08);color:#ffb800;}
.pc-zl.c{background:rgba(255,60,60,.08);color:#ff3c3c;}

/* SHI mini card */
.shi-card{
  background:var(--bg2);border:1px solid var(--edge);border-radius:12px;
  padding:16px 22px;
}
.sh-head{font-size:.6rem;letter-spacing:2px;color:var(--dim);margin-bottom:10px;text-transform:uppercase;}
.sh-row{display:flex;align-items:center;gap:16px;}
.sh-val{font-size:2rem;font-weight:800;font-family:var(--mono);color:var(--acc);}
.sh-right{flex:1;}
.sh-bar-track{height:5px;background:#040b15;border-radius:3px;}
.sh-bar-fill{height:100%;border-radius:3px;transition:width .6s ease,background .4s;}
.sh-label{font-size:.62rem;color:var(--dim);margin-top:5px;font-family:var(--mono);}
.sh-ml{font-size:.6rem;color:var(--acc5);margin-top:3px;}

/* ── PIEZO GRAPH ── */
.piezo-graph-card{
  background:var(--bg2);border:1px solid var(--edge);border-radius:12px;
  padding:16px 22px;margin-bottom:14px;
}
.pg-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}
.pg-title{font-size:.62rem;color:var(--acc);letter-spacing:2px;text-transform:uppercase;}
.pg-badge{font-family:var(--mono);font-size:.65rem;padding:2px 10px;
  border-radius:3px;border:1px solid var(--edge);color:var(--dim);}
.piezo-graph-card canvas{width:100%!important;height:160px!important;}

/* ── SESSIONS ── */
.sessions-wrap{background:var(--bg2);border:1px solid var(--edge);border-radius:12px;
  padding:16px 18px;margin-bottom:14px;}
.sec-hd{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}
.sec-hd h3{font-size:.62rem;color:var(--acc);letter-spacing:2px;}
.badge{background:#040b15;border:1px solid var(--edge);border-radius:3px;
  padding:2px 10px;font-size:.63rem;color:var(--dim);font-family:var(--mono);}
.scard{background:#040b15;border:1px solid var(--edge);border-radius:7px;
  padding:12px 16px;margin-bottom:8px;display:flex;align-items:center;gap:18px;}
.scard:last-child{margin-bottom:0;}
.scard-id{font-family:var(--mono);font-size:.78rem;color:var(--acc);min-width:90px;}
.scard-time{font-size:.65rem;color:var(--dim);}
.scard-stats{display:flex;gap:14px;margin-left:auto;}
.ss{text-align:center;}
.ss .v{font-size:1.05rem;font-weight:700;}
.ss .l{font-size:.56rem;color:var(--dim);letter-spacing:1px;}

/* ── LOGS ── */
.logs-wrap{background:var(--bg2);border:1px solid var(--edge);border-radius:12px;padding:16px 18px;}
table{width:100%;border-collapse:collapse;font-size:.72rem;}
thead th{text-align:left;padding:5px 8px;border-bottom:1px solid var(--edge);
  font-size:.58rem;letter-spacing:2px;color:var(--dim);text-transform:uppercase;}
tbody tr{border-bottom:1px solid #07101c;}
tbody tr:hover{background:#05101e;}
td{padding:5px 8px;font-family:var(--mono);color:#7a9abf;}
.tag{display:inline-block;padding:2px 10px;border-radius:4px;font-size:.62rem;font-weight:800;letter-spacing:1px;}
.tag.GOOD      {background:rgba(0,255,136,.07);color:#00ff88;border:1px solid rgba(0,255,136,.2);}
.tag.EARTHQUAKE{background:rgba(255,184,0,.07);color:#ffb800;border:1px solid rgba(255,184,0,.2);}
.tag.FIRE      {background:rgba(255,100,0,.08);color:#ff6600;border:1px solid rgba(255,100,0,.2);}
.tag.EXTREME   {background:rgba(255,0,50,.08);color:#ff0033;border:1px solid rgba(255,0,50,.2);animation:throb .8s infinite;}
.ptag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.6rem;font-weight:700;}
.ptag.GOOD    {color:#00ff88;} .ptag.PROBLEM{color:#ffb800;} .ptag.CRITICAL{color:#ff3c3c;}

footer{text-align:center;padding:14px;color:var(--dim);font-size:.6rem;
  border-top:1px solid var(--edge);letter-spacing:2px;margin-top:8px;}
.toast{position:fixed;bottom:22px;right:22px;z-index:9998;
  background:var(--bg3);border:1px solid var(--edge);border-radius:7px;
  padding:10px 18px;font-size:.78rem;max-width:360px;display:none;
  box-shadow:0 4px 30px rgba(0,0,0,.5);}
.toast.show{display:block;}
.toast.ok{border-color:var(--acc4);color:var(--acc4);}
.toast.err{border-color:var(--acc2);color:var(--acc2);}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--edge);border-radius:3px;}
</style>
</head>
<body>

<header>
  <div class="logo">🛡</div>
  <div>
    <div class="hname">Smart Bridge Digital Twin</div>
    <div class="hsub">Real-Time Structural &amp; Fire Monitoring · Piezo(Analog) + SW420(Digital) + Flame(Digital)</div>
  </div>
  <div class="hright">
    <span class="pill ml" id="mlPill" style="display:none">🧠 ML</span>
    <span class="pill" id="modePill">IDLE</span>
    <div class="pulse" id="pulse"></div>
  </div>
</header>

<div class="wrap">

  <!-- CONTROL -->
  <div class="ctrl">
    <div>
      <div class="flabel">COM Port</div>
      <input class="finput" id="portInput" value="COM3" placeholder="COM3 or /dev/ttyUSB0">
    </div>
    <button class="btn btn-start" id="btnStart" onclick="startSess()">▶ START</button>
    <button class="btn btn-stop"  id="btnStop"  onclick="stopSess()" disabled>■ STOP</button>
    <button class="btn btn-sim"   id="btnSim"   onclick="startSim()">⚙ SIMULATE</button>
    <span class="cmsg" id="cmsg">Format: Piezo:0.4523,Vibr:1,Fire:0 · Press START</span>
  </div>

  <!-- MAIN STATE PANEL -->
  <div class="state-panel">

    <!-- LEFT: Big state card + VF matrix -->
    <div class="state-card" id="stateCard">
      <div>
        <div class="sc-icon" id="scIcon">🛡</div>
        <div class="sc-name" id="scName">STANDBY</div>
        <div class="sc-desc" id="scDesc">System idle — awaiting session</div>
      </div>

      <!-- V/F matrix: 4 states -->
      <div class="vf-matrix">
        <div class="vf-cell cell-good" id="cell-00">
          <div class="vf-label">V=0  F=1</div>
          <div class="vf-state">✅ GOOD</div>
          <div class="vf-meaning">All clear</div>
        </div>
        <div class="vf-cell cell-eq" id="cell-10">
          <div class="vf-label">V=1  F=1</div>
          <div class="vf-state">🌍 EARTHQUAKE</div>
          <div class="vf-meaning">Vibration detected</div>
        </div>
        <div class="vf-cell cell-fire" id="cell-01">
          <div class="vf-label">V=0  F=0</div>
          <div class="vf-state">🔥 FIRE</div>
          <div class="vf-meaning">Fire detected</div>
        </div>
        <div class="vf-cell cell-ext" id="cell-11">
          <div class="vf-label">V=1  F=0</div>
          <div class="vf-state">🚨 EXTREME</div>
          <div class="vf-meaning">Both alerts</div>
        </div>
      </div>

      <!-- Live V and F digital values -->
      <div style="display:flex;gap:10px;margin-top:4px;">
        <div style="flex:1;background:#040b15;border:1px solid var(--edge);border-radius:8px;padding:10px 14px;text-align:center;">
          <div style="font-size:.58rem;color:var(--dim);letter-spacing:2px;margin-bottom:6px;">VIBRATION (V)</div>
          <div style="font-size:2.2rem;font-weight:800;font-family:var(--mono);" id="vibrDigit">—</div>
          <div style="font-size:.65rem;margin-top:4px;" id="vibrLabel">—</div>
        </div>
        <div style="flex:1;background:#040b15;border:1px solid var(--edge);border-radius:8px;padding:10px 14px;text-align:center;">
          <div style="font-size:.58rem;color:var(--dim);letter-spacing:2px;margin-bottom:6px;">FLAME (F)</div>
          <div style="font-size:2.2rem;font-weight:800;font-family:var(--mono);" id="flameDigit">—</div>
          <div style="font-size:.65rem;margin-top:4px;" id="flameLabel">—</div>
        </div>
      </div>
    </div>

    <!-- RIGHT: Piezo + SHI -->
    <div class="right-panel">

      <!-- Piezo Card -->
      <div class="piezo-card">
        <div class="pc-head">📡 Piezoelectric Sensor (Analog) — Normalized Reading</div>
        <div class="pc-value" id="piezoVal">—</div>
        <div>
          <span class="pc-zone" id="piezoZone">—</span>
        </div>
        <div class="pc-bar-wrap">
          <div class="pc-bar-track">
            <div class="pc-bar-fill" id="piezoBar" style="width:0%;background:#00ff88"></div>
          </div>
          <div class="pc-thresholds">
            <span class="pc-thresh">0.00</span>
            <span class="pc-thresh">0.60</span>
            <span class="pc-thresh">0.80</span>
            <span class="pc-thresh">1.00</span>
          </div>
          <div class="pc-zone-labels">
            <div class="pc-zl c" style="flex:0.6">CRITICAL</div>
            <div class="pc-zl p" style="flex:0.2">PROBLEM</div>
            <div class="pc-zl g" style="flex:0.2">GOOD</div>
          </div>
        </div>
      </div>

      <!-- SHI Card -->
      <div class="shi-card">
        <div class="sh-head">🧠 Structural Health Index (SHI)</div>
        <div class="sh-row">
          <div class="sh-val" id="shiVal">—</div>
          <div class="sh-right">
            <div class="sh-bar-track">
              <div class="sh-bar-fill" id="shiFill" style="width:0%"></div>
            </div>
            <div class="sh-label" id="shiLabel">Awaiting data…</div>
            <div class="sh-ml"   id="shiMl"></div>
          </div>
        </div>
      </div>

    </div>
  </div>

  <!-- PIEZO LIVE GRAPH -->
  <div class="piezo-graph-card">
    <div class="pg-header">
      <div class="pg-title">📈 Piezoelectric Sensor — Live Value &amp; SHI Trend</div>
      <div class="pg-badge" id="pgBadge">—</div>
    </div>
    <canvas id="chartDual"></canvas>
  </div>

  <!-- COMPLETED SESSIONS -->
  <div class="sessions-wrap" id="sessPanel" style="display:none">
    <div class="sec-hd">
      <h3>📂 COMPLETED SESSIONS</h3>
      <span class="badge" id="sessCount">0</span>
    </div>
    <div id="sessList"></div>
  </div>

  <!-- LIVE LOG -->
  <div class="logs-wrap">
    <div class="sec-hd">
      <h3>📋 LIVE LOG</h3>
      <span class="badge" id="logBadge">0 entries</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Piezo</th><th>Piezo Health</th>
          <th>V</th><th>F</th>
          <th>State</th><th>SHI</th><th>Trend</th>
        </tr>
      </thead>
      <tbody id="logBody"></tbody>
    </table>
  </div>

</div>

<footer>
  Smart Bridge Digital Twin &nbsp;·&nbsp;
  J.C. Bose University of Science &amp; Technology, YMCA &nbsp;·&nbsp;
  Piezo (Analog A0) · SW420 Vibration (Digital D2) · Flame Sensor (Digital D3)
</footer>

<div class="toast" id="toast"></div>

<script>
// ── DUAL CHART (Piezo + SHI) ─────────────────────────────────────────────────
const ctx = document.getElementById('chartDual').getContext('2d');
const gP = ctx.createLinearGradient(0,0,0,160);
gP.addColorStop(0,'rgba(0,200,255,.2)'); gP.addColorStop(1,'transparent');
const gS = ctx.createLinearGradient(0,0,0,160);
gS.addColorStop(0,'rgba(0,255,136,.15)'); gS.addColorStop(1,'transparent');

const chartDual = new Chart(ctx, {
  type:'line',
  data:{
    labels:[],
    datasets:[
      {label:'Piezo',data:[],borderColor:'#00c8ff',backgroundColor:gP,
       borderWidth:2,pointRadius:0,tension:.38,fill:true,yAxisID:'yP'},
      {label:'SHI',  data:[],borderColor:'#00ff88',backgroundColor:gS,
       borderWidth:1.5,pointRadius:0,tension:.38,fill:true,yAxisID:'yS',borderDash:[4,3]},
    ]
  },
  options:{
    animation:false,responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:true,labels:{color:'#3a5870',font:{size:10},boxWidth:12}}},
    scales:{
      x:{ticks:{color:'#2d4560',font:{size:9},maxTicksLimit:10},grid:{color:'#081220'}},
      yP:{min:0,max:1,position:'left',
        ticks:{color:'#00c8ff',font:{size:9},maxTicksLimit:5},grid:{color:'#081220'}},
      yS:{min:0,max:1,position:'right',
        ticks:{color:'#00ff88',font:{size:9},maxTicksLimit:5},grid:{display:false}},
    }
  }
});

function pushChart(ts, piezo, shi) {
  const WINDOW = 60;
  chartDual.data.labels.push(ts);
  chartDual.data.datasets[0].data.push(piezo);
  chartDual.data.datasets[1].data.push(shi);
  if(chartDual.data.labels.length > WINDOW){
    chartDual.data.labels.shift();
    chartDual.data.datasets[0].data.shift();
    chartDual.data.datasets[1].data.shift();
  }
  chartDual.update('none');
}

// ── VF CELL HIGHLIGHT ────────────────────────────────────────────────────────
const cellMap = {
  '01': 'cell-00',   // V=0,F=1 → GOOD
  '11': 'cell-10',   // V=1,F=1 → EARTHQUAKE
  '00': 'cell-01',   // V=0,F=0 → FIRE
  '10': 'cell-11',   // V=1,F=0 → EXTREME
};

function highlightCell(vib, flame) {
  Object.values(cellMap).forEach(id => {
    const el = document.getElementById(id);
    el.style.opacity = '.35';
    el.style.transform = 'scale(.97)';
  });
  const key = `${vib}${flame}`;
  const activeId = cellMap[key];
  if(activeId){
    const el = document.getElementById(activeId);
    el.style.opacity = '1';
    el.style.transform = 'scale(1.03)';
    el.style.boxShadow = '0 0 18px rgba(255,255,255,.08)';
  }
}

// ── UPDATE UI ────────────────────────────────────────────────────────────────
function updateUI(d) {
  if(!d || !d.state_name) return;

  // State card
  const sc = document.getElementById('stateCard');
  sc.className = 'state-card ' + d.state_name;
  document.getElementById('scIcon').textContent = d.state_icon;
  document.getElementById('scName').textContent = d.state_name;
  document.getElementById('scDesc').textContent = d.state_desc + '  ·  ' + d.timestamp;

  // VF digit displays
  const vEl = document.getElementById('vibrDigit');
  vEl.textContent = d.vib;
  vEl.style.color = d.vib === 1 ? '#ffb800' : '#00ff88';
  document.getElementById('vibrLabel').textContent = d.vib === 1 ? '⚡ VIBRATING' : 'STILL';
  document.getElementById('vibrLabel').style.color = d.vib === 1 ? '#ffb800' : '#00ff88';

  const fEl = document.getElementById('flameDigit');
  fEl.textContent = d.flame;
  fEl.style.color = d.flame === 1 ? '#ff4400' : '#00ff88';
  document.getElementById('flameLabel').textContent = d.flame === 1 ? '🔥 FIRE!' : 'CLEAR';
  document.getElementById('flameLabel').style.color = d.flame === 0 ? '#ff4400' : '#00ff88';

  // Highlight active VF cell
  highlightCell(d.vib, d.flame);

  // Piezo
  const p = d.piezo;
  document.getElementById('piezoVal').textContent = p.toFixed(4);
  document.getElementById('piezoVal').style.color = d.piezo_color;
  const zone = document.getElementById('piezoZone');
  zone.textContent = d.piezo_health === 'GOOD' ? '🟢 GOOD  (≥ 0.80)' :
                     d.piezo_health === 'PROBLEM' ? '🟡 PROBLEM  (0.60–0.79)' : '🔴 CRITICAL  (< 0.60)';
  zone.className = 'pc-zone zone-' + d.piezo_health;
  document.getElementById('piezoBar').style.width = (p*100).toFixed(1)+'%';
  document.getElementById('piezoBar').style.background = d.piezo_color;

  // SHI
  document.getElementById('shiVal').textContent = d.shi.toFixed(4);
  const fill = document.getElementById('shiFill');
  fill.style.width = (d.shi*100).toFixed(1)+'%';
  fill.style.background = d.shi >= 0.80 ? '#00ff88' : (d.shi >= 0.60 ? '#ffb800' : '#ff3c3c');
  document.getElementById('shiLabel').textContent =
    `Trend: ${d.trend.toUpperCase()}  ·  ${d.shi >= 0.80 ? '🟢 HEALTHY' : d.shi >= 0.60 ? '🟡 STRESSED' : '🔴 CRITICAL'}`;
  document.getElementById('shiMl').textContent =
    d.ml_used ? '🤖 RandomForest ML prediction' : '⚙️ SignalEngine (place rf_model.pkl here to enable ML)';

  // Graph badge
  document.getElementById('pgBadge').textContent = `Piezo: ${p.toFixed(4)}  SHI: ${d.shi.toFixed(4)}`;

  // Chart
  pushChart(d.timestamp, p, d.shi);
}

// ── POLLING ──────────────────────────────────────────────────────────────────
let polling=false, pollTimer=null;

async function pollLatest(){
  try{const r=await fetch('/api/latest');const d=await r.json();if(d&&d.state_name)updateUI(d);}
  catch(e){}
}

async function pollLogs(){
  try{
    const r=await fetch('/api/logs');const logs=await r.json();
    document.getElementById('logBadge').textContent=logs.length+' entries';
    document.getElementById('logBody').innerHTML=[...logs].reverse().slice(0,60).map(l=>`<tr>
      <td>${l.timestamp}</td>
      <td style="color:${l.piezo>=0.80?'#00ff88':l.piezo>=0.60?'#ffb800':'#ff3c3c'}">${l.piezo.toFixed(4)}</td>
      <td><span class="ptag ${l.piezo_health}">${l.piezo_health}</span></td>
      <td style="color:${l.vib?'#ffb800':'#00ff88'}">${l.vib}</td>
      <td style="color:${l.flame===0?'#ff4400':'#00ff88'}">${l.flame}</td>
      <td><span class="tag ${l.state_name}">${l.state_icon} ${l.state_name}</span></td>
      <td>${l.shi}</td>
      <td>${l.trend}</td>
    </tr>`).join('');
  }catch(e){}
}

async function pollSessions(){
  try{
    const r=await fetch('/api/sessions');const sess=await r.json();
    const panel=document.getElementById('sessPanel');
    if(!sess.length){panel.style.display='none';return;}
    panel.style.display='block';
    document.getElementById('sessCount').textContent=sess.length+' sessions';
    document.getElementById('sessList').innerHTML=[...sess].reverse().map(s=>`
      <div class="scard">
        <span class="scard-id">Session #${s.session_id}</span>
        <span class="scard-time">${s.start_time} → ${s.end_time} · ${s.total_reads} reads${s.ml_used?' · 🤖':''}</span>
        <div class="scard-stats">
          <div class="ss"><div class="v" style="color:#00ff88">${s.good_count}</div><div class="l">GOOD</div></div>
          <div class="ss"><div class="v" style="color:#ffb800">${s.eq_count}</div><div class="l">EQ</div></div>
          <div class="ss"><div class="v" style="color:#ff6600">${s.fire_count}</div><div class="l">FIRE</div></div>
          <div class="ss"><div class="v" style="color:#ff0033">${s.extreme_count}</div><div class="l">EXTREME</div></div>
          <div class="ss"><div class="v" style="color:var(--acc)">${s.avg_shi}</div><div class="l">AVG SHI</div></div>
          <div class="ss"><div class="v" style="color:#cc4466">${s.min_shi}</div><div class="l">MIN SHI</div></div>
        </div>
      </div>`).join('');
  }catch(e){}
}

function startPolling(){
  polling=true;
  (async function loop(){
    if(!polling) return;
    await Promise.all([pollLatest(),pollLogs(),pollSessions()]);
    pollTimer=setTimeout(loop,500);
  })();
}
function stopPolling(){polling=false;clearTimeout(pollTimer);}

function setUI(on){
  ['btnStart','btnSim'].forEach(id=>document.getElementById(id).disabled=on);
  document.getElementById('btnStop').disabled=!on;
  document.getElementById('portInput').disabled=on;
  document.getElementById('pulse').className='pulse'+(on?' on':'');
}

function toast(msg,type='ok'){
  const t=document.getElementById('toast');
  t.textContent=msg;t.className=`toast show ${type}`;
  clearTimeout(t._t);t._t=setTimeout(()=>t.className='toast',4500);
}

async function startSess(){
  const port=document.getElementById('portInput').value.trim()||'COM3';
  const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({port,simulation:false})});
  const d=await r.json();
  if(d.ok){
    setUI(true);
    const pill=document.getElementById('modePill');
    pill.className=d.mode==='arduino'?'pill live':'pill sim';
    pill.textContent=d.mode==='arduino'?'ARDUINO LIVE':'SIMULATION';
    if(d.ml_active) document.getElementById('mlPill').style.display='inline-block';
    document.getElementById('cmsg').className='cmsg ok';
    document.getElementById('cmsg').textContent=d.msg;
    startPolling();
  } else toast(d.msg,'err');
}

async function startSim(){
  const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({port:'',simulation:true})});
  const d=await r.json();
  if(d.ok){
    setUI(true);
    document.getElementById('modePill').className='pill sim';
    document.getElementById('modePill').textContent='SIMULATION';
    if(d.ml_active) document.getElementById('mlPill').style.display='inline-block';
    document.getElementById('cmsg').className='cmsg ok';
    document.getElementById('cmsg').textContent='Simulation running';
    startPolling();
  } else toast(d.msg,'err');
}

async function stopSess(){
  const r=await fetch('/api/stop',{method:'POST'});
  const d=await r.json();
  if(d.ok){
    stopPolling();setUI(false);
    document.getElementById('modePill').className='pill';
    document.getElementById('modePill').textContent='STOPPED';
    document.getElementById('mlPill').style.display='none';
    document.getElementById('cmsg').className='cmsg';
    document.getElementById('cmsg').textContent=`Session #${d.session_id} ended — ${d.total_reads} reads archived`;
    document.getElementById('stateCard').className='state-card';
    document.getElementById('scName').textContent='STANDBY';
    document.getElementById('scDesc').textContent='Session ended. Press START for new session.';
    document.getElementById('logBadge').textContent='0 entries';
    document.getElementById('logBody').innerHTML='';
    await pollSessions();
    toast(`Session #${d.session_id} archived — ${d.total_reads} reads`,'ok');
  }
}
</script>
</body>
</html>"""

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/start", methods=["POST"])
def api_start():
    data    = request.get_json(silent=True) or {}
    port    = data.get("port", ARDUINO_PORT)
    use_sim = data.get("simulation", False)
    return jsonify(start_session(port, use_sim))

@app.route("/api/stop", methods=["POST"])
def api_stop():
    return jsonify(stop_session())

@app.route("/api/latest")
def api_latest():
    with lock: return jsonify(state["latest"])

@app.route("/api/logs")
def api_logs():
    with lock: return jsonify(state["current_logs"])

@app.route("/api/sessions")
def api_sessions():
    with lock: return jsonify(state["sessions"])

@app.route("/api/graph")
def api_graph():
    with lock:
        return jsonify({
            "ts":    list(state["ts"]),
            "piezo": list(state["piezo"]),
            "shi":   list(state["shi"]),
        })

@app.route("/api/status")
def api_status():
    with lock:
        return jsonify({
            "running":    state["running"],
            "mode":       state["mode"],
            "arduino_ok": state["arduino_ok"],
            "session_id": state["session_id"],
            "ml_active":  state["ml_active"],
        })

if __name__ == "__main__":
    print("\n" + "="*62)
    print("  Smart Bridge Digital Twin — http://localhost:5000")
    print(f"  ML: {'✅ rf_model.pkl LOADED' if ML_AVAILABLE else '⚠️  Not found — using SignalEngine'}")
    print("  Arduino: Piezo:0.4523,Vibr:1,Fire:0")
    print("  4-States: V0F1=GOOD · V1F1=EARTHQUAKE · V0F0=FIRE · V1F0=EXTREME")
    print("="*62 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
