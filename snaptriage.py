#!/usr/bin/env python3
"""
SnapTriage - screenshot support-ticket triage, engineered.
GDG KAU - AI Engineering Workshop

Pipeline (each job goes to the cheapest component that can do it):
  1. READ    cheap vision model transcribes the screenshot (OCR);
             a stronger vision model is used only if OCR finds too little text
  2. TRIAGE  Jev call 1: category, severity, team (typed answers + confidence)
             -> the ticket is routed to the team right away
  3. PICK    Jev call 2: choose one known fix from that category, or "none"
  4. ANSWER  confident match -> approved template reply (zero LLM calls)
             otherwise       -> a top-tier LLM drafts the reply;
                                IT can approve it into the known-issues library

Setup:   put OPENROUTER_API_KEY=sk-or-... in a .env file next to this script
Run:     python snaptriage.py              -> web visualiser on http://localhost:8000
         python snaptriage.py image.png    -> run one screenshot in the terminal
No pip installs needed: standard library only (Python 3.9+).
"""

import base64, json, mimetypes, os, re, socket, sys, threading, time, uuid, webbrowser
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Force IPv4: on networks where IPv6 is advertised but not routed, every API
# call would otherwise wait out the full timeout before falling back.
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only(host, port, family=0, *args, **kwargs):
    return _orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)
socket.getaddrinfo = _ipv4_only

HERE = Path(__file__).resolve().parent
SAMPLES_DIR = HERE / "samples"
LIBRARY_FILE = HERE / "known_issues.json"
PORT = int(os.environ.get("PORT", "8000"))

# ----------------------------------------------------------------------------
# Config: thresholds and models (edit freely during the workshop)
# ----------------------------------------------------------------------------
TEAM_THRESHOLD = 0.60    # route to the predicted team at or above this confidence
FIX_THRESHOLD = 0.85     # auto-send a template only at or above this confidence
MIN_OCR_CHARS = 25       # less text than this -> use the stronger vision model

# First id that exists on OpenRouter wins; the regex is the safety net if the
# catalog has moved on. Checked once at startup against /api/v1/models.
MODEL_PREFS = {
    "ocr":      (["google/gemini-3.5-flash-lite", "google/gemini-2.5-flash-lite"], r"^google/gemini-[\d.]+-flash-lite$"),
    "vision":   (["google/gemini-3.8-flash", "google/gemini-3.5-flash", "google/gemini-2.5-flash"], r"^google/gemini-[\d.]+-flash$"),
    "strong":   (["anthropic/claude-opus-5.5", "anthropic/claude-opus-5", "anthropic/claude-opus-4.8"], r"^anthropic/claude-opus-[\d.]+$"),
    "baseline": (["google/gemini-3.8-flash", "google/gemini-3.5-flash", "google/gemini-2.5-flash"], r"^google/gemini-[\d.]+-flash$"),
}
JEV_MODEL = "typesafe/jev-1.13"

# ----------------------------------------------------------------------------
# Domain: categories, teams, and the starting known-issues library
# ----------------------------------------------------------------------------
CATEGORIES = {
    "auth":         "Login, sessions, passwords, locked accounts, single sign-on.",
    "network":      "Campus Wi-Fi, internet access, VPN, certificates and browser connection errors.",
    "lms":          "The e-learning platform: courses, assignment uploads, quizzes, submissions.",
    "registration": "Course registration: adding or dropping courses, sections, waitlists, holds, prerequisites.",
    "email":        "University email: mailbox, sending, receiving, storage quota.",
    "other":        "Anything else, including errors in the student's own code or unrelated apps.",
}
TEAMS = {
    "it_helpdesk":  ("IT Helpdesk",       "Accounts, passwords, email and general campus IT."),
    "network":      ("Network Team",      "Wi-Fi, internet connectivity, VPN, certificates."),
    "elearning":    ("E-Learning Team",   "The learning management system and course tools."),
    "registrar":    ("Registrar",         "Course registration, sections, waitlists, academic holds."),
    "general":      ("General Triage",    "Unclear or unrelated requests that need a human to look first."),
}
SEVERITY = [
    "Low: a minor inconvenience, the student can keep working",
    "Medium: blocks one task, but there is no deadline pressure",
    "High: blocks coursework or registration with a deadline soon",
    "Critical: a service is down or many students are affected",
]

DEFAULT_LIBRARY = [
    {"id": "AUTH-01", "category": "auth", "title": "Session timed out",
     "when": "The page says the session expired or shows error 401 and asks to log in again.",
     "reply": "Your login session timed out. Log out, clear cookies for the portal (or use a private window), and sign in again. If it keeps happening within a few minutes, reply to this ticket."},
    {"id": "AUTH-02", "category": "auth", "title": "Account locked after failed sign-ins",
     "when": "The account is locked or disabled after too many failed sign-in attempts.",
     "reply": "Your account was locked after several failed sign-in attempts. It unlocks automatically after 30 minutes. If you have forgotten your password, use 'Forgot password' on the sign-in page instead of trying again."},
    {"id": "AUTH-03", "category": "auth", "title": "Forgotten or expired password",
     "when": "The student forgot the password, or the password has expired and must be changed.",
     "reply": "Use 'Forgot password' on the sign-in page to set a new password with your registered phone or recovery email. If you no longer have access to either, visit the IT Helpdesk with your student ID."},
    {"id": "NET-01", "category": "network", "title": "Campus Wi-Fi authentication failed",
     "when": "The device cannot join the campus Wi-Fi and reports an authentication or 802.1X failure.",
     "reply": "Open your Wi-Fi settings, choose 'Forget this network' for the campus Wi-Fi, then reconnect and enter your full university username and current password. Check that your device date and time are set automatically."},
    {"id": "NET-02", "category": "network", "title": "Connected to Wi-Fi but no internet",
     "when": "The device is connected to Wi-Fi but websites do not load or it shows 'No internet'.",
     "reply": "Open any website to trigger the campus sign-in page and log in. If nothing appears, turn Wi-Fi off and on, or forget and rejoin the network."},
    {"id": "LMS-01", "category": "lms", "title": "Upload exceeds the file size limit",
     "when": "An assignment upload fails because the file is larger than the maximum allowed size.",
     "reply": "The file is larger than the upload limit. Compress it (remove videos or large images, or export a smaller PDF), or upload it to your university cloud drive and submit the share link. Ask your instructor if the limit must be raised."},
    {"id": "LMS-02", "category": "lms", "title": "Submission closed after the deadline",
     "when": "The assignment no longer accepts submissions because the due date has passed.",
     "reply": "The submission window has closed. Only your instructor can reopen it or grant an extension, so please contact them directly with your file attached."},
    {"id": "REG-01", "category": "registration", "title": "Section is full",
     "when": "The student cannot add a course section because it is full or has reached capacity.",
     "reply": "This section is full. Join the waitlist from the same page; seats are offered automatically in waitlist order. You can also check other sections of the same course."},
    {"id": "REG-02", "category": "registration", "title": "Prerequisite or registration hold",
     "when": "Registration is blocked by a missing prerequisite or a hold on the student's account.",
     "reply": "Registration is blocked by a prerequisite or an account hold. Check 'Holds' in the portal to see who placed it, and contact that office or your academic advisor to clear it."},
    {"id": "MAIL-01", "category": "email", "title": "Mailbox is full",
     "when": "Email cannot be sent or received because the mailbox storage quota is full.",
     "reply": "Your mailbox has reached its storage limit. Empty 'Deleted Items', delete large attachments, then wait a few minutes for the quota to refresh."},
]


def load_library():
    if not LIBRARY_FILE.exists():
        LIBRARY_FILE.write_text(json.dumps(DEFAULT_LIBRARY, indent=2, ensure_ascii=False), encoding="utf-8")
    return json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))


LIB_LOCK = threading.Lock()


def add_to_library(category, title, when, reply):
    with LIB_LOCK:
        lib = load_library()
        prefix = {"auth": "AUTH", "network": "NET", "lms": "LMS", "registration": "REG",
                  "email": "MAIL"}.get(category, "OTHER")
        nums = [int(f["id"].split("-")[1]) for f in lib if f["id"].startswith(prefix + "-")]
        fix = {"id": f"{prefix}-{(max(nums) if nums else 0) + 1:02d}", "category": category,
               "title": title.strip(), "when": when.strip(), "reply": reply.strip(), "approved_from_llm": True}
        lib.append(fix)
        LIBRARY_FILE.write_text(json.dumps(lib, indent=2, ensure_ascii=False), encoding="utf-8")
        return fix


# ----------------------------------------------------------------------------
# OpenRouter client (stdlib only)
# ----------------------------------------------------------------------------
def load_env():
    env_file = HERE / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai").rstrip("/")  # optional override for testing


class APIError(Exception):
    pass


def _post(path, body, timeout=120):
    data = json.dumps(body).encode()
    for attempt in range(3):
        req = urllib.request.Request(BASE + path, data=data, method="POST", headers={
            "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
            "HTTP-Referer": "https://gdg.community.dev", "X-Title": "SnapTriage (GDG KAU workshop)"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:400]
            if e.code in (429, 500, 502, 503, 529) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise APIError(f"{path} -> HTTP {e.code}: {msg}")
        except urllib.error.URLError as e:
            if attempt < 2:
                time.sleep(1)
                continue
            raise APIError(f"{path} -> {e.reason}")


def resolve_models():
    """Pick real model ids from the live catalog, falling back to the first preference."""
    chosen = {k: v[0][0] for k, v in MODEL_PREFS.items()}
    try:
        with urllib.request.urlopen(BASE + "/api/v1/models", timeout=20) as r:
            models = json.loads(r.read().decode())["data"]
    except Exception as e:
        print(f"  (could not read the model catalog: {e}; using defaults)")
        return chosen
    by_id = {m["id"]: m for m in models}
    for role, (prefs, pattern) in MODEL_PREFS.items():
        hit = next((p for p in prefs if p in by_id), None)
        if not hit:
            cands = sorted((m for m in models if re.match(pattern, m["id"])),
                           key=lambda m: m.get("created", 0), reverse=True)
            hit = cands[0]["id"] if cands else prefs[0]
        chosen[role] = hit
    return chosen


MODELS = {}


def chat(model, messages, max_tokens=800):
    """One chat call. Returns (text, cost_usd, ms)."""
    t0 = time.time()
    res = _post("/api/v1/chat/completions", {"model": model, "messages": messages,
                                              "max_tokens": max_tokens, "temperature": 0,
                                              "usage": {"include": True}})
    ms = (time.time() - t0) * 1000
    if "choices" not in res:
        raise APIError(f"{model}: unexpected response {str(res)[:300]}")
    content = res["choices"][0]["message"].get("content") or ""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content.strip(), float((res.get("usage") or {}).get("cost") or 0), ms


def decide(state, questions):
    """One Jev call. Returns (answers, cost_usd, ms)."""
    t0 = time.time()
    res = _post("/api/alpha/decisions", {"model": JEV_MODEL, "state": state, "questions": questions})
    ms = (time.time() - t0) * 1000
    if "answers" not in res:
        raise APIError(f"Jev: unexpected response {str(res)[:300]}")
    return res["answers"], float((res.get("usage") or {}).get("cost") or 0), ms


def image_part(b64, mime):
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise APIError(f"model did not return JSON: {text[:200]}")
    return json.loads(m.group(0))


# ----------------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------------
def run_pipeline(b64, mime, emit):
    ticket_id = "T-" + uuid.uuid4().hex[:6].upper()
    totals = {"cost": 0.0, "llm_calls": 0, "jev_calls": 0}
    t_start = time.time()
    emit({"type": "start", "ticket": ticket_id, "models": MODELS, "jev": JEV_MODEL})

    # 1. READ -----------------------------------------------------------------
    emit({"type": "stage", "stage": "read", "status": "running", "model": MODELS["ocr"]})
    text, cost, ms = chat(MODELS["ocr"], [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe all visible text in this screenshot exactly, top to bottom. "
                                 "Output only the text, no commentary. If there is no readable text, output nothing."},
        image_part(b64, mime)]}], max_tokens=600)
    totals["cost"] += cost
    read = {"text": text, "model": MODELS["ocr"], "ms": ms, "cost": cost, "fallback": False}
    if len(re.sub(r"\s+", "", text)) < MIN_OCR_CHARS:
        emit({"type": "stage", "stage": "read", "status": "fallback", "model": MODELS["vision"],
              "ocr_text": text, "ms": ms, "cost": cost})
        desc, cost2, ms2 = chat(MODELS["vision"], [{"role": "user", "content": [
            {"type": "text", "text": "This screenshot was sent to a university IT helpdesk but has very little text. "
                                     "Describe in 2-3 sentences what the screen shows and what seems to be wrong. "
                                     "Include any visible text verbatim."},
            image_part(b64, mime)]}], max_tokens=300)
        totals["cost"] += cost2
        totals["llm_calls"] += 1
        read.update({"text": desc, "ocr_text": text, "fallback": True, "fallback_model": MODELS["vision"],
                     "ms": ms + ms2, "cost": cost + cost2})
    emit({"type": "stage", "stage": "read", "status": "done", **read})
    ticket_text = read["text"] or "(no readable text)"

    # 2. TRIAGE (Jev call 1) ----------------------------------------------------
    emit({"type": "stage", "stage": "triage", "status": "running", "model": JEV_MODEL})
    answers, cost, ms = decide(
        {"source": "screenshot sent to a university IT helpdesk", "screen_text": ticket_text},
        {
            "category": {"type": "choice", "instructions": "What kind of problem is the student reporting?",
                         "criteria": CATEGORIES},
            "team": {"type": "choice", "instructions": "Which team should own this ticket?",
                     "criteria": {k: f"{v[0]}: {v[1]}" for k, v in TEAMS.items()}},
            "severity": {"type": "score", "instructions": "How severe is this problem for the student?",
                         "criteria": SEVERITY},
        })
    totals["cost"] += cost
    totals["jev_calls"] += 1
    cat, team, sev = answers["category"], answers["team"], answers["severity"]
    sev_idx = max(0, min(len(SEVERITY) - 1, round(float(sev.get("score", 0)))))
    triage = {"category": cat, "team": team, "severity": sev, "severity_label": SEVERITY[sev_idx].split(":")[0],
              "ms": ms, "cost": cost}
    emit({"type": "stage", "stage": "triage", "status": "done", **triage})

    # route right away: the ticket never waits on anything after this point
    if team["confidence"] >= TEAM_THRESHOLD:
        routed_to, why = TEAMS[team["choice"]][0], f"team confidence {team['confidence']:.2f} >= {TEAM_THRESHOLD}"
    else:
        routed_to, why = TEAMS["general"][0], f"team confidence {team['confidence']:.2f} < {TEAM_THRESHOLD}"
    emit({"type": "stage", "stage": "route", "status": "done", "team": routed_to, "why": why,
          "at_ms": (time.time() - t_start) * 1000})

    # 3. PICK A FIX (Jev call 2) ------------------------------------------------
    library = load_library()
    probs = sorted(cat.get("probabilities", {}).items(), key=lambda kv: -kv[1])
    cats = [cat["choice"]] if cat["confidence"] >= TEAM_THRESHOLD else [c for c, _ in probs[:2]]
    candidates = [f for f in library if f["category"] in cats]
    pick = None
    if candidates:
        emit({"type": "stage", "stage": "pick", "status": "running", "model": JEV_MODEL,
              "candidates": [f["id"] for f in candidates]})
        criteria = {f["id"]: f"{f['title']}: {f['when']}" for f in candidates}
        criteria["none"] = "None of the listed fixes clearly matches this exact problem, or the error is new or different."
        answers, cost, ms = decide(
            {"screen_text": ticket_text, "category": cat["choice"]},
            {"fix": {"type": "choice", "criteria": criteria,
                     "instructions": "Which known fix solves exactly the problem in this screenshot? "
                                     "Choose 'none' unless a fix clearly matches."}})
        totals["cost"] += cost
        totals["jev_calls"] += 1
        pick = answers["fix"]
        emit({"type": "stage", "stage": "pick", "status": "done", "fix": pick, "ms": ms, "cost": cost,
              "titles": {f["id"]: f["title"] for f in candidates}})
    else:
        emit({"type": "stage", "stage": "pick", "status": "skipped", "why": f"no known fixes for: {', '.join(cats)}"})

    # 4. ANSWER -----------------------------------------------------------------
    matched = pick and pick["choice"] != "none" and pick["confidence"] >= FIX_THRESHOLD
    if matched:
        fix = next(f for f in library if f["id"] == pick["choice"])
        reply = fix["reply"]
        emit({"type": "stage", "stage": "answer", "status": "done", "path": "template", "fix_id": fix["id"],
              "fix_title": fix["title"], "why": f"fix confidence {pick['confidence']:.2f} >= {FIX_THRESHOLD}"})
        draft = None
    else:
        why = ("no fix chosen" if not pick else
               "Jev chose 'none'" if pick["choice"] == "none" else
               f"fix confidence {pick['confidence']:.2f} < {FIX_THRESHOLD}")
        emit({"type": "stage", "stage": "answer", "status": "running", "path": "llm", "model": MODELS["strong"], "why": why})
        known = "\n".join(f"- {f['id']} {f['title']}: {f['when']}" for f in library if f["category"] in cats) or "- none"
        prompt = (
            "You are the support assistant of a university IT helpdesk. A student sent a screenshot.\n\n"
            f"Screen text / description:\n{ticket_text}\n\n"
            f"Triage: category={cat['choice']}, routed to {routed_to}, severity={triage['severity_label']}.\n"
            f"Known fixes in this area (none matched with confidence):\n{known}\n\n"
            "Write the reply the student receives: friendly, at most 110 words, concrete numbered steps, "
            "and say when to reply to the ticket if it does not work. Do not invent university-specific URLs or phone numbers.\n"
            "Also propose a reusable library entry so IT can approve it for future tickets.\n"
            'Return only JSON: {"reply": "...", "fix_title": "short title", "fix_when": "one sentence describing when this fix applies"}')
        text, cost, ms = chat(MODELS["strong"], [{"role": "user", "content": prompt}], max_tokens=900)
        totals["cost"] += cost
        totals["llm_calls"] += 1
        try:
            out = parse_json(text)
        except Exception:
            out = {"reply": text, "fix_title": "New issue", "fix_when": ticket_text[:160]}
        reply = out.get("reply", text)
        draft = {"category": cat["choice"] if cat["choice"] in CATEGORIES else "other",
                 "title": out.get("fix_title", "New issue"), "when": out.get("fix_when", ""), "reply": reply}
        emit({"type": "stage", "stage": "answer", "status": "done", "path": "llm", "model": MODELS["strong"],
              "why": why, "ms": ms, "cost": cost, "draft": draft})

    total_ms = (time.time() - t_start) * 1000
    result = {"type": "result", "ticket": ticket_id, "team": routed_to, "reply": reply,
              "path": "template" if matched else "llm", "total_ms": total_ms, **totals}
    emit(result)
    return result


def run_baseline(b64, mime, emit):
    """The fair baseline: one multimodal call that reads, decides and answers."""
    emit({"type": "baseline", "status": "running", "model": MODELS["baseline"]})
    try:
        teams = ", ".join(v[0] for v in TEAMS.values())
        text, cost, ms = chat(MODELS["baseline"], [{"role": "user", "content": [
            {"type": "text", "text":
                "You are a university IT helpdesk. Read this screenshot a student sent and return only JSON: "
                f'{{"category": one of {list(CATEGORIES)}, "team": one of [{teams}], '
                '"severity": "Low|Medium|High|Critical", "confidence": 0-1, '
                '"reply": "friendly reply to the student, max 110 words, numbered steps"}'},
            image_part(b64, mime)]}], max_tokens=700)
        try:
            out = parse_json(text)
        except Exception:
            out = {"reply": text}
        emit({"type": "baseline", "status": "done", "model": MODELS["baseline"], "ms": ms, "cost": cost, "out": out})
    except Exception as e:
        emit({"type": "baseline", "status": "error", "error": str(e)})


# ----------------------------------------------------------------------------
# Web visualiser
# ----------------------------------------------------------------------------
STATS = {"tickets": 0, "template": 0, "llm": 0, "cost": 0.0, "ms": 0.0, "route_ms": 0.0,
         "base_n": 0, "base_cost": 0.0, "base_ms": 0.0}
STATS_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            return self._send(200, PAGE.replace("__LOGO__", LOGO_B64), "text/html; charset=utf-8")
        if p == "/api/samples":
            files = sorted(f.name for f in SAMPLES_DIR.glob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
            return self._send(200, files)
        if p == "/api/library":
            return self._send(200, load_library())
        if p == "/api/config":
            return self._send(200, {"models": MODELS, "jev": JEV_MODEL, "team_threshold": TEAM_THRESHOLD,
                                    "fix_threshold": FIX_THRESHOLD, "stats": STATS, "has_key": bool(API_KEY)})
        if p.startswith("/samples/"):
            f = (SAMPLES_DIR / Path(p).name)
            if f.exists():
                return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "image/png")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/approve":
            fix = add_to_library(body.get("category", "other"), body.get("title", "New issue"),
                                 body.get("when", ""), body.get("reply", ""))
            return self._send(200, fix)
        if self.path != "/api/run":
            return self._send(404, {"error": "not found"})
        # stream newline-delimited JSON events while the pipeline runs
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        lock = threading.Lock()

        def emit(ev):
            with lock:
                try:
                    self.wfile.write((json.dumps(ev) + "\n").encode())
                    self.wfile.flush()
                except Exception:
                    pass

        b64, mime = body["image"], body.get("mime", "image/png")
        pool = ThreadPoolExecutor(max_workers=1)
        base_future = pool.submit(run_baseline, b64, mime, lambda ev: (emit(ev), _base_stats(ev))) if body.get("baseline") else None
        try:
            res = run_pipeline(b64, mime, lambda ev: (emit(ev), _route_stats(ev)))
            with STATS_LOCK:
                STATS["tickets"] += 1
                STATS[res["path"]] += 1
                STATS["cost"] += res["cost"]
                STATS["ms"] += res["total_ms"]
        except Exception as e:
            emit({"type": "error", "error": str(e)})
        if base_future:
            base_future.result()
        emit({"type": "stats", "stats": STATS})


def _route_stats(ev):
    if ev.get("stage") == "route":
        with STATS_LOCK:
            STATS["route_ms"] += ev["at_ms"]


def _base_stats(ev):
    if ev.get("type") == "baseline" and ev.get("status") == "done":
        with STATS_LOCK:
            STATS["base_n"] += 1
            STATS["base_cost"] += ev["cost"]
            STATS["base_ms"] += ev["ms"]


# ----------------------------------------------------------------------------
# Terminal mode
# ----------------------------------------------------------------------------
def run_cli(path):
    data = Path(path).read_bytes()
    mime = mimetypes.guess_type(path)[0] or "image/png"

    def emit(ev):
        if ev["type"] == "stage" and ev["status"] in ("done", "fallback", "skipped"):
            s, extra = ev["stage"], ""
            if s == "read":
                extra = ("[vision fallback] " if ev.get("fallback") else "") + ev.get("text", ev.get("ocr_text", ""))[:160].replace("\n", " | ")
            elif s == "triage":
                extra = (f"category={ev['category']['choice']} ({ev['category']['confidence']:.2f})  "
                         f"team={ev['team']['choice']} ({ev['team']['confidence']:.2f})  severity={ev['severity_label']}")
            elif s == "route":
                extra = f"-> {ev['team']}  ({ev['why']})"
            elif s == "pick":
                extra = (f"{ev['fix']['choice']} ({ev['fix']['confidence']:.2f})" if ev["status"] == "done" else ev["why"])
            elif s == "answer":
                extra = f"{ev['path']}: {ev.get('fix_id', ev.get('model', ''))}  ({ev['why']})"
            cost = f" ${ev['cost']:.6f}" if "cost" in ev else ""
            ms = f" {ev['ms']:.0f}ms" if "ms" in ev else ""
            print(f"  {s.upper():7}{ms}{cost}  {extra}")
        elif ev["type"] == "result":
            print(f"\n  Reply ({ev['path']}):\n  " + ev['reply'].replace("\n", "\n  ") + "\n")
            print(f"  total {ev['total_ms']:.0f} ms, ${ev['cost']:.6f}, LLM calls: {ev['llm_calls']}, Jev calls: {ev['jev_calls']}")

    run_pipeline(base64.b64encode(data).decode(), mime, emit)


def main():
    if not API_KEY:
        print("Missing OPENROUTER_API_KEY. Create a .env file next to snaptriage.py containing:\n"
              "  OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)
    load_library()
    print("SnapTriage - resolving models on OpenRouter...")
    MODELS.update(resolve_models())
    for k, v in MODELS.items():
        print(f"  {k:9} {v}")
    print(f"  {'jev':9} {JEV_MODEL}")
    if len(sys.argv) > 1:
        return run_cli(sys.argv[1])
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"\nVisualiser running at {url}  (Ctrl+C to stop)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


LOGO_B64 = "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"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SnapTriage Visualiser</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--b:#5085EC;--r:#E2604F;--y:#EDB03D;--g:#5FB063;--ink:#16213d;--navy:#24365f;--navy2:#33497d;--mut:#5b6478;--soft:#eef1f8;--line:#dfe4ef;--bg:#f6f7fb}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,system-ui,Arial,sans-serif;color:var(--ink);background:var(--bg);font-size:14px;line-height:1.5}
code,.mono{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12.5px}
header{background:radial-gradient(120% 140% at 85% 0%,var(--navy2),var(--navy) 60%,#1b2a4d);color:#fff;padding:14px 24px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
header img{width:40px;height:40px}
header h1{font-size:20px;margin:0;font-weight:800;letter-spacing:-.01em}
header .sub{font-size:12px;opacity:.75}
header .models{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
.mchip{background:rgba(255,255,255,.1);border-radius:99px;padding:3px 10px;font-size:11.5px}
.mchip b{font-weight:600;opacity:.7;margin-right:4px}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;padding:14px 24px 0}
.stat{background:#fff;border-radius:16px;padding:10px 14px}
.stat .k{font-size:10.5px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--mut)}
.stat .v{font-size:20px;font-weight:800;margin-top:2px}
.stat .v small{font-size:12px;font-weight:500;color:var(--mut)}
main{display:grid;grid-template-columns:340px 1fr;gap:18px;padding:14px 24px 40px;align-items:start}
.panel{background:#fff;border-radius:22px;padding:16px}
.panel h2{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);margin:0 0 10px;font-weight:800}
.drop{border-radius:18px;background:var(--soft);padding:18px;text-align:center;cursor:pointer;transition:.15s;font-weight:600}
.drop:hover,.drop.over{background:#e3ecfd}
.drop small{display:block;font-weight:400;color:var(--mut);margin-top:2px}
.gallery{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
.thumb{border-radius:12px;overflow:hidden;cursor:pointer;background:var(--soft);position:relative;aspect-ratio:16/10;outline:2px solid transparent;transition:.15s}
.thumb img{width:100%;height:100%;object-fit:cover;object-position:top}
.thumb span{position:absolute;left:6px;bottom:6px;background:rgba(22,33,61,.85);color:#fff;font-size:10.5px;border-radius:99px;padding:1px 8px}
.thumb:hover{outline-color:var(--b)}
.toggle{display:flex;align-items:center;gap:10px;margin-top:14px;font-size:13px;cursor:pointer;user-select:none}
.sw{width:38px;height:22px;border-radius:99px;background:#d6dbe6;position:relative;flex:none;transition:.15s}
.sw:after{content:"";position:absolute;left:3px;top:3px;width:16px;height:16px;border-radius:50%;background:#fff;transition:.15s}
.toggle.on .sw{background:var(--g)}.toggle.on .sw:after{left:19px}
.preview{margin-top:14px;border-radius:14px;overflow:hidden;background:var(--soft);display:none}
.preview img{width:100%;display:block}
.flow{display:flex;flex-direction:column;gap:12px}
.empty{padding:60px 20px;text-align:center;color:var(--mut)}
.empty b{display:block;color:var(--ink);font-size:18px;margin-bottom:4px}
.card{background:#fff;border-radius:22px;padding:14px 16px;position:relative;opacity:.45;transition:.25s}
.card.active,.card.done,.card.skipped{opacity:1}
.card.skipped{opacity:.6}
.ch{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.tag{font-size:11px;font-weight:800;padding:2px 10px;border-radius:99px;border:2px solid var(--ink);box-shadow:2px 2px 0 var(--ink);background:var(--c)}
.ch .t{font-weight:800;font-size:15px}
.meta{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
.pill{font-size:11.5px;background:var(--soft);border-radius:99px;padding:2px 9px;color:var(--mut)}
.pill.money{background:#e2f2e3;color:#215e2a;font-weight:600}
.pill.time{background:#e3ecfd;color:#20449a;font-weight:600}
.status{font-size:11.5px;font-weight:700;border-radius:99px;padding:2px 9px;background:var(--soft);color:var(--mut)}
.status.running{background:#fdf1d6;color:#7a5200}
.status.done{background:#e2f2e3;color:#215e2a}
.status.skipped{background:#eceef3}
.spin{display:inline-block;width:9px;height:9px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;animation:s .7s linear infinite;margin-right:5px;vertical-align:-1px}
@keyframes s{to{transform:rotate(360deg)}}
.body{margin-top:10px}
.ocr{background:var(--navy);color:#e8edf8;border-radius:14px;padding:10px 12px;white-space:pre-wrap;max-height:150px;overflow:auto}
.note{font-size:12.5px;color:var(--mut);margin-top:6px}
.note b{color:var(--ink)}
.bars{display:grid;grid-template-columns:150px 1fr 46px;gap:5px 10px;align-items:center;font-size:12.5px}
.bars .lab{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bars .lab.win{font-weight:700}
.track{height:12px;border-radius:99px;background:var(--soft);position:relative;overflow:visible}
.fill{height:100%;border-radius:99px;background:var(--c,var(--b));width:0;transition:width .6s cubic-bezier(.2,.8,.2,1)}
.thr{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink);border-radius:2px}
.thr:after{content:attr(data-l);position:absolute;top:-16px;left:-11px;font-size:9.5px;font-weight:700;background:#fff;padding:0 2px}
.bars.hasthr{padding-top:14px}
.bars.wide{grid-template-columns:260px 1fr 46px}
.num{text-align:right;font-variant-numeric:tabular-nums;color:var(--mut)}
.qgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.qh{font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin-bottom:6px;display:flex;justify-content:space-between}
.route{background:var(--navy);color:#fff;border-radius:99px;padding:10px 18px;display:flex;align-items:center;gap:10px;opacity:.35;transition:.25s;flex-wrap:wrap}
.route.done{opacity:1;box-shadow:4px 4px 0 var(--ink)}
.route b{font-size:15px}.route .w{font-size:12px;opacity:.75;margin-left:auto}
.reply{background:var(--soft);border-radius:18px 18px 18px 4px;padding:12px 14px;white-space:pre-wrap}
.path{display:inline-flex;gap:6px;align-items:center;font-weight:700;font-size:12.5px;border-radius:99px;padding:3px 12px}
.path.template{background:#e2f2e3;color:#215e2a}.path.llm{background:#fce5e2;color:#8a2a1d}
.approve{margin-top:10px;background:#fdf1d6;border-radius:16px;padding:10px 12px;display:none}
.approve label{display:block;font-size:11px;font-weight:700;color:var(--mut);margin:6px 0 2px;text-transform:uppercase;letter-spacing:.06em}
.approve input,.approve textarea{width:100%;border:0;border-radius:10px;padding:7px 9px;font:inherit;font-size:13px;background:#fff}
.btn{border:2px solid var(--ink);box-shadow:2px 2px 0 var(--ink);background:var(--y);font:inherit;font-weight:800;font-size:12.5px;border-radius:99px;padding:5px 14px;cursor:pointer;margin-top:8px}
.btn:active{transform:translate(2px,2px);box-shadow:none}
.btn.ghost{background:#fff}
.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.summary div{background:var(--soft);border-radius:14px;padding:8px 12px}
.summary .k{font-size:10.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
.summary .v{font-size:17px;font-weight:800}
.vs{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
.vs>div{border-radius:16px;padding:10px 12px;background:var(--soft)}
.vs h4{margin:0 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.err{background:#fce5e2;color:#8a2a1d;border-radius:14px;padding:10px 12px;font-weight:600}
.lib{margin-top:16px}
.fix{display:flex;gap:8px;align-items:baseline;padding:5px 0;border-bottom:1px solid var(--soft);font-size:12.5px}
.fix code{font-size:11px;background:var(--soft);border-radius:6px;padding:0 5px;flex:none}
.fix.new code{background:var(--y)}
.legend{font-size:12px;color:var(--mut);margin-top:10px;line-height:1.6}
@media (max-width:900px){main{grid-template-columns:1fr;padding:12px}.stats{grid-template-columns:repeat(3,1fr);padding:12px 12px 0}.qgrid,.vs{grid-template-columns:1fr}.summary{grid-template-columns:1fr 1fr}.bars,.bars.wide{grid-template-columns:110px 1fr 40px}header .models{margin-left:0}}
</style></head><body>
<header>
  <img src="data:image/png;base64,__LOGO__" alt="">
  <div><h1>SnapTriage</h1><div class="sub">Google Developer Group · KAU · AI Engineering Workshop</div></div>
  <div class="models" id="models"></div>
</header>
<section class="stats">
  <div class="stat"><div class="k">Tickets</div><div class="v" id="s-t">0</div></div>
  <div class="stat"><div class="k">Answered w/o LLM</div><div class="v" id="s-a">–</div></div>
  <div class="stat"><div class="k">Avg cost / ticket</div><div class="v" id="s-c">–</div></div>
  <div class="stat"><div class="k">Avg time to route</div><div class="v" id="s-r">–</div></div>
  <div class="stat"><div class="k">Avg total time</div><div class="v" id="s-m">–</div></div>
  <div class="stat"><div class="k">Baseline avg</div><div class="v" id="s-b">–</div></div>
</section>
<main>
  <aside class="panel">
    <h2>Input</h2>
    <div class="drop" id="drop">Drop a screenshot here<small>or click to choose a file</small></div>
    <input type="file" id="file" accept="image/*" hidden>
    <div class="toggle on" id="base"><div class="sw"></div>Also run the baseline (one LLM call)</div>
    <div class="gallery" id="gallery"></div>
    <div class="preview" id="preview"><img id="pimg" alt=""></div>
    <div class="lib"><h2>Known-issues library <span id="libn" style="font-weight:600"></span></h2><div id="lib"></div></div>
  </aside>
  <section class="flow" id="flow">
    <div class="panel empty"><b>Pick a sample or drop a screenshot</b>Each stage lights up as it runs, with its model, time and cost.</div>
  </section>
</main>
<template id="tpl">
  <div class="card" id="c-read"><div class="ch"><span class="tag" style="--c:var(--g)">1 · READ</span><span class="t">Cheap OCR model</span><span class="status">waiting</span><div class="meta"></div></div><div class="body"></div></div>
  <div class="card" id="c-triage"><div class="ch"><span class="tag" style="--c:var(--b)">2 · TRIAGE</span><span class="t">Jev call 1</span><span class="status">waiting</span><div class="meta"></div></div><div class="body"></div></div>
  <div class="route" id="c-route"><span class="tag" style="--c:#fff;border-color:#fff;box-shadow:none;color:var(--ink)">ROUTED</span><b>waiting for triage…</b><span class="w"></span></div>
  <div class="card" id="c-pick"><div class="ch"><span class="tag" style="--c:var(--b)">3 · PICK A FIX</span><span class="t">Jev call 2</span><span class="status">waiting</span><div class="meta"></div></div><div class="body"></div></div>
  <div class="card" id="c-answer"><div class="ch"><span class="tag" style="--c:var(--r)">4 · ANSWER</span><span class="t">Template or top-tier LLM</span><span class="status">waiting</span><div class="meta"></div></div><div class="body"></div></div>
  <div class="card" id="c-sum"><div class="ch"><span class="tag" style="--c:var(--y)">RESULT</span><span class="t">This ticket</span></div><div class="body"></div></div>
  <div class="card" id="c-base" style="display:none"><div class="ch"><span class="tag" style="--c:#c9d3ea">BASELINE</span><span class="t">One multimodal LLM call does everything</span><span class="status">waiting</span><div class="meta"></div></div><div class="body"></div></div>
</template>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const money=c=>c==null?'–':(c===0?'$0':'$'+(c<0.001?c.toFixed(6):c.toFixed(4)));
const ms=m=>m==null?'–':(m<1000?Math.round(m)+' ms':(m/1000).toFixed(1)+' s');
let CFG={}, baseline=true, busy=false, newIds=new Set(), cur={};

async function init(){
  CFG=await (await fetch('/api/config')).json();
  const m=CFG.models; $('#models').innerHTML=[['OCR',m.ocr],['Jev',CFG.jev],['LLM',m.strong]].map(([k,v])=>`<span class="mchip"><b>${k}</b>${esc(v)}</span>`).join('');
  stats(CFG.stats);
  const files=await (await fetch('/api/samples')).json();
  $('#gallery').innerHTML=files.map(f=>`<div class="thumb" data-f="${esc(f)}"><img src="/samples/${encodeURIComponent(f)}" alt=""><span>${esc(f.replace(/^\d+_|\.\w+$/g,'').replace(/_/g,' '))}</span></div>`).join('');
  document.querySelectorAll('.thumb').forEach(t=>t.onclick=async()=>{const r=await fetch('/samples/'+encodeURIComponent(t.dataset.f));run(await r.blob());});
  loadLib();
}
async function loadLib(){
  const lib=await (await fetch('/api/library')).json();
  $('#libn').textContent='('+lib.length+')';
  $('#lib').innerHTML=lib.map(f=>`<div class="fix ${newIds.has(f.id)||f.approved_from_llm?'new':''}"><code>${esc(f.id)}</code><span>${esc(f.title)}</span></div>`).join('');
}
function stats(s){
  $('#s-t').textContent=s.tickets;
  if(s.tickets){
    $('#s-a').innerHTML=Math.round(100*s.template/s.tickets)+'%';
    $('#s-c').textContent=money(s.cost/s.tickets);
    $('#s-r').textContent=ms(s.route_ms/s.tickets);
    $('#s-m').textContent=ms(s.ms/s.tickets);
  }
  if(s.base_n) $('#s-b').innerHTML=money(s.base_cost/s.base_n)+' <small>'+ms(s.base_ms/s.base_n)+'</small>';
}
$('#base').onclick=()=>{baseline=!baseline;$('#base').classList.toggle('on',baseline);};
const drop=$('#drop');
drop.onclick=()=>$('#file').click();
$('#file').onchange=e=>e.target.files[0]&&run(e.target.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over');};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');const f=e.dataTransfer.files[0];f&&run(f);};
document.addEventListener('paste',e=>{const it=[...e.clipboardData.items].find(i=>i.type.startsWith('image/'));it&&run(it.getAsFile());});

function card(id){return $('#c-'+id);}
function setStatus(id,st,label){const c=card(id);c.className='card '+(st==='running'?'active':st);const s=c.querySelector('.status');s.className='status '+st;s.innerHTML=(st==='running'?'<span class="spin"></span>':'')+(label||st);}
function setMeta(id,ev){const pills=[];if(ev.model)pills.push(`<span class="pill">${esc(ev.model)}</span>`);if(ev.ms!=null)pills.push(`<span class="pill time">${ms(ev.ms)}</span>`);if(ev.cost!=null)pills.push(`<span class="pill money">${money(ev.cost)}</span>`);card(id).querySelector('.meta').innerHTML=pills.join('');}
function bars(probs,winner,thr,color,labels,keepOrder,wide){
  const rows=keepOrder?Object.entries(probs):Object.entries(probs).sort((a,b)=>b[1]-a[1]);
  return `<div class="bars ${thr!=null?'hasthr':''} ${wide?'wide':''}" style="--c:${color}">`+rows.map(([k,p])=>`<div class="lab ${k===winner?'win':''}" title="${esc(labels&&labels[k]||k)}">${esc(labels&&labels[k]?k+' · '+labels[k]:k)}</div><div class="track"><div class="fill" data-w="${(p*100).toFixed(1)}" style="${k===winner?'':'opacity:.45'}"></div>${thr!=null&&k===winner?`<div class="thr" style="left:${thr*100}%" data-l="${thr}"></div>`:''}</div><div class="num">${p.toFixed(2)}</div>`).join('')+'</div>';
}
function animate(){requestAnimationFrame(()=>document.querySelectorAll('.fill[data-w]').forEach(f=>{f.style.width=f.dataset.w+'%';f.removeAttribute('data-w');}));}

async function run(blob){
  if(busy) return; busy=true;
  const url=URL.createObjectURL(blob); $('#pimg').src=url; $('#preview').style.display='block';
  $('#flow').innerHTML=$('#tpl').innerHTML;
  if(baseline) card('base').style.display='block';
  const b64=await new Promise(r=>{const fr=new FileReader();fr.onload=()=>r(fr.result.split(',')[1]);fr.readAsDataURL(blob);});
  cur={};
  try{
    const res=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:b64,mime:blob.type||'image/png',baseline})});
    const rd=res.body.getReader(),dec=new TextDecoder();let buf='';
    for(;;){const {done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;while((i=buf.indexOf('\n'))>=0){const line=buf.slice(0,i);buf=buf.slice(i+1);if(line.trim())handle(JSON.parse(line));}}
  }catch(e){handle({type:'error',error:String(e)});}
  busy=false; loadLib();
}

function handle(ev){
  if(ev.type==='stats') return stats(ev.stats);
  if(ev.type==='error'){ $('#flow').insertAdjacentHTML('afterbegin',`<div class="err">Error: ${esc(ev.error)}</div>`); return; }
  if(ev.type==='baseline') return baselineEv(ev);
  if(ev.type==='result') return result(ev);
  if(ev.type!=='stage') return;
  const id=ev.stage, b=id==='route'?null:card(id).querySelector('.body');
  if(id==='read'){
    if(ev.status==='running'){setStatus('read','running','reading');setMeta('read',ev);}
    if(ev.status==='fallback'){setStatus('read','running','vision fallback');b.innerHTML=`<div class="note">OCR found only <b>${(ev.ocr_text||'').replace(/\s/g,'').length} characters</b>, so a stronger vision model describes the screen instead.</div>`;setMeta('read',{model:ev.model});}
    if(ev.status==='done'){setStatus('read','done',ev.fallback?'done · fallback used':'done');setMeta('read',{model:ev.fallback?ev.model+' → '+ev.fallback_model:ev.model,ms:ev.ms,cost:ev.cost});
      b.innerHTML=`<div class="ocr mono">${esc(ev.text||'(no text)')}</div>`+(ev.fallback?`<div class="note">Vision fallback used: the description above came from <b>${esc(ev.fallback_model)}</b>.</div>`:'');}
  }
  if(id==='triage'){
    if(ev.status==='running'){setStatus('triage','running','deciding');setMeta('triage',ev);}
    if(ev.status==='done'){setStatus('triage','done');setMeta('triage',{model:CFG.jev,ms:ev.ms,cost:ev.cost});
      const sv=ev.severity, sp={}; Object.entries(sv.probabilities||{}).forEach(([k,v])=>sp[(sv.legend&&sv.legend[k]||k).split(':')[0]]=v);
      b.innerHTML=`<div class="qgrid"><div><div class="qh"><span>Category</span><span>conf ${ev.category.confidence.toFixed(2)}</span></div>${bars(ev.category.probabilities,ev.category.choice,null,'var(--b)')}</div>
      <div><div class="qh"><span>Team</span><span>conf ${ev.team.confidence.toFixed(2)}</span></div>${bars(ev.team.probabilities,ev.team.choice,CFG.team_threshold,'var(--b)')}</div></div>
      <div style="margin-top:12px"><div class="qh"><span>Severity: ${esc(ev.severity_label)}</span><span>conf ${(sv.confidence??0).toFixed(2)}</span></div>${bars(sp,ev.severity_label,null,'var(--y)',null,true)}</div>`;animate();}
  }
  if(id==='route'){const r=card('route');r.classList.add('done');r.querySelector('b').textContent='→ '+ev.team;r.querySelector('.w').textContent=`${ms(ev.at_ms)} after upload · ${ev.why}`;}
  if(id==='pick'){
    if(ev.status==='running'){setStatus('pick','running','choosing');setMeta('pick',ev);b.innerHTML=`<div class="note">Choosing between <b>${ev.candidates.length}</b> known fixes + “none”.</div>`;}
    if(ev.status==='skipped'){setStatus('pick','skipped');b.innerHTML=`<div class="note">${esc(ev.why)}</div>`;}
    if(ev.status==='done'){setStatus('pick','done');setMeta('pick',{model:CFG.jev,ms:ev.ms,cost:ev.cost});const t=Object.assign({none:'no known fix'},ev.titles);
      b.innerHTML=`<div class="qh"><span>Chosen: ${esc(ev.fix.choice)}</span><span>conf ${ev.fix.confidence.toFixed(2)} · auto-send at ≥ ${CFG.fix_threshold}</span></div>${bars(ev.fix.probabilities,ev.fix.choice,CFG.fix_threshold,'var(--b)',t,false,true)}`;animate();}
  }
  if(id==='answer'){
    if(ev.status==='running'){setStatus('answer','running','top-tier LLM writing');setMeta('answer',ev);b.innerHTML=`<div class="note">${esc(ev.why)} → calling the top-tier LLM. The ticket is already with the team, so nobody waits on this.</div>`;}
    if(ev.status==='done'){
      setStatus('answer','done');
      if(ev.path==='template'){setMeta('answer',{cost:0});card('answer').querySelector('.meta').insertAdjacentHTML('afterbegin','<span class="pill">0 LLM calls</span>');
        b.innerHTML=`<span class="path template">Template · ${esc(ev.fix_id)} ${esc(ev.fix_title)}</span><div class="note">${esc(ev.why)}</div>`;}
      else{setMeta('answer',ev);cur.draft=ev.draft;
        b.innerHTML=`<span class="path llm">LLM answer · ${esc(ev.model)}</span><div class="note">${esc(ev.why)}</div>
        <div class="approve" id="appr" style="display:block"><b>Teach the library.</b> If IT approves this answer, the next ticket like this gets it instantly with zero LLM calls.
        <label>Title</label><input id="a-title" value="${esc(ev.draft.title)}"><label>Applies when</label><input id="a-when" value="${esc(ev.draft.when)}">
        <label>Reply</label><textarea id="a-reply" rows="4">${esc(ev.draft.reply)}</textarea><button class="btn" id="a-btn">Approve into library</button></div>`;
        $('#a-btn').onclick=approve;}
    }
  }
}
function result(ev){
  const b=card('sum').querySelector('.body');card('sum').classList.add('done');cur.res=ev;
  b.innerHTML=`<div class="reply">${esc(ev.reply)}</div><div class="summary" style="margin-top:12px">
   <div><div class="k">Routed to</div><div class="v" style="font-size:14px">${esc(ev.team)}</div></div>
   <div><div class="k">Total time</div><div class="v">${ms(ev.total_ms)}</div></div>
   <div><div class="k">Total cost</div><div class="v">${money(ev.cost)}</div></div>
   <div><div class="k">Calls</div><div class="v" style="font-size:14px">${ev.jev_calls} Jev · ${ev.llm_calls} LLM</div></div></div>`;
  compare();
}
function baselineEv(ev){
  const c=card('base'),b=c.querySelector('.body');
  if(ev.status==='running'){setStatus('base','running','running in parallel');setMeta('base',ev);}
  if(ev.status==='error'){setStatus('base','skipped','error');b.innerHTML=`<div class="err">${esc(ev.error)}</div>`;}
  if(ev.status==='done'){setStatus('base','done');setMeta('base',ev);cur.base=ev;const o=ev.out||{};
    b.innerHTML=`<div class="note">category <b>${esc(o.category)}</b> · team <b>${esc(o.team)}</b> · severity <b>${esc(o.severity)}</b> · self-reported confidence <b>${esc(o.confidence)}</b></div><div class="reply" style="margin-top:8px">${esc(o.reply)}</div><div id="cmp"></div>`;compare();}
}
function compare(){
  if(!cur.res||!cur.base) return; const r=cur.res,bs=cur.base, el=$('#cmp'); if(!el) return;
  el.innerHTML=`<div class="vs"><div><h4>SnapTriage</h4>${money(r.cost)} · ${ms(r.total_ms)} total<br>routed after the first Jev call<br>${r.llm_calls} LLM call${r.llm_calls===1?'':'s'}</div>
  <div><h4>Baseline</h4>${money(bs.cost)} · ${ms(bs.ms)}<br>nothing routed until the call ends<br>1 LLM call, confidence self-reported</div></div>`;
}
async function approve(){
  const body={category:cur.draft.category,title:$('#a-title').value,when:$('#a-when').value,reply:$('#a-reply').value};
  const fix=await (await fetch('/api/approve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  newIds.add(fix.id); $('#appr').innerHTML=`<b>Added as ${esc(fix.id)}.</b> Run the same screenshot again: it should now take the template path.`; loadLib();
}
init();
</script></body></html>
"""

if __name__ == "__main__":
    main()
