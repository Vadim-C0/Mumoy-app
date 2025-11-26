from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import httpx
import os
import time
import logging
from dotenv import load_dotenv
import re
import html
from collections import defaultdict
from threading import Lock

# ---------------- Load Env ----------------
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://yourapp.onrender.com")
DOCS_USER = os.getenv("DOCS_USER")
DOCS_PASS = os.getenv("DOCS_PASS")

if not all([HF_API_KEY, YOUTUBE_API_KEY, DOCS_USER, DOCS_PASS]):
    raise RuntimeError("All environment variables must be set!")

# ---------------- App Setup ----------------
app = FastAPI(title="Mood Music App", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mood_app")

# ---------------- HuggingFace model ----------------
# High-accuracy model (GoEmotions)
HF_MODEL_URL = "https://api-inference.huggingface.co/models/SamLowe/roberta-base-go_emotions"
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
    "X-Wait-For-Model": "true",
}

# ---------------- In-Memory Storage ----------------
memory_limiter = defaultdict(list)
memory_cache = {}
memory_lock = Lock()

# ---------------- Rate Limits ----------------
RATE_LIMIT = 5
TIME_WINDOW = 60 * 1000  # ms

# ---------------- Mood Mapping (map many GO labels -> simple moods) ----------------
# The GoEmotions model has many fine-grained labels; map them to your simple genres.
go_to_mood = {
    # joy/positive
    "joy": "joy",
    "amusement": "joy",
    "relief": "joy",
    "approval": "joy",
    "excitement": "joy",
    # love/affection
    "admiration": "love",
    "gratitude": "love",
    "love": "love",
    # anger/hostile
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    # sadness
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    # fear
    "fear": "fear",
    "nervousness": "fear",
    "embarrassment": "fear",
    # disgust
    "disgust": "disgust",
    "repulsion": "disgust",
    # surprise
    "surprise": "surprise",
    "confusion": "surprise",
}

# Fallback mapping: if label not present, we'll keep raw label as a fallback mood.

# ---------------- Mood -> genre ----------------
mood_to_genre = {
    "anger": "metal music",
    "joy": "pop music",
    "sadness": "lofi music",
    "fear": "acoustic music",
    "disgust": "punk music",
    "surprise": "electronic music",
    "love": "romantic music",
}

# ---------------- Middleware ----------------
class ForwardedHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        xff = request.headers.get("x-forwarded-for")
        if xff:
            request.scope["client"] = (xff.split(",")[0].strip(), 0)

        xfp = request.headers.get("x-forwarded-proto")
        if xfp:
            request.scope["scheme"] = xfp

        return await call_next(request)

app.add_middleware(ForwardedHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # use forwarded proto when behind a proxy
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        if proto != "https":
            url = request.url.replace(scheme="https")
            return RedirectResponse(url)
        return await call_next(request)

app.add_middleware(HTTPSRedirectMiddleware)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.update({
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-XSS-Protection": "1; mode=block",
            "X-Download-Options": "noopen",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' https://www.youtube.com https://www.youtube-nocookie.com; "
                "style-src 'self'; "  # avoid 'unsafe-inline'
                "frame-src https://www.youtube.com https://www.youtube-nocookie.com; "
                "img-src 'self' data: https://i.ytimg.com; "
                "connect-src 'self' https://api-inference.huggingface.co https://www.googleapis.com;"
            )
        })
        return response

app.add_middleware(SecurityHeadersMiddleware)

class LimitBodySizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        max_size = 1024 * 10  # 10 KB
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            return JSONResponse({"detail": "Request too large"}, status_code=413)
        return await call_next(request)

app.add_middleware(LimitBodySizeMiddleware)

# ---------------- Utilities ----------------

def sanitize_input(text: str) -> str:
    text = text.strip()[:250]
    # allow common punctuation and letters/numbers; deny control chars
    if not re.match(r"^[\w\s.,!?'\-\u00C0-\u024F]*$", text, re.UNICODE):
        raise ValueError("Invalid characters in input.")
    return html.escape(text)


def get_client_key(request: Request) -> str:
    ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    ua = request.headers.get("user-agent", "unknown")
    return f"{ip}:{ua}"


async def is_rate_limited(key: str) -> bool:
    now = int(time.time() * 1000)
    with memory_lock:
        timestamps = memory_limiter[key]
        memory_limiter[key] = [t for t in timestamps if t > now - TIME_WINDOW]
        if len(memory_limiter[key]) >= RATE_LIMIT:
            return True
        memory_limiter[key].append(now)
    return False


def _validate_video_id(video_id: str):
    if not video_id:
        return None
    if re.match(r"^[A-Za-z0-9_-]{11}$", video_id):
        return video_id
    return None


async def get_emotion_genre_video(mood: str):
    key = mood.lower()

    # Cache
    if key in memory_cache:
        data = memory_cache[key]
        return data["emotion"], data["genre"], data["video_id"]

    # HuggingFace API
    emotion = "unknown"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(HF_MODEL_URL, headers=HEADERS, json={"inputs": mood})
            resp.raise_for_status()
            result = resp.json()

            # GoEmotions inference typically returns a list of dicts: [{"label": ..., "score": ...}, ...]
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                # pick the label with the highest score
                best = max(result, key=lambda x: x.get("score", 0))
                raw_label = best.get("label", "unknown").lower()
                # map fine-grained GO label to our simple mood
                emotion = go_to_mood.get(raw_label, raw_label)

    except httpx.HTTPStatusError as e:
        # if model removed or disabled, surface the response for logs
        logger.error(f"HuggingFace API status error: {e.response.status_code} {e.response.text}")
    except Exception as e:
        logger.error(f"HuggingFace API error: {e}")

    # map to genre
    genre = mood_to_genre.get(emotion, "chill music")

    # YouTube search
    video_id = None
    try:
        params = {
            "part": "snippet",
            "q": genre,
            "type": "video",
            "videoCategoryId": "10",
            "maxResults": 1,
            "key": YOUTUBE_API_KEY,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            yt_res = await client.get("https://www.googleapis.com/youtube/v3/search", params=params)
            yt_res.raise_for_status()
            items = yt_res.json().get("items", [])
            if items and "id" in items[0] and "videoId" in items[0]["id"]:
                video_id = _validate_video_id(items[0]["id"]["videoId"])

    except Exception as e:
        logger.warning(f"YouTube API error: {e}")

    # sanitize outputs before caching
    safe_emotion = html.escape(emotion)
    safe_genre = html.escape(genre)

    memory_cache[key] = {"emotion": safe_emotion, "genre": safe_genre, "video_id": video_id}
    return safe_emotion, safe_genre, video_id


# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "video_id": None})


@app.post("/", response_class=HTMLResponse)
async def submit_mood(request: Request, mood: str = Form(...)):
    client_key = get_client_key(request)

    if await is_rate_limited(client_key):
        raise HTTPException(status_code=429, detail="Too many requests. Wait a minute.")

    try:
        mood = sanitize_input(mood)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid input.")

    if not mood:
        raise HTTPException(status_code=400, detail="Input cannot be empty.")

    emotion, genre, video_id = await get_emotion_genre_video(mood)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "video_id": video_id, "emotion": emotion, "genre": genre},
    )


@app.get("/health", response_class=JSONResponse)
async def health_check():
    return {"status": "ok", "memory_cache": "ok"}


# ---------------- Protected Docs ----------------
security = HTTPBasic()


def check_docs(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = credentials.username == DOCS_USER
    correct_pass = credentials.password == DOCS_PASS
    if not (correct_user and correct_pass):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/docs", dependencies=[Depends(check_docs)])
async def get_docs():
    from fastapi.openapi.docs import get_swagger_ui_html

    return get_swagger_ui_html(openapi_url=app.openapi_url, title="Docs")


@app.get("/redoc", dependencies=[Depends(check_docs)])
async def get_redoc():
    from fastapi.openapi.docs import get_redoc_html

    return get_redoc_html(openapi_url=app.openapi_url, title="ReDoc")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
