"""Thin HTTP client for the kie.ai API (https://docs.kie.ai).

Three things live here, and nothing else knows kie.ai exists:

* ``upload_image``   — push a local frame somewhere public, because every other kie.ai
                       endpoint takes image *URLs*, not file bytes.
* ``chat``           — OpenAI-compatible chat completions, used for the vision
                       art-director stage. Synchronous.
* ``run_image_task`` — the asynchronous createTask/recordInfo job flow that the image
                       models use. Submits, polls, returns the result URLs.
"""

import base64
import io
import json
import os
import time

import requests
from PIL import Image

from . import config


class KieError(RuntimeError):
    """Any failure talking to kie.ai, already phrased for a human to read."""


def api_key() -> str:
    key = os.environ.get(config.KIE_API_KEY_ENV, "").strip()
    if not key:
        raise KieError(
            f"{config.KIE_API_KEY_ENV} is not set. Create a key at https://kie.ai/api-key "
            f'and export it, e.g.  $env:{config.KIE_API_KEY_ENV} = "your-key"'
        )
    return key


def _headers() -> dict:
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}


def _explain_http_error(response: requests.Response, what: str) -> KieError:
    """kie.ai returns its real status in the JSON body, so read that before the HTTP code."""
    code, message = response.status_code, response.text[:400]
    try:
        body = response.json()
        code = body.get("code", code)
        message = body.get("msg") or body.get("message") or message
    except ValueError:
        pass

    if code == 401:
        return KieError(f"kie.ai rejected the API key while {what}. Check {config.KIE_API_KEY_ENV}.")
    if code == 402:
        return KieError(
            f"The kie.ai account has run out of credits, so it stopped while {what}. "
            "Top up at https://kie.ai/billing and try again."
        )
    if code == 429:
        return KieError(f"kie.ai is rate limiting us while {what}. Give it a few seconds and try again.")
    return KieError(f"kie.ai returned {code} while {what}: {message}")


def _explain_body_error(code: int, body: dict, what: str) -> KieError:
    """Same translation as _explain_http_error, for failures reported inside a 200 body."""
    message = body.get("msg") or body.get("message") or "no detail given"

    if code == 401:
        return KieError(f"kie.ai rejected the API key while {what}. Check {config.KIE_API_KEY_ENV}.")
    if code == 402:
        return KieError(
            f"The kie.ai account has run out of credits, so it stopped while {what}. "
            "Top up at https://kie.ai/billing and try again."
        )
    if code == 429:
        return KieError(f"kie.ai is rate limiting us while {what}. Give it a few seconds and try again.")
    if code >= 500:
        return KieError(f"kie.ai had a server error while {what}: {message}")
    return KieError(f"kie.ai refused the request while {what} (code {code}): {message}")


def _post(url: str, payload: dict, what: str) -> dict:
    """POST with retries on transport errors and 5xx — those are transient, 4xx are not."""
    last_error: Exception | None = None

    for attempt in range(1, config.KIE_MAX_RETRIES + 1):
        try:
            response = requests.post(
                url, headers=_headers(), json=payload,
                timeout=config.KIE_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as e:
            last_error = e
            if attempt < config.KIE_MAX_RETRIES:
                time.sleep(config.KIE_RETRY_BACKOFF_SECONDS * attempt)
            continue

        if response.status_code >= 500:
            last_error = _explain_http_error(response, what)
            if attempt < config.KIE_MAX_RETRIES:
                time.sleep(config.KIE_RETRY_BACKOFF_SECONDS * attempt)
            continue

        if response.status_code >= 400:
            raise _explain_http_error(response, what)

        try:
            body = response.json()
        except ValueError as e:
            raise KieError(f"kie.ai returned a non-JSON response while {what}.") from e

        # kie.ai signals failures in the body while still answering HTTP 200 — a transient
        # outage arrives as {"code": 500, "msg": "Server exception"} with a 200 status. The
        # HTTP-status checks above cannot see that, so it used to sail through as success
        # and fail confusingly further down ("returned no choices"), with no retry, on an
        # error that a retry usually fixes.
        #
        # The chat endpoint omits `code` entirely when it succeeds, so only an explicit
        # non-200 counts as a failure here.
        code = body.get("code") if isinstance(body, dict) else None
        if isinstance(code, int) and code != 200:
            error = _explain_body_error(code, body, what)
            if code >= 500:
                last_error = error
                if attempt < config.KIE_MAX_RETRIES:
                    time.sleep(config.KIE_RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise error

        return body

    # An already-translated KieError says everything useful; wrapping it again just nests
    # the same sentence twice. Only raw transport errors need the extra context.
    if isinstance(last_error, KieError):
        raise KieError(f"{last_error} Tried {config.KIE_MAX_RETRIES} times.")
    raise KieError(f"kie.ai is unreachable after {config.KIE_MAX_RETRIES} attempts while {what}: {last_error}")


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

def _encode_jpeg(path: str, max_width: int) -> str:
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def upload_image(path: str, max_width: int) -> str:
    """Upload a local image and return its public URL (kie.ai keeps these for 24 hours)."""
    payload = {
        "base64Data": f"data:image/jpeg;base64,{_encode_jpeg(path, max_width)}",
        "uploadPath": config.KIE_UPLOAD_PATH,
        "fileName": f"{os.path.splitext(os.path.basename(path))[0]}_{int(time.time() * 1000)}.jpg",
    }
    body = _post(config.KIE_UPLOAD_URL, payload, "uploading a frame")

    url = (body.get("data") or {}).get("downloadUrl")
    if not url:
        raise KieError(f"kie.ai accepted the upload but returned no downloadUrl: {json.dumps(body)[:300]}")
    return url


# ---------------------------------------------------------------------------
# Chat completions (vision)
# ---------------------------------------------------------------------------

def chat(messages: list[dict], model: str | None = None) -> str:
    """Call an OpenAI-compatible kie.ai chat model and return the assistant's text."""
    model = model or config.KIE_CHAT_MODEL
    url = config.KIE_CHAT_URL_TEMPLATE.format(model=model)
    payload = {
        "model": model,
        "messages": messages,
        "reasoning_effort": config.KIE_CHAT_REASONING_EFFORT,
    }
    body = _post(url, payload, f"asking {model} to art-direct the thumbnail")

    choices = body.get("choices") or []
    if not choices:
        raise KieError(f"{model} returned no choices: {json.dumps(body)[:300]}")

    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):
        # Some deployments return the content as typed parts rather than a bare string.
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not content or not content.strip():
        raise KieError(f"{model} returned an empty response.")
    return content.strip()


# ---------------------------------------------------------------------------
# Async image jobs
# ---------------------------------------------------------------------------

def _poll_task(task_id: str, on_progress=None) -> list[str]:
    deadline = time.monotonic() + config.KIE_POLL_TIMEOUT_SECONDS
    last_state = ""

    while time.monotonic() < deadline:
        try:
            response = requests.get(
                config.KIE_RECORD_INFO_URL,
                headers=_headers(),
                params={"taskId": task_id},
                timeout=config.KIE_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException:
            time.sleep(config.KIE_POLL_INTERVAL_SECONDS)
            continue

        if response.status_code >= 400:
            raise _explain_http_error(response, "checking the render job")

        data = (response.json() or {}).get("data") or {}
        state = data.get("state", "")

        if state != last_state and on_progress:
            on_progress(state, data.get("progress"))
            last_state = state

        if state == "success":
            try:
                result = json.loads(data.get("resultJson") or "{}")
            except ValueError as e:
                raise KieError("kie.ai reported success but its resultJson was unreadable.") from e
            urls = result.get("resultUrls") or []
            if not urls:
                raise KieError("kie.ai reported success but returned no image URLs.")
            return urls

        if state == "fail":
            raise KieError(
                f"kie.ai failed to render the thumbnail: "
                f"{data.get('failMsg') or 'no reason given'} (code {data.get('failCode')})"
            )

        time.sleep(config.KIE_POLL_INTERVAL_SECONDS)

    raise KieError(
        f"kie.ai took longer than {config.KIE_POLL_TIMEOUT_SECONDS} seconds to render this. "
        "It may just be busy, so it is worth another go."
    )


def run_image_task(model: str, task_input: dict, on_progress=None) -> list[str]:
    """Submit a createTask job and block until it produces image URLs."""
    body = _post(
        config.KIE_CREATE_TASK_URL,
        {"model": model, "input": task_input},
        f"submitting a {model} render",
    )

    task_id = (body.get("data") or {}).get("taskId")
    if not task_id:
        raise KieError(f"kie.ai accepted the job but returned no taskId: {json.dumps(body)[:300]}")

    return _poll_task(task_id, on_progress=on_progress)


def download(url: str, dest_path: str) -> str:
    """Fetch a rendered image off kie.ai's CDN onto disk.

    Retries hard, because by this point the render is already paid for and finished. A
    single transient DNS hiccup on the CDN host used to throw the whole thing away and
    silently fall back to the ungraded source frame, which is the most expensive possible
    way to fail: full cost, worst output.
    """
    last_error: Exception | None = None

    for attempt in range(1, config.KIE_DOWNLOAD_RETRIES + 1):
        try:
            response = requests.get(url, timeout=config.KIE_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as e:
            last_error = e
            if attempt < config.KIE_DOWNLOAD_RETRIES:
                time.sleep(config.KIE_RETRY_BACKOFF_SECONDS * attempt)
            continue

        if not response.content:
            last_error = KieError("the CDN returned an empty file")
            if attempt < config.KIE_DOWNLOAD_RETRIES:
                time.sleep(config.KIE_RETRY_BACKOFF_SECONDS * attempt)
            continue

        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(response.content)
        return dest_path

    raise KieError(
        f"The thumbnail rendered, but it could not be downloaded after "
        f"{config.KIE_DOWNLOAD_RETRIES} attempts: {last_error}"
    )
