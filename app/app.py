import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import streamlit as st
import streamlit.components.v1 as components
from fastapi import HTTPException

APP_DIR = Path(__file__).resolve().parent
FRONTEND_BUILD = APP_DIR / "frontend" / "dist"
sys.path.insert(0, str(APP_DIR))
from backend import main as api  # noqa: E402


def dispatch(request):
    parts = [unquote(part) for part in urlsplit(request.get("url", "")).path.split("/") if part]
    method = request.get("method", "GET").upper()
    try:
        body = json.loads(request.get("body") or "{}")
        if method == "GET" and parts == ["api", "algorithms"]:
            payload = api.algorithms()
        elif method == "GET" and len(parts) == 4 and parts[:2] == ["api", "notebook"]:
            payload = api.notebook(parts[2], parts[3])
        elif method == "GET" and len(parts) == 3 and parts[:2] == ["api", "info"]:
            payload = api.info(parts[2])
        elif method == "GET" and len(parts) == 4 and parts[:2] == ["api", "playground"]:
            payload = api.playground_schema(parts[2], parts[3])
        elif method == "POST" and len(parts) == 4 and parts[:2] == ["api", "predict"]:
            payload = api.predict(parts[2], parts[3], body)
        else:
            return {"status":404, "payload":{"detail":"API route not found"}}
        return {"status":200, "payload":payload}
    except HTTPException as exc:
        return {"status":exc.status_code, "payload":{"detail":exc.detail}}
    except Exception as exc:
        return {"status":500, "payload":{"detail":f"Request failed: {exc}"}}


if not FRONTEND_BUILD.joinpath("index.html").is_file():
    raise RuntimeError("React component assets are missing. Run `npm ci` and `npm run build` in app/frontend.")

st.set_page_config(page_title="ML Capstone — Field Decks", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer {display:none !important;}
[data-testid="stMainBlockContainer"], [data-testid="stAppViewBlockContainer"] {max-width:none !important; padding:0 !important;}
</style>
""", unsafe_allow_html=True)

field_decks = components.declare_component("field_decks", path=str(FRONTEND_BUILD))
reply = st.session_state.get("_field_decks_api_reply")
message = field_decks(apiReply=reply, key="field-decks", height=900)

if isinstance(message, dict) and message.get("kind") == "api-request":
    request_id = message.get("requestId")
    if request_id and request_id != st.session_state.get("_field_decks_last_request"):
        st.session_state["_field_decks_last_request"] = request_id
        result = dispatch(message)
        st.session_state["_field_decks_api_reply"] = {"requestId":request_id, **result}
        st.rerun()
