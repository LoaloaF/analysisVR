"""NLChat agent loop over a free, OpenAI-compatible backend.

Turns a natural-language question into tool calls + a written summary, using any
OpenAI-compatible endpoint. Configured entirely by environment variables so no
key or provider is hard-coded:

    Groq (free tier, cloud):
        export LLM_BASE_URL=https://api.groq.com/openai/v1
        export LLM_MODEL=llama-3.3-70b-versatile
        export LLM_API_KEY=<your free groq key>     # or GROQ_API_KEY

    Google Gemini (free tier, cloud):
        export LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
        export LLM_MODEL=gemini-2.0-flash
        export LLM_API_KEY=<your free AI Studio key>  # or GEMINI_API_KEY

    Ollama (local, fully free + private):
        export LLM_BASE_URL=http://localhost:11434/v1
        export LLM_MODEL=qwen2.5:32b
        export LLM_API_KEY=ollama                      # any dummy value

Requires `pip install openai` (the SDK is just the transport; it talks to
whatever LLM_BASE_URL points at -- not necessarily OpenAI).
"""

import json
import os

from CustomLogger import CustomLogger as Logger
from .tools import TOOL_SCHEMAS, TOOL_IMPLS


def _load_dotenv():
    """Minimal .env loader (no dependency). Reads <repo>/.env into os.environ,
    without overriding variables already set in the shell (shell wins)."""
    here = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(here, "..", "..", ".env")  # repo root
    try:
        with open(env_path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_dotenv()

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_TURNS = 6
SYSTEM_PROMPT = (
    "You are a data-analysis assistant for a virtual-reality rodent-behavior lab. "
    "Answer questions by calling the available plotting tools. Paradigm 1100 is the "
    "1D track. The tool result contains computed statistics (after 'Stats --'); base "
    "your summary on those exact numbers and cite them (e.g. percentages, trends "
    "first-third to last-third). Write 2-3 sentences describing what the figure shows. "
    "Never invent numbers that are not in the tool result. If the user implies a subset "
    "(a specific cue or part of the session), pass it through; otherwise include everything."
)


def _api_key():
    """First non-empty of the common key env vars, so any provider's key works."""
    for name in ("LLM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(name)
        if val:
            return val
    return None


def _to_openai_tools(anthropic_schemas):
    """Convert the registry's input_schema to OpenAI function shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in anthropic_schemas
    ]


def run_agent(question, global_data=None):
    """question -> (figures, summary). figures is a list of (title, plotly_fig)."""
    L = Logger()
    try:
        from openai import OpenAI
    except ImportError:
        return [], "`openai` is not installed. Run `pip install openai` and restart."

    key = _api_key()
    if not key:
        return [], ("No LLM key set. Export LLM_API_KEY (or GROQ_API_KEY / "
                    "GEMINI_API_KEY) and restart. See dashsrc/nlchat/agent.py header.")

    client = OpenAI(base_url=BASE_URL, api_key=key)
    tools = _to_openai_tools(TOOL_SCHEMAS)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    figures = []       # (title, plotly_fig)
    seen = {}          # (tool, args) -> result text, so repeated identical calls don't re-run
    seen_captions = set()  # dedup identical figures even if the args were spelled differently

    for _ in range(MAX_TURNS):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools, tool_choice="auto",
            )
        except Exception as e:  # surface auth / rate-limit / connection errors to the UI
            return figures, f"LLM request failed ({type(e).__name__}): {e}"

        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return figures, (msg.content or "_(no summary returned)_").strip()

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            key = (name, json.dumps(args, sort_keys=True, default=str))
            impl = TOOL_IMPLS.get(name)
            if impl is None:
                text = f"Unknown tool '{name}'."
            elif key in seen:
                text = seen[key]  # identical args already ran -> reuse, no duplicate figure
            else:
                L.logger.info(f"NLChat tool call: {name}({args})")
                try:
                    fig, text = impl(global_data=global_data, **args)
                except Exception as e:
                    fig, text = None, f"{name} failed: {type(e).__name__}: {e}"
                seen[key] = text
                if fig is not None and text not in seen_captions:
                    figures.append((text, fig))
                    seen_captions.add(text)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})

    return figures, "_Stopped: reached the maximum number of agent turns._"
