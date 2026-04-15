"""
app.py — Streamlit UI for the Voice-Controlled Local AI Agent.

Runs with:  streamlit run app.py
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import streamlit as st

import config
from core import memory
from core.pipeline import run as run_pipeline


# Page config

st.set_page_config(
    page_title="Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Loading CSS stylesheet

css_file = Path(__file__).parent / "assets" / "css" / "styles.css"
if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Session state init

for key, default in {
    "_agent_history": [],
    "pending_intent": None,
    "pending_transcription": None,
    "pending_audio": None,
    "pending_ext": None,
    "last_results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Header

st.markdown(
    """
<div class="agent-header">
  <div>
    <p class="agent-title">🎙️ Voice AI Agent</p>
    <p class="agent-subtitle">LOCAL · VOICE-CONTROLLED · TOOL-AUGMENTED</p>
  </div>
</div>
<hr>
""",
    unsafe_allow_html=True,
)


# Sidebar — config + history


with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown(f"**STT Backend:** `{config.STT_BACKEND}`")
    if config.STT_BACKEND == "whisper":
        st.markdown(f"**Whisper model:** `{config.WHISPER_MODEL}`")
    st.markdown(f"**LLM Backend:** `{config.LLM_BACKEND}`")
    if config.LLM_BACKEND == "gemini":
        st.markdown(f"**Gemini model:** `{config.GEMINI_MODEL}`")
    elif config.LLM_BACKEND == "openrouter":
        base = config.OPENAI_BASE_URL or "https://openrouter.ai/api/v1"
        model = config.OPENROUTER_MODEL
        st.markdown(f"**OpenRouter model:** `{model}`  \n**Endpoint:** `{base}`")
    elif config.LLM_BACKEND == "groq":
        st.markdown(f"**Groq model:** `{config.GROQ_LLM_MODEL}`")
    elif config.LLM_BACKEND == "ollama":
        st.markdown(f"**Ollama model:** `{config.OLLAMA_MODEL}`")
    st.markdown(f"**Output dir:** `{config.OUTPUT_DIR}`")

    st.markdown("---")
    human_in_loop = st.toggle("🔒 Human-in-the-Loop", value=True,
                               help="Require confirmation before executing file operations")
    st.markdown("---")

    st.markdown("### 📜 Session History")
    hist = memory.get_history(st.session_state)
    if hist:
        st.markdown(
            '<div class="history-row"><span class="ts">Time</span>'
            "<span>Transcription</span><span>Intents</span><span>Status</span></div>",
            unsafe_allow_html=True,
        )
        for h in reversed(hist[-20:]):
            status = '<span class="ok">✓ OK</span>' if h.success else '<span class="fail">✗ ERR</span>'
            badges = " ".join(
                f'<span class="intent-badge badge-{i}">{i}</span>' for i in h.intents
            )
            trunc = (h.transcription[:60] + "…") if len(h.transcription) > 60 else h.transcription
            st.markdown(
                f'<div class="history-row">'
                f'<span class="ts">{h.timestamp}</span>'
                f"<span>{trunc}</span>"
                f"<span>{badges}</span>"
                f"<span>{status}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if st.button("🗑️ Clear history"):
            memory.clear(st.session_state)
            st.rerun()
    else:
        st.markdown('<span style="color:#475569;font-size:0.8rem;">No history yet.</span>',
                    unsafe_allow_html=True)


# Main

col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("#### 🎤 Audio Input")
    input_method = st.radio(
        "Input method", ["🎙️ Microphone", "📁 Upload file"], horizontal=True, label_visibility="collapsed"
    )

    audio_bytes: bytes | None = None
    file_ext = ".wav"

    if input_method == "🎙️ Microphone":
        try:
            from streamlit_mic_recorder import mic_recorder  # type: ignore
            recording = mic_recorder(
                start_prompt="⏺  Start recording",
                stop_prompt="⏹  Stop",
                just_once=False,
                key="mic",
            )
            if recording:
                audio_bytes = recording["bytes"]
                file_ext = ".wav"
                st.audio(audio_bytes, format="audio/wav")
        except ImportError:
            st.warning(
                "`streamlit-mic-recorder` not installed. "
                "Run `pip install streamlit-mic-recorder` or use file upload.",
                icon="⚠️",
            )
    else:
        uploaded = st.file_uploader(
            "Upload audio file", type=["wav", "mp3", "ogg", "m4a"], label_visibility="collapsed"
        )
        if uploaded:
            audio_bytes = uploaded.read()
            file_ext = Path(uploaded.name).suffix.lower() or ".wav"
            st.audio(audio_bytes, format=f"audio/{file_ext.lstrip('.')}")

    run_btn = st.button("🚀 Run Pipeline", disabled=audio_bytes is None, use_container_width=True)


# Pipeline execution

with col_results:
    st.markdown("#### 🔄 Pipeline Output")
    
    if st.session_state.pending_intent is not None and human_in_loop:
        pending = st.session_state.pending_intent
        
        # Display confirmation card
        st.markdown(
            """
            <div class="pipeline-card warning" style="margin-top: -10px;">
                <div class="card-label">⚠️ AWAITING YOUR CONFIRMATION</div>
                <div class="card-content" style="margin-top: 12px;">
                    <p style="margin: 0 0 10px 0; font-size: 0.95rem; color: #fbbf24;">
                        <strong>Review the detected intent below and confirm before execution:</strong>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Intent details
        st.markdown(
            f"<div style='margin-top: 16px;'><strong>🎯 Detected Intents:</strong></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            " ".join(f'<span class="intent-badge badge-{i}">{i}</span>' for i in pending.intents),
            unsafe_allow_html=True,
        )
        
        # Additional details
        if pending.filename:
            st.markdown(f"<div style='margin-top: 12px;'><strong>📁 File:</strong> <code>{pending.filename}</code></div>", unsafe_allow_html=True)
        if pending.language:
            st.markdown(f"<div style='margin-top: 8px;'><strong>💻 Language:</strong> <code>{pending.language}</code></div>", unsafe_allow_html=True)
        if pending.description:
            st.markdown(f"<div style='margin-top: 8px;'><strong>📝 Description:</strong> {pending.description}</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        # Confirmation buttons with prominent styling
        col1, col2, col3 = st.columns([2, 1, 1], gap="medium")
        with col1:
            st.markdown("<div></div>", unsafe_allow_html=True)
        
        with col2:
            if st.button("✅ Confirm & Execute", use_container_width=True, key="confirm_intent"):
                # Execute with confirmed=True
                try:
                    with st.spinner("⚙️ Executing tools..."):
                        from tools.executor import execute as exec_tools
                        from core import memory as mem
                        
                        results = exec_tools(pending, st.session_state.pending_transcription)
                        entry = mem.HistoryEntry.build(
                            st.session_state.pending_transcription, pending, results
                        )
                        mem.append(st.session_state, entry)
                        st.session_state.last_results = {
                            "transcription": st.session_state.pending_transcription,
                            "intent": pending,
                            "results": results,
                        }
                        st.session_state.pending_intent = None
                        st.session_state.pending_transcription = None
                        st.success("✅ Execution completed!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error executing: {str(e)}")
                    st.session_state.pending_intent = None
        
        with col3:
            if st.button("❌ Cancel", use_container_width=True, key="cancel_intent"):
                st.session_state.pending_intent = None
                st.session_state.pending_transcription = None
                st.rerun()
        
        st.markdown("<hr style='margin-top: 24px;'>", unsafe_allow_html=True)
    
    elif run_btn and audio_bytes:
        st.session_state.last_results = None
        state = {}
        status_placeholder = st.empty()
        results_area = st.container()

        pipeline = run_pipeline(
            audio_bytes=audio_bytes,
            file_ext=file_ext,
            session_state=st.session_state,
            confirmed=not human_in_loop,
        )

        for event in pipeline:
            stage = event["stage"]
            data = event["data"]

            if stage == "transcribing":
                status_placeholder.info("📡 Transcribing audio…")

            elif stage == "transcribed":
                state["transcription"] = data
                status_placeholder.info("🧠 Classifying intent…")

            elif stage == "classifying":
                pass

            elif stage == "classified":
                state["intent"] = data
                if human_in_loop:
                    # Store for confirmation UI
                    st.session_state.pending_intent = data
                    st.session_state.pending_transcription = state.get("transcription", "")
                    status_placeholder.success("✅ Intent classified — awaiting your confirmation.")
                    st.rerun()  # Force immediate rerun to show confirmation UI
                    break

            elif stage == "executing":
                status_placeholder.info("⚙️ Executing tools…")

            elif stage == "executed":
                state["results"] = data
                status_placeholder.success("✅ Done!")

            elif stage == "done":
                state.update(data)
                status_placeholder.success("✅ Pipeline complete!")
                st.session_state.last_results = state

            elif stage == "awaiting_confirmation":
                status_placeholder.warning("⚠️ Confirm before execution (see above).")
                break

            elif stage == "error":
                status_placeholder.error(f"❌ {data}")
                break
    
    else:
        if st.session_state.pending_intent is None:
            st.markdown(
                '<div style="color:#475569;font-size:0.85rem;padding:40px 0;text-align:center;">'
                "Record or upload audio, then click <strong>Run Pipeline</strong>."
                "</div>",
                unsafe_allow_html=True,
            )

    res = st.session_state.last_results
    if res:
        # Transcription 
        st.markdown(
            f'<div class="pipeline-card neutral">'
            f'<div class="card-label">01 · Transcription</div>'
            f'<div class="card-content">{res.get("transcription", "—")}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Intent
        parsed = res.get("intent")
        if parsed:
            badges = "".join(
                f'<span class="intent-badge badge-{i}">{i}</span>' for i in parsed.intents
            )
            compound_tag = " <em style='font-size:0.65rem;color:#fbbf24'>COMPOUND</em>" if parsed.is_compound else ""
            extra = ""
            if parsed.filename:
                extra += f"<br>📄 <strong>File:</strong> <code>{parsed.filename}</code>"
            if parsed.language:
                extra += f"&nbsp;&nbsp;💻 <strong>Language:</strong> <code>{parsed.language}</code>"
            if parsed.description:
                extra += f"<br>📝 {parsed.description}"
            st.markdown(
                f'<div class="pipeline-card neutral">'
                f'<div class="card-label">02 · Detected Intent{compound_tag}</div>'
                f'<div class="card-content">{badges}{extra}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        # Actions & Outputs
        results_list = res.get("results", [])
        for idx, r in enumerate(results_list, 1):
            cls = "success" if r.success else "error"
            icon = "✅" if r.success else "❌"
            file_tag = (
                f"<br>📁 <code>{r.file_path}</code>" if r.file_path else ""
            )
            # if action is write_code
            if r.success:
                output_html = (
                    f"<pre>{r.output}</pre>"
                    if "code" in r.action_taken.lower() or "\n" in r.output
                    else f'<div class="output-box">{r.output}</div>'
                )
            else:
                output_html = f'<div class="error-box">Error: {r.error or "Unknown error"}</div>'
            st.markdown(
                f'<div class="pipeline-card {cls}">'
                f'<div class="card-label">0{2+idx} · Action {idx} — {icon} {r.action_taken}</div>'
                f'<div class="card-content">{file_tag}{output_html}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

# Footer
st.markdown(
    "<hr><p style='text-align:center;color:#1e293b;font-size:0.7rem;'>"
    "Voice AI Agent · all file operations restricted to <code>output/</code>"
    "</p>",
    unsafe_allow_html=True,
)