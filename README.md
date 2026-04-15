# 🎙️ Voice-Controlled Local AI Agent

A fully local, privacy voice agent that accepts audio input, transcribes it,
classifies the user's intent with an LLM, executes local tools, and displays the
entire pipeline in a clean dark-themed Streamlit UI.

---

## 📁 Project Structure

```
voice-agent/
├── app.py                  
├── config.py               
├── requirements.txt
├── .env        
├── output/                 
│
├── core/
│   ├── stt.py              
│   ├── intent.py           
│   ├── pipeline.py         
│   └── memory.py           
│
└── tools/
    └── executor.py         
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/akshat-2600/voice-agent.git
cd voice-agent
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your preferred backends (see Configuration below)
```

### 3. Pull an Ollama model (if using local LLM)

```bash
# Install Ollama: https://ollama.com
ollama pull mistral          # ~4 GB, good balance of speed/quality
# or
ollama pull llama3           # larger, more capable
# or
ollama pull phi3:mini        # lightweight, fast on CPU
```

### 4. Run

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## ⚙️ Configuration

All settings live in `.env` (see `.env.example`):

| Variable         | Default          | Description                                   |
|------------------|------------------|-----------------------------------------------|
| `STT_BACKEND`    | `whisper`        | `whisper` (local) or `groq` (API)             |
| `WHISPER_MODEL`  | `base`           | `tiny` / `base` / `small` / `medium` / `large`|
| `GROQ_API_KEY`   | —                | Required only when using Groq for STT or LLM  |
| `LLM_BACKEND`    | `ollama`         | `ollama` (local) or `groq` (API)              |
| `OLLAMA_MODEL`   | `mistral`        | Any model you've pulled with `ollama pull`    |
| `OLLAMA_HOST`    | `localhost:11434`| Custom Ollama host if needed                  |
| `OUTPUT_DIR`     | `output`         | Safe sandbox for all file operations          |

---

## 🎯 Supported Intents

| Intent        | Example command                                          |
|---------------|----------------------------------------------------------|
| `write_code`  | *"Create a Python file with a binary search function."* |
| `create_file` | *"Create a new folder called reports."*                 |
| `summarize`   | *"Summarise this paragraph and save it to summary.txt."*|
| `chat`        | *"What is the difference between REST and GraphQL?"*    |

**Compound commands** are supported — e.g. *"Write a retry function and save it to utils.py"*
triggers both `write_code` and automatic file creation in one pass.

---

## 🔐 Safety

- **All file I/O is restricted to the `output/` directory.** The path is validated
  against path-traversal attacks on every operation.
- **Human-in-the-Loop toggle** (enabled by default in the sidebar): before any file
  operation executes you'll see a confirmation card with the detected intent, filename,
  and description. Approve or cancel.

---

## 💡 Hardware Notes / Why Groq?

### STT — Whisper

| Model  | VRAM / RAM | Speed (CPU) | Accuracy |
|--------|-----------|-------------|----------|
| tiny   | ~390 MB   | ~3–5 s      | fair     |
| base   | ~74 MB    | ~5–10 s     | good     |
| small  | ~244 MB   | ~20–30 s    | very good|
| medium | ~769 MB   | ~60–90 s    | great    |
| large  | ~1.5 GB   | very slow   | best     |

**`base` is the recommended default for CPU-only machines.**

If your machine cannot run Whisper efficiently (e.g. no GPU, < 8 GB RAM), set
`STT_BACKEND=groq` in `.env`. Groq's hosted Whisper-large-v3 processes a 30-second
clip in under 1 second — it's the best fallback for speed.

### LLM — Ollama vs Groq

| Backend | Latency (intent) | Privacy | Requires |
|---------|-----------------|---------|----------|
| Ollama  | 2–8 s (CPU)     | 🔒 100% local | `ollama` installed + model pulled |
| Groq    | < 1 s           | Cloud   | `GROQ_API_KEY`                    |

---

## ✨ Bonus Features Implemented

| Feature | Status |
|---------|--------|
| Compound commands |  Multiple intents per audio clip |
| Human-in-the-Loop |  Sidebar toggle + confirmation card |
| Graceful degradation |  `[STT Error]` / `[LLM Error]` handling at every stage |
| Session memory |  Scrollable history panel in sidebar |

---

## 📝 Example Flows

**Flow 1 — Code generation**

> *"Write a Python function that retries a failed HTTP request up to 3 times."*

1. Whisper → `"Write a Python function that retries…"`
2. LLM → intent: `write_code`, language: `python`, file: `retry.py`
3. LLM generates the retry function
4. Saves `output/retry.py`
5. UI shows transcription, intent badge, generated code, file path

**Flow 2 — Compound command**

> *"Summarise this paragraph and save it to notes.txt."*

1. STT → text
2. LLM → intents: `["summarize"]`, filename: `notes.txt`
3. LLM generates summary, saves `output/notes.txt`
4. UI shows all steps

---

## 🛠️ Extending the Agent

To add a new tool:
1. Add a new intent string to `VALID_INTENTS` in `core/intent.py`.
2. Add the intent to the LLM system prompt examples.
3. Implement a `tool_<name>(intent, text) -> ActionResult` function in `tools/executor.py`.
4. Register it in `_TOOL_MAP`.

---

## 📄 License

MIT
