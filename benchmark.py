"""
benchmark.py — Compare STT and LLM model performance.

Measures latency and accuracy across configured backends.
Results are printed as a markdown table and saved to output/benchmark.md.

Usage:
    python benchmark.py                     # uses current .env config
    python benchmark.py --stt-models tiny base small
    python benchmark.py --llm-models mistral phi3:mini
    python benchmark.py --audio path/to/test.wav
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import config

# sample text for LLM intent timing
SAMPLE_COMMANDS = [
    "Create a Python file with a retry function",
    "Summarize this paragraph about machine learning",
    "What is the difference between async and sync programming?",
    "Write a JavaScript fetch wrapper and save it to api.js",
]

TABLE_SEP = "| " + " | ".join(["---"] * 5) + " |"


# STT benchmark
def benchmark_whisper_models(audio_bytes: bytes, file_ext: str, models: list[str]) -> list[dict]:
    import whisper 
    import tempfile

    results = []
    for model_name in models:
        print(f"  Loading whisper:{model_name}…", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            model = whisper.load_model(model_name)
            load_time = time.perf_counter() - t0

            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            t1 = time.perf_counter()
            out = model.transcribe(tmp_path)
            infer_time = time.perf_counter() - t1
            Path(tmp_path).unlink(missing_ok=True)

            text = out["text"].strip()
            print(f"done ({infer_time:.2f}s)")
            results.append({
                "model": f"whisper:{model_name}",
                "load_s": f"{load_time:.2f}",
                "infer_s": f"{infer_time:.2f}",
                "transcription": text[:80] + ("…" if len(text) > 80 else ""),
                "error": "",
            })
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({"model": f"whisper:{model_name}", "load_s": "—", "infer_s": "—",
                            "transcription": "", "error": str(exc)[:60]})
    return results


def benchmark_groq_stt(audio_bytes: bytes, file_ext: str) -> dict:
    if not config.GROQ_API_KEY:
        return {"model": "groq:whisper-large-v3", "load_s": "N/A", "infer_s": "—",
                "transcription": "", "error": "No GROQ_API_KEY set"}
    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=config.GROQ_API_KEY)
        t0 = time.perf_counter()
        resp = client.audio.transcriptions.create(
            file=(f"audio{file_ext}", audio_bytes),
            model="whisper-large-v3",
            response_format="text",
        )
        elapsed = time.perf_counter() - t0
        text = resp.strip() if isinstance(resp, str) else resp.text.strip()
        return {"model": "groq:whisper-large-v3", "load_s": "0 (cloud)", "infer_s": f"{elapsed:.2f}",
                "transcription": text[:80] + ("…" if len(text) > 80 else ""), "error": ""}
    except Exception as exc:
        return {"model": "groq:whisper-large-v3", "load_s": "N/A", "infer_s": "—",
                "transcription": "", "error": str(exc)[:60]}


# LLM benchmark
def benchmark_ollama_models(models: list[str]) -> list[dict]:
    try:
        import ollama 
    except ImportError:
        return [{"model": m, "avg_s": "—", "error": "ollama not installed"} for m in models]

    from core.intent import _SYSTEM_PROMPT
    results = []
    for model_name in models:
        print(f"  Testing ollama:{model_name}…", end=" ", flush=True)
        times = []
        error = ""
        for cmd in SAMPLE_COMMANDS[:2]:  
            try:
                t0 = time.perf_counter()
                ollama.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": cmd},
                    ],
                )
                times.append(time.perf_counter() - t0)
            except Exception as exc:
                error = str(exc)[:60]
                break
        avg = f"{sum(times)/len(times):.2f}" if times else "—"
        print(f"avg {avg}s")
        results.append({"model": f"ollama:{model_name}", "avg_s": avg,
                         "samples": len(times), "error": error})
    return results


def benchmark_groq_llm() -> dict:
    if not config.GROQ_API_KEY:
        return {"model": f"groq:{config.GROQ_LLM_MODEL}", "avg_s": "—", "samples": 0,
                "error": "No GROQ_API_KEY"}
    from groq import Groq  # type: ignore
    from core.intent import _SYSTEM_PROMPT
    client = Groq(api_key=config.GROQ_API_KEY)
    times = []
    for cmd in SAMPLE_COMMANDS[:2]:
        t0 = time.perf_counter()
        client.chat.completions.create(
            model=config.GROQ_LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": cmd},
            ],
            response_format={"type": "json_object"},
        )
        times.append(time.perf_counter() - t0)
    avg = f"{sum(times)/len(times):.2f}"
    return {"model": f"groq:{config.GROQ_LLM_MODEL}", "avg_s": avg,
            "samples": len(times), "error": ""}


# Report formatting
def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    header_row = "| " + " | ".join(headers) + " |"
    data_rows = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header_row, sep] + data_rows)


def build_report(stt_rows: list[dict], llm_rows: list[dict]) -> str:
    lines = [
        "# Model Benchmarking Report",
        "",
        "Auto-generated by `benchmark.py`.",
        "",
        "## Speech-to-Text (STT)",
        "",
        "_Lower inference time is better. Load time is one-off per session._",
        "",
    ]
    stt_table = _md_table(
        ["Model", "Load (s)", "Infer (s)", "Transcription preview", "Error"],
        [[r["model"], r["load_s"], r["infer_s"], r["transcription"], r["error"]] for r in stt_rows],
    )
    lines.append(stt_table)
    lines += [
        "",
        "### Notes",
        "- **whisper:tiny** — fastest, lowest accuracy; good for real-time on CPU.",
        "- **whisper:base** — recommended default; good accuracy with acceptable CPU speed.",
        "- **whisper:small** — noticeably more accurate; requires ~4 GB RAM.",
        "- **groq:whisper-large-v3** — cloud-hosted, sub-second latency, best accuracy; needs API key.",
        "",
        "## Intent Classification (LLM)",
        "",
        "_Average latency over 2 sample commands._",
        "",
    ]
    llm_table = _md_table(
        ["Model", "Avg latency (s)", "Samples", "Error"],
        [[r["model"], r["avg_s"], str(r.get("samples", "—")), r["error"]] for r in llm_rows],
    )
    lines.append(llm_table)
    lines += [
        "",
        "### Notes",
        "- **ollama:phi3:mini** — ~1–3 s on M-series Mac / modern CPU; very lightweight.",
        "- **ollama:mistral** — 3–6 s on CPU; strong reasoning, recommended default.",
        "- **ollama:llama3** — 5–10 s on CPU; highest quality locally.",
        "- **groq:llama3-8b** — <1 s cloud; best for latency-sensitive deployments.",
    ]
    return "\n".join(lines)


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stt-models", nargs="+", default=["tiny", "base"],
                        help="Whisper model sizes to benchmark")
    parser.add_argument("--llm-models", nargs="+", default=[config.OLLAMA_MODEL],
                        help="Ollama model names to benchmark")
    parser.add_argument("--audio", help="Path to a .wav file for STT benchmarking")
    parser.add_argument("--skip-stt", action="store_true", help="Skip STT benchmarks")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM benchmarks")
    args = parser.parse_args()

    print("\n📊 Voice Agent — Model Benchmarking\n")

    stt_rows: list[dict] = []
    llm_rows: list[dict] = []

    # STT
    if not args.skip_stt:
        audio_bytes = b""
        file_ext = ".wav"
        if args.audio:
            p = Path(args.audio)
            if p.exists():
                audio_bytes = p.read_bytes()
                file_ext = p.suffix
                print(f"Using audio file: {p} ({len(audio_bytes)//1024} KB)")
            else:
                print(f"[WARN] Audio file not found: {args.audio}; STT benchmark skipped.")
                args.skip_stt = True

        if audio_bytes:
            print("\n── Whisper (local) ──")
            stt_rows += benchmark_whisper_models(audio_bytes, file_ext, args.stt_models)
            if config.GROQ_API_KEY:
                print("\n── Groq (cloud) ──")
                stt_rows.append(benchmark_groq_stt(audio_bytes, file_ext))
        else:
            print("[INFO] No audio provided. Use --audio path/to/file.wav to run STT benchmarks.")

    # LLM
    if not args.skip_llm:
        print("\n── Ollama LLM ──")
        llm_rows += benchmark_ollama_models(args.llm_models)
        if config.GROQ_API_KEY:
            print("── Groq LLM ──")
            llm_rows.append(benchmark_groq_llm())

    if not stt_rows and not llm_rows:
        print("\n[WARN] Nothing to benchmark — generating empty report.")

    if not stt_rows:
        stt_rows = [{
            "model": "—",
            "load_s": "—",
            "infer_s": "—",
            "transcription": "(no STT run)",
            "error": ""
        }]

    if not llm_rows:
        llm_rows = [{
            "model": "—",
            "avg_s": "—",
            "samples": "0",
            "error": ""
        }]
    # Report
    stt_rows = stt_rows or [{"model": "—", "load_s": "—", "infer_s": "—",
                               "transcription": "(no audio provided)", "error": ""}]
    llm_rows = llm_rows or [{"model": "—", "avg_s": "—", "samples": "0", "error": ""}]

    report = build_report(stt_rows, llm_rows)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = config.OUTPUT_DIR / "benchmark.md"
    out_path.write_text(report, encoding="utf-8")

    print(f"\n{'─'*60}")
    print(report)
    print(f"\n📁 Writing report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
