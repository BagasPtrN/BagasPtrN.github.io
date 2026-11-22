"""
Backend pengenal suara Batak Toba untuk halaman live caption.

Menerima potongan audio (WAV 16 kHz mono, ~3–8 detik) dari halaman web,
mengubahnya jadi teks Batak Toba dengan Meta MMS (adapter `bbc`), lalu
(opsional) menerjemahkannya ke bahasa Indonesia.

Endpoint API: /transcribe  ->  (teks_batak, teks_indonesia)

Variabel lingkungan (Settings → Variables di Space):
  MODEL_ID   default: facebook/mms-1b-all
  ASR_LANG   default: bbc   (kode MMS untuk Batak Toba)
  QUANTIZE   default: 1     (int8 di CPU, ±2x lebih cepat)
  TARGET_LANG default: id   (bahasa tujuan terjemahan)
"""

import os
import time
from functools import lru_cache

import numpy as np
import requests
import soundfile as sf
import torch
import gradio as gr
from transformers import AutoProcessor, Wav2Vec2ForCTC

# ZeroGPU (khusus akun PRO) – kalau paket `spaces` tersedia, pakai GPU.
try:
    import spaces  # type: ignore

    GPU = spaces.GPU
    ON_ZERO_GPU = True
except Exception:  # bukan ZeroGPU
    def GPU(fn=None, **_):
        return fn if fn else (lambda f: f)

    ON_ZERO_GPU = False

MODEL_ID = os.getenv("MODEL_ID", "facebook/mms-1b-all")
ASR_LANG = os.getenv("ASR_LANG", "bbc")
TARGET_LANG = os.getenv("TARGET_LANG", "id")
SR = 16_000

DEVICE = "cuda" if (ON_ZERO_GPU or torch.cuda.is_available()) else "cpu"
torch.set_num_threads(max(1, os.cpu_count() or 1))

print(f"[init] memuat {MODEL_ID} (bahasa={ASR_LANG}) di {DEVICE} …", flush=True)
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, target_lang=ASR_LANG)
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_ID, target_lang=ASR_LANG, ignore_mismatched_sizes=True
)
model.eval()

if DEVICE == "cpu" and os.getenv("QUANTIZE", "1") == "1":
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("[init] kuantisasi int8 aktif", flush=True)
    except Exception as e:  # tetap jalan tanpa kuantisasi
        print(f"[init] kuantisasi gagal, lanjut fp32: {e}", flush=True)

if DEVICE == "cuda":
    model = model.half()
model.to(DEVICE)
print(f"[init] siap dalam {time.time() - t0:.1f} dtk", flush=True)


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:  # resample sederhana (halaman web sudah kirim 16 kHz)
        n = int(round(len(audio) * SR / sr))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio


@GPU(duration=10)
def run_asr(audio: np.ndarray) -> str:
    inputs = processor(audio, sampling_rate=SR, return_tensors="pt")
    values = inputs["input_values"].to(DEVICE)
    if DEVICE == "cuda":
        values = values.half()
    with torch.inference_mode():
        logits = model(values).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return processor.decode(ids).strip()


@lru_cache(maxsize=2048)
def translate(text: str, sl: str = "bbc", tl: str = TARGET_LANG) -> str:
    """Google Translate (endpoint publik gtx). Tidak resmi: bisa dibatasi
    sewaktu-waktu. Ganti fungsi ini kalau mau pakai API resmi / model sendiri."""
    if not text:
        return ""
    try:
        r = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": sl, "tl": tl, "dt": "t", "q": text},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return "".join(seg[0] for seg in data[0] if seg and seg[0]).strip()
    except Exception as e:
        print(f"[translate] gagal: {e}", flush=True)
        return ""


def transcribe(audio_path, do_translate=True):
    if not audio_path:
        return "", ""
    t = time.time()
    audio = load_audio(audio_path)
    if len(audio) < SR * 0.3:
        return "", ""
    # normalisasi volume supaya mic pelan tetap terbaca
    peak = float(np.abs(audio).max()) or 1.0
    audio = audio / peak * 0.9
    batak = run_asr(audio)
    indo = translate(batak) if (do_translate and batak) else ""
    print(
        f"[asr] {len(audio)/SR:.1f}s audio → {time.time()-t:.1f}s | {batak!r}",
        flush=True,
    )
    return batak, indo


with gr.Blocks(title="Pengenal Suara Batak Toba") as demo:
    gr.Markdown(
        "## 🎙️ Pengenal Suara Batak Toba (MMS `bbc`)\n"
        "Backend untuk halaman live caption. Bisa juga dicoba langsung di sini."
    )
    with gr.Row():
        audio_in = gr.Audio(
            type="filepath", sources=["upload", "microphone"], label="Audio"
        )
        with gr.Column():
            do_tr = gr.Checkbox(value=True, label="Terjemahkan ke bahasa Indonesia")
            btn = gr.Button("Transkripsi", variant="primary")
    out_bt = gr.Textbox(label="Batak Toba")
    out_id = gr.Textbox(label="Bahasa Indonesia")
    btn.click(
        transcribe, [audio_in, do_tr], [out_bt, out_id], api_name="transcribe"
    )

demo.queue(default_concurrency_limit=1, max_size=64).launch(show_error=True)
