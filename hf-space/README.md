---
title: Batak Toba ASR
emoji: 🎙️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

Backend pengenal suara Batak Toba (Meta MMS, adapter `bbc`) untuk halaman
live caption. Endpoint API: `/transcribe(audio, terjemahkan) -> (batak, indonesia)`.

Model: [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) — CC-BY-NC 4.0 (nonkomersial).
