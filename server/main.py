import os
import sys
import tempfile
import subprocess

sys.path.append('/usr/bin/ffmpeg')

import torchaudio  # PyTorch audio library
from fastapi import FastAPI, File, Response, UploadFile  # FastAPI web framework
from fastapi.responses import FileResponse, JSONResponse  # FastAPI response types

from speechbrain.inference.ASR import EncoderDecoderASR  # SpeechBrain ASR model
from speechbrain.inference.TTS import Tacotron2  # SpeechBrain TTS model
from speechbrain.inference.vocoders import HIFIGAN  # SpeechBrain vocoder model

from .inference_wav import inference_wav  # Custom inference module

app = FastAPI()

tacotron2 = Tacotron2.from_hparams(
    source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts"
)
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder"
)
asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")

@app.get("/tts/")
async def tts(text: str):
    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        filepath = temp_file.name

    # TTS 및 Vocoder 실행
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)

    # 음성 파일 저장
    torchaudio.save(filepath, waveforms.squeeze(1), 22050)

    # 생성된 파일 반환
    headers = {
        "Content-Disposition": f'attachment; filename="{os.path.basename(filepath)}"'
    }
    return FileResponse(
        filepath, media_type="audio/wav", headers=headers,
    )

@app.post("/stt/")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(await file.read())
        filepath = temp_file.name

    # 음성 파일을 텍스트로 변환
    try:
        transcription = asr_model.transcribe_file(filepath)
        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}

@app.post("/infer/")
async def predict(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        # Save uploaded m4a file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_m4a_file:
            m4a_file_path = temp_m4a_file.name
            temp_m4a_file.write(await file.read())

        # Convert m4a to wav
        wav_file_path = m4a_file_path.replace(".m4a", ".wav")
        subprocess.run(["ffmpeg", "-i", m4a_file_path, wav_file_path])

        # Perform inference using the inference_wav function
        score_prosody = inference_wav(
            lang="en",
            label_type1="pron",
            label_type2="prosody",
            dir_model='/home/coldbrew/fluent/scoring_system/fastapi_server/server/model_ckpt/',
            device="cpu",
            audio_len_max=200000,
            wav=wav_file_path
        )
        score_articulation = inference_wav(
            lang="en",
            label_type1="pron",
            label_type2="articulation",
            dir_model='/home/coldbrew/fluent/scoring_system/fastapi_server/server/model_ckpt/',
            device="cpu",
            audio_len_max=200000,
            wav=wav_file_path
        )
        score1 = float(score_prosody)
        score2 = float(score_articulation)
        total_score = score1 + score2
        # Clean up temporary files
        os.remove(m4a_file_path)
        os.remove(wav_file_path)

        results.append({"filename": file.filename, "score_prosody": score1, "score_articulation": score2, "total_score" : total_score})

    return JSONResponse(content=results)
