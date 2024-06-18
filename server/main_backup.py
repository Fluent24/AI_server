import os
import tempfile
import subprocess
import boto3
import json
import asyncio
import aiofiles
import torch
import numpy as np
import joblib
from transformers import Wav2Vec2ForCTC
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import FileResponse, JSONResponse
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
import torchaudio
import audiofile

app = FastAPI()

tts_dir = os.path.join(tempfile.gettempdir(), "tmpdir_tts")
vocoder_dir = os.path.join(tempfile.gettempdir(), "tmpdir_vocoder")
asr_dir = os.path.join(tempfile.gettempdir(), "pretrained_models/asr-crdnn-rnnlm-librispeech")

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "meta.llama3-70b-instruct-v1:0"

async def save_temp_file(file: UploadFile, suffix: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir='/tmp') as temp_file:
        temp_file.write(await file.read())
        return temp_file.name

def transcribe_file(asr_model, file_path: str):
    try:
        transcription = asr_model.transcribe_file(file_path)
        return transcription
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

@app.get("/tts/")
async def tts(text: str):
    tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir=tts_dir)
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=vocoder_dir)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        filepath = temp_file.name
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    torchaudio.save(filepath, waveforms.squeeze(1), 22050)
    headers = {"Content-Disposition": f'attachment; filename="{os.path.basename(filepath)}"'}
    return FileResponse(filepath, media_type="audio/wav", headers=headers)

@app.post("/tts/")
async def tts(text: str = Form(...)):
    tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir=tts_dir)
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=vocoder_dir)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        filepath = temp_file.name
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    torchaudio.save(filepath, waveforms.squeeze(1), 22050)
    headers = {"Content-Disposition": f'attachment; filename="{os.path.basename(filepath)}"'}
    return FileResponse(filepath, media_type="audio/wav", headers=headers)

@app.post("/stt/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir=asr_dir)
        temp_dir = os.path.join(os.getcwd(), 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        try:
            subprocess.run(["sudo", "chmod", "777", temp_file_path], check=True)
        except subprocess.CalledProcessError as e:
            os.remove(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Failed to set file permissions: {str(e)}")
        transcription = asr_model.transcribe_file(temp_file_path)
        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")

async def convert_m4a_to_wav(m4a_file_path, wav_file_path):
    process = await asyncio.create_subprocess_exec("ffmpeg", "-i", m4a_file_path, wav_file_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise Exception(f"FFmpeg error: {stderr.decode()}")

global_svr_models = {}
global_scalers = {}
global_base_model = None
asr_model = None

def load_models_and_scalers():
    languages = ['en']
    label_types = [('pron', 'prosody'), ('pron', 'articulation')]
    dir_model = '/mnt/f/fluent/AI_server/server/model_svr_ckpt'
    for lang in languages:
        for label_type1, label_type2 in label_types:
            model_path = os.path.join(dir_model, f'lang_{lang}/svr_model_{label_type1}+{label_type2}.joblib')
            scaler_path = os.path.join(dir_model, f'lang_{lang}/scaler_{label_type1}+{label_type2}.joblib')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
            svr_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            global_svr_models[(label_type1, label_type2)] = svr_model
            global_scalers[(label_type1, label_type2)] = scaler
    global global_base_model
    base_model_name = 'facebook/wav2vec2-large-robust-ft-libri-960h'
    global_base_model = Wav2Vec2ForCTC.from_pretrained(base_model_name).to('cuda')
    global asr_model
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir=asr_dir)
    print(f"Models and scalers loaded successfully.")

load_models_and_scalers()

def inference_wav(label_type1: str, label_type2: str, device: str, audio_len_max: int, wav: str):
    svr_model = global_svr_models[(label_type1, label_type2)]
    scaler = global_scalers[(label_type1, label_type2)]
    base_model = global_base_model
    x, sr = audiofile.read(wav)
    x = torch.tensor(x[:min(x.shape[-1], audio_len_max)], device=device).reshape(1, -1)
    with torch.no_grad():
        feat_x = base_model(x, output_attentions=False, output_hidden_states=True, return_dict=True).hidden_states[-1]
        feat_x = torch.mean(feat_x, axis=1).cpu().numpy()
    del x
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    feat_x = scaler.transform(feat_x)
    pred_score = svr_model.predict(feat_x)
    pred_score = np.clip(pred_score, 0, 5)
    del feat_x
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return pred_score[0]

@app.post("/infer/")
async def predict(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        m4a_file_path = None
        wav_file_path = None
        try:
            async with aiofiles.tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_m4a_file:
                m4a_file_path = temp_m4a_file.name
                await temp_m4a_file.write(await file.read())
            wav_file_path = m4a_file_path.replace(".m4a", ".wav")
            await convert_m4a_to_wav(m4a_file_path, wav_file_path)
            score_prosody = inference_wav(label_type1="pron", label_type2="prosody", device="cuda", audio_len_max=400000, wav=wav_file_path)
            transcription = transcribe_file(asr_model, wav_file_path)
            score1 = float(score_prosody)
            score2 = score1
            total_score = score1 * 2
            results.append({"filename": file.filename, "score_prosody": score1, "score_articulation": score2, "total_score": total_score, "transcription": transcription})
        finally:
            if m4a_file_path and os.path.exists(m4a_file_path):
                os.remove(m4a_file_path)
            if wav_file_path and os.path.exists(wav_file_path):
                os.remove(wav_file_path)
    return JSONResponse(content=results)

@app.get("/generate-sentences/")
async def generate_sentences(category: str = Query(..., description="Category for sentence generation")):
    categories = ["travel", "romance", "exercise", "meetings", "food", "movies", "music"]
    if category not in categories:
        raise HTTPException(status_code=400, detail="Invalid category. Choose from travel, romance, exercise, meetings, food, movies, music.")
    
    user_message = f"Generate 10 different {category}-related English sentences at each level: Bronze, Silver, Gold, Diamond, and Master."
    prompt = f"user\n{user_message}\nassistant\n"
    
    request = {"prompt": prompt, "max_gen_len": 2048, "temperature": 0.5, "top_p": 0.9}
    response = bedrock_client.invoke_model(body=json.dumps(request), modelId=model_id)
    model_response = json.loads(response["body"].read())
    response_text = model_response["generation"]
    
    return {"generated_text": response_text}