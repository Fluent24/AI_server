import os
import sys
import shutil
import tempfile
import subprocess
import boto3
import json
import concurrent.futures
import asyncio
import aiofiles
import numpy as np
import joblib
import torch
from transformers import Wav2Vec2ForCTC
import audiofile
sys.path.append('/usr/bin/ffmpeg')

import torchaudio  # PyTorch audio library
from fastapi import FastAPI, File, Response, UploadFile, HTTPException, Query, Form
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse, JSONResponse  # FastAPI response types

from speechbrain.inference.ASR import EncoderDecoderASR  # SpeechBrain ASR model
from speechbrain.inference.TTS import Tacotron2  # SpeechBrain TTS model
from speechbrain.inference.vocoders import HIFIGAN  # SpeechBrain vocoder model

#from .inference_wav_SVR import inference_wav
from .inference_wav_optim import inference_wav
from .inference_wav import inference_wav2
app = FastAPI()

tts_dir = os.path.join(tempfile.gettempdir(), "tmpdir_tts")
vocoder_dir = os.path.join(tempfile.gettempdir(), "tmpdir_vocoder")
asr_dir = os.path.join(tempfile.gettempdir(), "pretrained_models/asr-crdnn-rnnlm-librispeech")




# New client for Bedrock Runtime
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
    tacotron2 = Tacotron2.from_hparams(
    source="speechbrain/tts-tacotron2-ljspeech", savedir=tts_dir
    )
    hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech", savedir=vocoder_dir
    )
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

@app.post("/tts/")
async def tts(text: str = Form(...)):
    tacotron2 = Tacotron2.from_hparams(
    source="speechbrain/tts-tacotron2-ljspeech", savedir=tts_dir
    )
    hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech", savedir=vocoder_dir
    )
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
    try:
        # ASR 모델 초기화
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir=asr_dir
        )

        # 임시 디렉토리 설정
        temp_dir = os.path.join(os.getcwd(), 'tmp')
        os.makedirs(temp_dir, exist_ok=True)

        # 임시 파일을 지정된 디렉토리에 생성하고 파일 내용을 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        print("Temporary file path:", temp_file_path)

        # 파일 권한 설정 (sudo 권한으로)
        try:
            subprocess.run(["sudo", "chmod", "777", temp_file_path], check=True)
        except subprocess.CalledProcessError as e:
            os.remove(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Failed to set file permissions: {str(e)}")

        # 음성 파일을 텍스트로 변환
        try:
            transcription = asr_model.transcribe_file(temp_file_path)
            # os.remove(temp_file_path)  # 임시 파일 삭제
            return {"transcription": transcription}
        except Exception as e:
            os.remove(temp_file_path)  # 임시 파일 삭제
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")

@app.post("/infer/")
async def predict(files: list[UploadFile] = File(...)):
    results = []
    asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir=asr_dir
    )

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
            dir_model='/mnt/f/fluent/AI_server/server/model_svr_ckpt',
            device="cpu",
            audio_len_max=1000000,
            wav=wav_file_path
        )
        # 음성 파일을 텍스트로 변환
        try:
            transcription = asr_model.transcribe_file(wav_file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Transcribe failed")
            
        score1 = float(score_prosody)
        score2 = score1
        total_score = score1*2
        # Clean up temporary files
        os.remove(m4a_file_path)
        os.remove(wav_file_path)

        results.append({"filename": file.filename, "score_prosody": score1, "score_articulation": score2, "total_score" : total_score, "transcription":transcription})

    return JSONResponse(content=results)

@app.post("/infer2/")
async def predict(files: list[UploadFile] = File(...)):
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  
        futures = []
        for file in files:
            try:
                # Save uploaded m4a file to a temporary location in /tmp directory
                m4a_file_path = await save_temp_file(file, suffix=".m4a")

                # Convert m4a to wav
                wav_file_path = m4a_file_path.replace(".m4a", ".wav")
                result = subprocess.run(["ffmpeg", "-i", m4a_file_path, wav_file_path], check=True)

                # Submit parallel inference tasks
                prosody_future = executor.submit(
                    inference_wav2, 
                    lang="en",
                    label_type1="pron",
                    label_type2="prosody",
                    dir_model='/mnt/f/fluent/AI_server/server/model_ckpt',
                    device="cuda",
                    audio_len_max=400000,
                    wav=wav_file_path
                )
                articulation_future = executor.submit(
                    inference_wav2,
                    lang="en",
                    label_type1="pron",
                    label_type2="articulation",
                    dir_model='/mnt/f/fluent/AI_server/server/model_ckpt',
                    device="cuda",
                    audio_len_max=400000,
                    wav=wav_file_path
                )
                transcription_future = executor.submit(transcribe_file, asr_model, wav_file_path)

                futures.append((file, prosody_future, articulation_future, transcription_future, m4a_file_path, wav_file_path))

            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"File handling failed: {str(e)}")

        # Gather results
        for file, prosody_future, articulation_future, transcription_future, m4a_file_path, wav_file_path in futures:
            try:
                score_prosody = prosody_future.result()
                score_articulation = articulation_future.result()
                transcription = transcription_future.result()
                
                score1 = float(score_prosody)
                score2 = float(score_articulation)
                total_score = score1 + score2

                results.append({
                    "filename": file.filename,
                    "score_prosody": score1,
                    "score_articulation": score2,
                    "total_score": total_score,
                    "transcription": transcription
                })

            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })

            finally:
                os.remove(m4a_file_path)
                os.remove(wav_file_path)

    return JSONResponse(content=results)

async def convert_m4a_to_wav(m4a_file_path, wav_file_path):
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", m4a_file_path, wav_file_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise Exception(f"FFmpeg error: {stderr.decode()}")

asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir=asr_dir
)
@app.post("/infer3/") #svr model
async def predict(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        # Save uploaded m4a file to a temporary file
        try:
            async with aiofiles.tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_m4a_file:
                m4a_file_path = temp_m4a_file.name
                await temp_m4a_file.write(await file.read())

            # Convert m4a to wav
            wav_file_path = m4a_file_path.replace(".m4a", ".wav")
            await convert_m4a_to_wav(m4a_file_path, wav_file_path)

            # Perform inference using the inference_wav function
            score_prosody = inference_wav(
                lang="en",
                label_type1="pron",
                label_type2="prosody",
                dir_model='/mnt/f/fluent/AI_server/server/model_svr_ckpt',
                device="cuda",
                audio_len_max=400000,
                wav=wav_file_path
            )
            # 음성 파일을 텍스트로 변환
            try:
                transcription = asr_model.transcribe_file(wav_file_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail="Transcribe failed")

            score1 = float(score_prosody)
            score2 = score1
            total_score = score1 * 2

            results.append({"filename": file.filename, "score_prosody": score1, "score_articulation": score2, "total_score": total_score, "transcription": transcription})

        finally:
            # Clean up temporary files
            if os.path.exists(m4a_file_path):
                os.remove(m4a_file_path)
            if os.path.exists(wav_file_path):
                os.remove(wav_file_path)

    return JSONResponse(content=results)
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
    # ASR 모델 초기화
    global asr_model
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir=asr_dir
    )

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
    
    # Clear intermediate variables to free memory
    del x
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Scale features
    feat_x = scaler.transform(feat_x)

    # Predict pronunciation score using SVR
    pred_score = svr_model.predict(feat_x)
    pred_score = np.clip(pred_score, 0, 5)

    print(f'score: {pred_score[0]}')

    # Clear feature variable to free memory
    del feat_x
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return pred_score[0]

#@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models_and_scalers()
    yield

@app.post("/infer4/")
async def predict(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        m4a_file_path = None
        wav_file_path = None
        try:
            # Save uploaded m4a file to a temporary file
            async with aiofiles.tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_m4a_file:
                m4a_file_path = temp_m4a_file.name
                await temp_m4a_file.write(await file.read())

            # Convert m4a to wav
            wav_file_path = m4a_file_path.replace(".m4a", ".wav")
            await convert_m4a_to_wav(m4a_file_path, wav_file_path)

            # Perform inference using the inference_wav function
            score_prosody = inference_wav(
                label_type1="pron",
                label_type2="prosody",
                device="cuda",
                audio_len_max=400000,
                wav=wav_file_path
            )
            # 음성 파일
            try:
                transcription = transcribe_file(asr_model,wav_file_path)
                print(transcription)
            except Exception as e:
                raise HTTPException(status_code=500, detail="Transcribe failed")

            score1 = float(score_prosody)
            score2 = score1
            total_score = score1 * 2

            results.append({"filename": file.filename, "score_prosody": score1, "score_articulation": score2, "total_score": total_score, "transcription": transcription})

        finally:
            # Clean up temporary files
            if m4a_file_path and os.path.exists(m4a_file_path):
                os.remove(m4a_file_path)
            if wav_file_path and os.path.exists(wav_file_path):
                os.remove(wav_file_path)

    return JSONResponse(content=results)

# Utility function to convert m4a to wav
async def convert_m4a_to_wav(input_file, output_file):
    command = ["ffmpeg", "-i", input_file, output_file]
    process = await asyncio.create_subprocess_exec(*command)
    await process.communicate()

@app.post("/generate-sentences/")
async def generate_sentences(category: str = Query(..., description="Category for sentence generation")):
    categories = ["travel", "romance", "exercise", "meetings", "food", "movies", "music"]
    if category not in categories:
        raise HTTPException(status_code=400, detail="Invalid category. Choose from travel, romance, exercise, meetings, food, movies, music.")

    user_message = f"Generate 10 different {category}-related English sentences at each level: Bronze, Silver, Gold, Diamond, and Master."

    prompt = f"""
    user
    {user_message}

    assistant
    """

    request = {
        "prompt": prompt,
        "max_gen_len": 2048,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    response = bedrock_client.invoke_model(body=json.dumps(request), modelId=model_id)
    model_response = json.loads(response["body"].read())
    response_text = model_response["generation"]
    
    return {"generated_text": response_text}


@app.get("/generate-sentences/")
async def generate_sentences(category: str = Query(..., description="Category for sentence generation")):
    categories = ["travel", "romance", "exercise", "meetings", "food", "movies", "music"]
    if category not in categories:
        raise HTTPException(status_code=400, detail="Invalid category. Choose from travel, romance, exercise, meetings, food, movies, music.")

    user_message = f"Generate 10 different {category}-related English sentences at each level: Bronze, Silver, Gold, Diamond, and Master."

    prompt = f"""
    user
    {user_message}

    assistant
    """

    request = {
        "prompt": prompt,
        "max_gen_len": 2048,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    response = bedrock_client.invoke_model(body=json.dumps(request), modelId=model_id)
    model_response = json.loads(response["body"].read())
    response_text = model_response["generation"]
    
    return {"generated_text": response_text}
