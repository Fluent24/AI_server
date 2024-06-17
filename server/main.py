import os
import sys
import shutil
import tempfile
import subprocess
import boto3
import json
import concurrent.futures

sys.path.append('/usr/bin/ffmpeg')

import torchaudio  # PyTorch audio library
from fastapi import FastAPI, File, Response, UploadFile, HTTPException, Query, Form
from fastapi.responses import FileResponse, JSONResponse  # FastAPI response types

from speechbrain.inference.ASR import EncoderDecoderASR  # SpeechBrain ASR model
from speechbrain.inference.TTS import Tacotron2  # SpeechBrain TTS model
from speechbrain.inference.vocoders import HIFIGAN  # SpeechBrain vocoder model

from .inference_wav_SVR import inference_wav

app = FastAPI()

tts_dir = os.path.join(tempfile.gettempdir(), "tmpdir_tts")
vocoder_dir = os.path.join(tempfile.gettempdir(), "tmpdir_vocoder")
asr_dir = os.path.join(tempfile.gettempdir(), "pretrained_models/asr-crdnn-rnnlm-librispeech")




# New client for Bedrock Runtime
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "meta.llama3-70b-instruct-v1:0"
async def save_temp_file(file: UploadFile, suffix: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
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
            audio_len_max=440000,
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
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir=asr_dir
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  
        futures = []
        for file in files:
            try:
                # Save uploaded m4a file to a temporary location
                m4a_file_path = await save_temp_file(file, suffix=".m4a")

                # Convert m4a to wav
                wav_file_path = m4a_file_path.replace(".m4a", ".wav")
                result = subprocess.run(["ffmpeg", "-i", m4a_file_path, wav_file_path], check=True)

                # Submit parallel inference tasks
                prosody_future = executor.submit(
                    inference_wav, 
                    lang="en",
                    label_type1="pron",
                    label_type2="prosody",
                    dir_model='/mnt/f/fluent/AI_server/server/model_svr_ckpt',
                    device="cpu",
                    audio_len_max=400000,
                    wav=wav_file_path
                )
                articulation_future = executor.submit(
                    inference_wav,
                    lang="en",
                    label_type1="pron",
                    label_type2="articulation",
                    dir_model='/mnt/f/fluent/AI_server/server/model_svr_ckpt',
                    device="cpu",
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
