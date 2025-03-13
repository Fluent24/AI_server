# Fluent AI 서버
<img src="fluent%20Architecture.png" width="600" alt="Architecture">  
<img src="poster.png" width="600"  alt="poster">

이 저장소는 오디오 파일을 처리하고 텍스트로 변환(STT), 텍스트를 음성으로 변환(TTS), 그리고 발음 평가를 수행하는 다양한 스크립트를 포함하고 있습니다. 주요 기능은 FastAPI를 사용하여 RESTful API를 제공하는 것입니다.
## 목차

- [Fluent AI 서버](#fluent-ai-서버)
  - [목차](#목차)
  - [프로젝트 구조](#프로젝트-구조)
  - [설치](#설치)
  - [사용법](#사용법)
    - [서버 중지](#서버-중지)
    - [파일 경로 수정](#파일-경로-수정)
  - [요청 예시](#요청-예시)
    - [요청](#요청)
    - [답변](#답변)
  - [server/inference\_wav.py](#serverinference_wavpy)
  - [API 엔드포인트](#api-엔드포인트)
  - [스크립트 설명](#스크립트-설명)
  - [라이센스](#라이센스)

## 프로젝트 구조

```
.
├── Dockerfile
├── README.md
├── bedrock_example_codes
│   ├── bedrock.py
│   └── bedrock_wrapper.py
├── list.py
├── make.sh
├── requirements.txt
├── server
│   ├── inference_wav.py
│   ├── inference_wav_SVR.py
│   ├── main.py
│   ├── model_ckpt
│   │   └── lang_en #/infer/ 용
│   │       ├── pron_articulation_checkpoint.pt
│   │       └── pron_prosody_checkpoint.pt
|   ├── model_svr_ckpt (구글 드라이버에서 다운 필요) https://drive.google.com/file/d/1F71jnc8ycJK4A27Du95p_3yC-lnnIlv1/view?usp=share_link
│   │   └── lang_en #/infer2/ 용
│   │       ├── scaler_pron+articulation.joblib  
│   │       └── scaler_pron+prosody.joblib  
│   │       └── svr_model_pron+articulation.joblib  
│   │       └── svr_model_pron+prosody.joblib
│   ├── score_model.py
│   └── test.py
├── server.log
├── start.sh
└── tmpdir_tts
```

## 설치

필요한 의존성을 설치하려면 다음 명령을 실행하세요:

```bash
pip install -r requirements.txt
```

Docker를 사용하는 경우, Docker 이미지를 빌드하고 컨테이너를 실행할 수 있습니다:

```bash
docker build -t fluent-ai-server .
docker run -p 8000:8000 fluent-ai-server
```

## 사용법

FastAPI 서버를 실행하려면, 다음 명령을 실행하세요:

```bash
sh start.sh
```

### 서버 중지

```bash
pkill uvicorn
```

### 파일 경로 수정

`server/main.py` 파일에서 `dir_model` 경로를 수정하세요:

```python
dir_model='' -> './AI_server/server/model_ckpt'
```

## 요청 예시

### 요청

```bash
curl -X POST http://localhost:10010/infer2/ -H "Content-Type: multipart/form-data" -F "files=@/home/ec2-user/AI_server/server/exa
mple_TTS.wav"
```

### 답변

```json
[{"filename":"example_TTS.wav","score_prosody":4.49035915865262,"score_articulation":4.49035915865262,"total_score":8.98071831730524,"transcription":"DID HE WRITE STORIES WHEN HE WAS YOUNGER I THINK HE USED TO WRITE NOVELS"}]
```

## server/inference_wav.py

- **함수 이름**: `def inference_wav(lang: str, label_type1: str, label_type2: str, dir_model: str, device: str, audio_len_max: int, wav: str):`
- **함수 출력**: `return pred_score[0][0]`

## API 엔드포인트

- **TTS(Text-to-Speech)**: 텍스트를 음성 파일로 변환합니다.
  - **GET or POST `/tts/`**
  - 요청 파라미터: `text` (변환할 텍스트)
  - 응답: 생성된 음성 파일

- **STT(Speech-to-Text)**: 음성 파일을 텍스트로 변환합니다.
  - **POST `/stt/`**
  - 요청 파일: 음성 파일 (`UploadFile`)
  - 응답: 변환된 텍스트

- **발음 평가**: 오디오 파일의 발음을 평가합니다.
  - **POST `/infer/`**
  - 요청 파일: 오디오 파일 리스트 (`UploadFile`)
  - 응답: 평가 점수
  - 
- - **발음 평가2**: 오디오 파일의 발음을 평가 + STT합니다.
  - **POST `/infer2/`**
  - 요청 파일: 오디오 파일 리스트 (`UploadFile`)
  - 응답: 평가 점수  

- **문장 생성**: 특정 카테고리에 맞는 문장을 생성합니다.
  - **GET or POST `/generate-sentences/`**
  - 요청 파라미터: `category` (문장 생성 카테고리)
  - 응답: 생성된 문장

## 스크립트 설명

- `inference_wav.py`: 오디오 파일의 특징을 추출하고 모델을 통해 발음 점수를 예측합니다.
- `main.py`: FastAPI를 설정하고 API 엔드포인트를 정의합니다.
- `score_model.py`: 발음 평가를 위한 MLP 모델을 정의합니다.
- `bedrock_example_codes/`: AWS Bedrock을 사용한 예제 코드들입니다.
- `list.py`: 데이터셋 목록을 생성하는 스크립트입니다.
- `start.sh`: 서버를 시작하는 스크립트입니다.


## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
