# AI_server

### 설치 코드
`sh start.sh`
### 서버 중지  
pkill uvicorn

cd server  
vi main.py  
dir_model='/home/coldbrew/fluent/scoring_system/fastapi_server/server/model_ckpt/' -> '/home/ec2-user/scoring_system/fastapi_server/serverfastapi_server/server'
### fast api 실행 코드
`sh start.sh`

### 필요한 파일
```
  server - main.py
         - inference_wav.py
         - score_model.py
         - score_input.m4a
         - model_ckpt/
           - pron_articulation_checkpoint.pt
           - pron_prosody_checkpoint.pt
```

### 요청 예시
- 요청 :   
$ `curl -X POST "http://localhost:10010/infer/" -H "Content-Type: multipart/form-data" -F "files=@score_input.m4a"  `
- 답변 :   
`[{"filename":"score_input.m4a","score_prosody":3.5421228408813477,"score_articulation":2.034834384918213,"total_score":5.5769572257995605}]`  


### server/inference_wav.py
- 함수 이름 : `def inference_wav(lang: str, label_type1: str, label_type2: str, dir_model: str, device: str, audio_len_max: int, wav: str): ` 
- 함수 출력 : `return pred_score[0][0]`