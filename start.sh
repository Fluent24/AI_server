#!/bin/bash
pip install -r requirements.txt
#!/bin/bash

# Uvicorn 프로세스 찾기
pid=$(ps aux | grep uvicorn | grep -v grep | awk '{print $2}')

# 프로세스가 존재하면 종료
if [ ! -z "$pid" ]; then
    echo "Uvicorn is running with PID $pid, killing..."
    kill -9 $pid
    sleep 1  # 잠시 대기 후 프로세스가 종료되었는지 확인
    echo "Uvicorn process killed."
else
    echo "No running Uvicorn process found."
fi

# Uvicorn 서버 재시작
echo "Starting Uvicorn server..."
nohup uvicorn server.main:app --reload --host=0.0.0.0 --port=10010 > server.log 2>&1 &

echo "Uvicorn server started."