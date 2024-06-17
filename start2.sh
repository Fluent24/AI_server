#!/bin/bash
pip install -r requirements.txt

Uvicorn 프로세스 찾기
pid=$(ps aux | grep uvicorn | grep -v grep | awk '{print $2}')

프로세스가 존재하면 종료
if [ ! -z "$pid" ]; then
    echo "Uvicorn is running with PID $pid, killing..."
    kill -9 $pid
    sleep 2  # 잠시 대기 후 프로세스가 종료되었는지 확인
    # 종료 확인
    if ps -p $pid > /dev/null; then
        echo "Uvicorn process did not terminate, force killing again..."
        kill -9 $pid
        sleep 2
    fi
    echo "Uvicorn process killed."
else
    echo "No running Uvicorn process found."
fi

포트 점유 확인 및 해제
port=10010
if lsof -i:$port; then
    echo "Port $port is in use, freeing it..."
    fuser -k $port/tcp
    sleep 2
    echo "Port $port freed."
fi

Uvicorn 서버 재시작
echo "Starting Uvicorn server..."
rm -f app.log
rm -f server.log
nohup python3 -m uvicorn server.main:app --reload --host=0.0.0.0 --port=10010 --log-config logging.yaml > server.log 2>&1 &
echo "Uvicorn server started."
