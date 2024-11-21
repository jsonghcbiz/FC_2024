# Base Image 설정
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 로컬 파일을 컨테이너 내부로 복사
COPY hello.py /app

# 컨테이너 실행시 기본 명령어 설정
CMD ["python", "hello.py"]