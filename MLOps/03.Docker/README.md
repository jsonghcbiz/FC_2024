(1) Docker Destkop 다운, 설치

(2) dockerfile 생성
(3) dockerfile 빌드 
1. 이미지 빌드
> docker build -t hello-docker .
2. 이미지가 잘 빌드 되었는지 확인 
> docker images
3. 빌드된 이미지를 실행 (컨테이너를 띄운다)
> docker run '이미지 이름'