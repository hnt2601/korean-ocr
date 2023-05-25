docker build -t hoangnt/korean-ocr:v1 -f Dockerfile .
docker run --gpus all --name kor-ocr -p 8009:8009 --network="ocr" -v /media/hoangnt/Projects/Meditech/korean-ocr:/workspace -dit hoangnt/korean-ocr:v1
