docker run -it --rm sam-inference /bin/bash

docker build -t sam-inference .

aws ecr create-repository --repository-name sam-inference

docker tag sam-inference:latest (숫자).dkr.ecr.ap-northeast-2.amazonaws.com/sam-inference:latest

docker push 922377915118.dkr.ecr.ap-northeast-2.amazonaws.com/sam-inference:latest
