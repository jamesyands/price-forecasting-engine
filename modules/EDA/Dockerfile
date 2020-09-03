FROM harbor.eniot.io/eap/base/python3-x86
WORKDIR /app
COPY requirements.txt /app
RUN pip3 install --upgrade pip -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip3 install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
COPY main.py /app
ENTRYPOINT python3 main.py
