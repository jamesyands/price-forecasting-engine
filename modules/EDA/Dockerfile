FROM harbor.eniot.io/eap/base/python3-x86
WORKDIR /pipeline/component/src
COPY ./src /pipeline/component/src
RUN pip3 install --upgrade pip -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip3 install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
ENTRYPOINT python3 time-series-decomposition.py
