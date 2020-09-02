import sys


def main(args):
    import argparse
    import json
    from pathlib import Path
    from seldon_core.seldon_client import SeldonClient
    import numpy as np
    parser = argparse.ArgumentParser(description='Model Serving Example')

    # 算子输入参数
    parser.add_argument("--deployment", type=str, required=True)
    parser.add_argument("--namespace", type=str, required=True)
    parser.add_argument("--ndarray", type=str, required=True)
    parser.add_argument("--names", type=str, required=True)

    # 0.1版本自定义算子参数暂不持之file big data传输, 0.3会支持
    # parser.add_argument("--data_path", type=str, required=True)

    # 算子输出, 系统会自动分配一个路径, 输出值必须写入这个路径才会被系统识别打包存到S3, 为其他算子输入引用做准备
    parser.add_argument("--output", type=str, required=True)

    GATEWAY_ENDPOINT = 'istio-ingressgateway.istio-system.svc.cluster.local'
    GATEWAY = 'istio'

    parsed_args = parser.parse_args(args)
    # 输入变量
    deployment = parsed_args.deployment
    namespace = parsed_args.namespace
    ndarray = json.loads(parsed_args.ndarray)  # 这个地方需要json.loads 是因为数据是以json array string为参数值传递给main函数
    names = json.loads(parsed_args.names)  # 这个地方需要json.loads 是因为数据是以json array string为参数值传递给main函数

    # 输出变量
    output_path = parsed_args.output

    np_data = np.asarray(ndarray)
    payload_type = "ndarray"

    sc = SeldonClient(
        namespace=namespace,
        deployment_name=deployment,
        gateway_endpoint=GATEWAY_ENDPOINT,
        gateway=GATEWAY
    )

    print('np_data: {}'.format(np_data))
    print('names: {}'.format(names))
    client_prediction = sc.predict(
        data=np_data,
        names=names,
        payload_type=payload_type,
        transport="rest")

    response = client_prediction.response
    if response and 'data' in response and 'ndarray' in response['data']:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as output_f:
            output_f.write('{}'.format(client_prediction.response['data']['ndarray']))
    else:
        print('Failed, client prediction response is {}'.format(json.dumps(response)))
        exit(1)

if __name__ == '__main__':
    main(sys.argv[1:])
