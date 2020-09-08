import sys
import argparse
from pathlib import Path
import numpy as np

def do_work(input1_file, output1_file, parameter1):
    output_text = ''
    output1_file.write(output_text)

def main(args):
    parser = argparse.ArgumentParser(description='Time-Series-Decomposition Input and Output')

    # 算子输入参数
    parser.add_argument("--input1", type=list, required=True)
    parser.add_argument("--parameter1", type=str, required=True)

    # 算子输出, 系统会自动分配一个路径, 输出值必须写入这个路径才会被系统识别打包存到S3, 为其他算子输入引用做准备
    parser.add_argument("--output1", type=list, required=True)
    args = parser.parse_args(args)

    # 创建输出路径，存放输出文件（路径可以存在，也可以不存在）
    Path(args.output1).parent.mkdir(parents=True, exist_ok=True)

    with open(args.input1, 'r') as input1_file:
        with open(args.output1, 'w') as output1_file:
            do_work(input1_file, output1_file, args.parameter1)

if __name__ == '__main__':
    main(sys.argv[1:])
