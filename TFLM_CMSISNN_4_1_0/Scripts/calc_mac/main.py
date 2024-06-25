import tflite
import OP
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str, nargs='?', default='ResNet_MLPerf_quant.tflite')
args = parser.parse_args()

with open(args.model_path, 'rb') as f:
    buf = f.read()
    model = tflite.Model.GetRootAsModel(buf)
    calc = OP.MAC_calculator(model, silent = True)

    model_file_name = os.path.basename(args.model_path)
    model_name = os.path.splitext(model_file_name)[0]
    tot_mac = calc.CalcMAC()

    print(f"{model_name}: {tot_mac}")
    # print(f"\nModel Name: {model_name}")
    # print(f"Model Path: {args.model_path}")
    # print(f"Total: {tot_mac}")