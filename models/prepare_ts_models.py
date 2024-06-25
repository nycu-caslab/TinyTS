import glob
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rewriter_path", nargs='?', type=str,
                    default="TinyTS/Tensor-Splitting-Graph-Rewriter/Compiler.py")
parser.add_argument("--schema_path", nargs='?', type=str,
                    default="TinyTS/Tensor-Splitting-Code-Generator/utils/schema.fbs")
parser.add_argument("--models_dir", nargs='?', type=str,
                    default="models/")
args = parser.parse_args()
rewriter_path = str(args.rewriter_path)
schema_path = str(args.schema_path)
models_dir = str(args.models_dir)

tflite_list = [path for path in glob.glob(f'{models_dir}/**/tflite/**/*.tflite', recursive=True)]
for tflite_path in sorted(tflite_list):
    model_name = os.path.splitext(os.path.basename(tflite_path))[0]
    folder_root = os.path.dirname(tflite_path).replace('tflite/', 'ts_model/')
    for split_height in [1,2]:
        for exec_order in ['DF', 'BF']:
            subfolder_dir = os.path.join(folder_root, f"{exec_order}_{split_height}")
            os.makedirs(subfolder_dir, exist_ok=True)
            out_path = f"{subfolder_dir}/{model_name}_splitted_{exec_order}_{split_height}.tflite"
            print(out_path)
            cmd =   f"python {rewriter_path} {tflite_path} --schema_path {schema_path}" \
                    f" --exec_order {exec_order} --split_height {split_height}" \
                    f" --out_path {out_path}"
            if 'MCUNet' in subfolder_dir:
                cmd += " --pad_fusion"
            subprocess.run(cmd.split(' '))
