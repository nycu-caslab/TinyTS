import glob
import os
import subprocess
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--codegen_path", nargs='?', type=str,
                    default="TinyTS/Tensor-Splitting-Code-Generator/main.py")
parser.add_argument("--schema_path", nargs='?', type=str,
                    default="TinyTS/Tensor-Splitting-Code-Generator/utils/schema.fbs")
parser.add_argument("--models_dir", nargs='?', type=str,
                    default="models/")
parser.add_argument("--output_dir", nargs='?', type=str,
                    default="output")
parser.add_argument("--evict_in", action='store_true', default=False)
args = parser.parse_args()
codegen_path = os.path.abspath(str(args.codegen_path))
codegen_folder = os.path.dirname(codegen_path)
schema_path = os.path.abspath(str(args.schema_path))
models_dir = os.path.abspath(str(args.models_dir))
output_dir = os.path.abspath(str(args.output_dir))
evict_in = args.evict_in

os.chdir(codegen_folder)

output_log = []
tflite_list = [path for path in glob.glob(f'{models_dir}/**/*_splitted_*.tflite', recursive=True)]
for tflite_path in sorted(tflite_list):
    model_name = os.path.splitext(os.path.basename(tflite_path))[0]
    match_pattern = re.search('_splitted_(.F)_([0-9]+)$', model_name)
    exec_order  = match_pattern.groups()[0]
    split_height = match_pattern.groups()[1]
    cmd =   f"python {codegen_path} {tflite_path}" \
            f" --schema_path {schema_path} --split_height {split_height}" 
    enabled_mem_plan_addons = []
    if "MCUNet" in tflite_path:
        if "IM10" in tflite_path:
            # IM10 uses inplace_concat
            enabled_mem_plan_addons.append("inplace_concat")
        elif (('MW5' in tflite_path or 'IM320' in tflite_path) and
            (exec_order == 'DF' and split_height == '2')):
            # MW5_DF2 and IM320_DF2 use inplace concat
            enabled_mem_plan_addons.append("inplace_concat")
        else:
            # Most of MCUNet models use inplace split
            enabled_mem_plan_addons.append("inplace_split")
    else:
        if 'kws_MLPerf' in tflite_path:
            # kws_MLPerf uses inplace concat
            enabled_mem_plan_addons.append("inplace_concat")
        elif 'vww_MLPerf' in tflite_path:
            # vww_MLPerf uses inplace split
            enabled_mem_plan_addons.append("inplace_split")
        else:
            # Remained benchmark models use out-of-place 
            pass
    if len(enabled_mem_plan_addons) > 0:
        cmd += f' --enabled_mem_plan_addons "{",".join(enabled_mem_plan_addons)}"'
    cmd += f' -v'
    if evict_in:
        cmd += f' --exp_esti --evict_in'

    
    output = subprocess.check_output(cmd.split(' ')).decode()
    m = re.search('SRAM requirement for (.*)_splitted_([DB]F_[0-9]+): ([0-9,]+) B', output).groups()
    m = list(m)
    m[-1] = str(round(int(m[-1].replace(',', ''))/1000.0))
    output = ' '.join(m)
    print(output)
    output_log.append(output)
    exit(1)
with open(os.path.join(output_dir, 'TinyTS_SRAM.txt'), 'w') as f:
    f.write('\n'.join(output_log))

