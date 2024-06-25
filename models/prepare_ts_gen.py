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
parser.add_argument("--evict_in", action='store_true', default=False)
args = parser.parse_args()
codegen_path = os.path.abspath(str(args.codegen_path))
codegen_folder = os.path.dirname(codegen_path)
schema_path = os.path.abspath(str(args.schema_path))
models_dir = os.path.abspath(str(args.models_dir))
evict_in = args.evict_in

os.chdir(codegen_folder)

tflite_list = [path for path in glob.glob(f'{models_dir}/**/*_splitted_*.tflite', recursive=True)]

commands = []

for tflite_path in sorted(tflite_list):
    model_name = os.path.splitext(os.path.basename(tflite_path))[0]
    print(model_name)
    match_pattern = re.search('_splitted_(.F)_([0-9]+)$', model_name)
    exec_order  = match_pattern.groups()[0]
    split_height = match_pattern.groups()[1]
    if evict_in:
        output_root_dir = os.path.dirname(tflite_path).replace('ts_model/', 'ts_gen_evict_in/')
    else:
        output_root_dir = os.path.dirname(tflite_path).replace('ts_model/', 'ts_gen/')
    os.makedirs(output_root_dir, exist_ok=True)
    print(output_root_dir)
    
    cmd =   f"python {codegen_path} {tflite_path} --out_dir {output_root_dir}" \
            f" --schema_path {schema_path} --split_height {split_height}" 
    if evict_in:
        cmd += ' --exp_esti --evict_in'
    
    enabled_mem_plan_addons = []
    if "MCUNet" in output_root_dir:
        if "IM10" in output_root_dir:
            # IM10 uses inplace_concat
            enabled_mem_plan_addons.append("inplace_concat")
        elif (('MW5' in output_root_dir or 'IM320' in output_root_dir) and
            (exec_order == 'DF' and split_height == '2')):
            # MW5_DF2 and IM320_DF2 use inplace concat
            enabled_mem_plan_addons.append("inplace_concat")
        else:
            # Most of MCUNet models use inplace split
            enabled_mem_plan_addons.append("inplace_split")
    else:
        if 'kws_MLPerf' in output_root_dir:
            # kws_MLPerf uses inplace concat
            enabled_mem_plan_addons.append("inplace_concat")
        elif 'vww_MLPerf' in output_root_dir:
            # vww_MLPerf uses inplace split
            enabled_mem_plan_addons.append("inplace_split")
        else:
            # Remained benchmark models use out-of-place 
            pass
    if len(enabled_mem_plan_addons) > 0:
        cmd += f' --enabled_mem_plan_addons "{",".join(enabled_mem_plan_addons)}"'

    commands.append(cmd)

procs = [subprocess.Popen(cmd.split()) for cmd in commands]

for p in procs:
    p.wait()
