import json
import copy
import argparse
import os
import tempfile

from code_gen import CodeGenerator

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--out_dir", nargs='?', default='.')
parser.add_argument("--schema_path", nargs='?', default='utils/schema.fbs')
parser.add_argument("--enabled_mem_plan_addons", nargs='?', default='')
parser.add_argument("--alignment", nargs='?', type=int, default=16)
parser.add_argument("--split_height", nargs='?', type=int, default=2)
parser.add_argument("--enable_te", action='store_true', default=True)
parser.add_argument("--exp_esti", action='store_true', default=False)
parser.add_argument("--evict_in", action='store_true', default=False)
parser.add_argument("--inplace_add", action='store_true', default=False)
parser.add_argument("--inplace_dw_conv", action='store_true', default=False)
parser.add_argument("-v", action='store_true')
parser.add_argument("-vv", action='store_true')
parser.add_argument("--plot", action='store_true')
args = parser.parse_args()

plot_result = args.plot
if args.vv:
    plot_result = True

filename = os.path.basename(args.model_path)
model_name = os.path.splitext(filename)[0]
schema_path = args.schema_path
out_dir = str(args.out_dir)
tmp_dir = tempfile.TemporaryDirectory(dir='.')
tmp_dir_path = tmp_dir.name

enabled_mem_plan_addons_str = str(args.enabled_mem_plan_addons).lstrip('"').rstrip('"')
enabled_mem_plan_addons = []
if len(enabled_mem_plan_addons_str) > 0:
    enabled_mem_plan_addons += enabled_mem_plan_addons_str.split(',')

enable_te = bool(args.enable_te)
exp_esti = bool(args.exp_esti)
evict_in = bool(args.evict_in)
inplace_add = bool(args.inplace_add)
inplace_dw_conv = bool(args.inplace_dw_conv)

if evict_in:
    try:
        enabled_mem_plan_addons.remove('inplace_split')
    except:
        pass

if os.path.splitext(filename)[1] != '.tflite':
    raise "input model path doesn't match: .tflite extension is required'"

json_model_path = os.path.join(tmp_dir_path, f'{model_name}.json')
os.system(f'flatc --json -o {tmp_dir_path} --raw-binary {schema_path} -- {args.model_path}')
os.system(r'sed -i "s/\([^ ]*\):/\"\1\":/" ' + json_model_path)
with open(json_model_path,'r') as f:
    model = json.load(f)

# cg = CodeGenerator(model, args.splitted)
cg = CodeGenerator(model, model_name, 
                    args.alignment, args.split_height,
                    plot_result, args.enable_te,
                    output_root_dir=out_dir,
                    enabled_mem_plan_addons=enabled_mem_plan_addons,
                    exp_esti=exp_esti,
                    evict_in=evict_in,
                    inplace_add=inplace_add,
                    inplace_dw_conv=inplace_dw_conv)
cg.GenCxtData()
cg.GenOpParam()
cg.GenEval()
if args.vv:
    cg.mem_profiler.Summary_verbose()
elif args.v == True:
    cg.mem_profiler.Summary()

tmp_dir.cleanup()

exit()
