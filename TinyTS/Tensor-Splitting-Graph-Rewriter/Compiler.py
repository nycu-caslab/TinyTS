import json
import copy
import argparse
import os
from MyGraph import Graph
from AutoSplit import Splitter
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--schema_path", nargs='?', default="utils/schema.fbs")
parser.add_argument("--out_path", nargs='?', default="__DEFAULT__")
parser.add_argument("--exec_order", nargs='?', default="DF")
parser.add_argument("--split_height", nargs='?', type=int, default=2)
parser.add_argument("--pad_fusion", action='store_true')

args = parser.parse_args()
filename = os.path.basename(args.model_path)
model_name = os.path.splitext(filename)[0]
schema_path = args.schema_path

tmp_dir = tempfile.TemporaryDirectory(dir='.')
tmp_dir_path = tmp_dir.name

if args.out_path == '__DEFAULT__':
    args.out_path = os.path.splitext(args.model_path)[0] + f"_splitted_{args.exec_order}_{args.split_height}.tflite"

if os.path.splitext(filename)[1] != '.tflite':
    raise "input model path doesn't match: .tflite extension is required'"


json_model_path = os.path.join(tmp_dir_path, f'{model_name}.json')
os.system(f'flatc --json -o {tmp_dir_path} --raw-binary {schema_path} -- {args.model_path}')
os.system(r'sed -i "s/\([^ ]*\):/\"\1\":/" ' + json_model_path)

with open(json_model_path,'r') as f:
    model = json.load(f)

opcodes = model['operator_codes']
buffers = model['buffers']
subgraphs = model["subgraphs"]
tensors = subgraphs[0]["tensors"]
operators = subgraphs[0]["operators"]

new_opcodes = copy.deepcopy(opcodes)

has_split = False
has_concat = False

for opcode in opcodes:
    if opcode.get('deprecated_builtin_code',0) == 2:
        has_concat = True
    elif opcode.get('deprecated_builtin_code',0) ==49:
        has_split = True

if has_concat == False:
    new_opcodes.append({
        "deprecated_builtin_code": 2,
        "version": 1,
        "builtin_code": "CONCATENATION"
        })
if has_split == False:
    new_opcodes.append({
        "deprecated_builtin_code": 49,
        "version": 1,
        "builtin_code": "SPLIT"
        })


new_model = copy.deepcopy(model)

new_model['operator_codes'] = new_opcodes

tensor_id_mapping = [ x for x in range(len(tensors))]

ori_graph = Graph(operators, tensors, buffers, new_opcodes, subgraphs[0]['inputs'], subgraphs[0]['outputs'], args.exec_order)

splitter = Splitter(ori_graph, args.split_height)
if args.pad_fusion:
    splitter.PaddingFusion()
new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = splitter.perform_split()

new_model['buffers'] = new_buffers
new_model['subgraphs'][0]['tensors'] = new_tensors
new_model['subgraphs'][0]['inputs'] = new_inputs
new_model['subgraphs'][0]['outputs'] = new_outputs
new_model['subgraphs'][0]['operators'] = new_operators

with open(json_model_path, 'w') as f:
    json.dump(new_model, f, indent=2)

os.system(f'flatc -o {tmp_dir_path} --binary {schema_path} {json_model_path}')
os.system(f'mv {os.path.join(tmp_dir_path, filename)} {args.out_path}')

tmp_dir.cleanup()
