import glob
import os
import subprocess
import re

for tflite_path in glob.glob('**/*.tflite', recursive=True):
    model_name = os.path.basename(tflite_path).rstrip('.tflite')
    output_path = os.path.dirname(tflite_path.replace('tflite/', 'tflm_format/')) + '/' + 'model_data.cc'
    print(output_path)
    xxd_ret = subprocess.check_output(f"xxd -i {tflite_path}".split(' ')).decode('UTF-8')
    tflm_format = re.sub('unsigned char .*_tflite\\[\\] = {', 'alignas(16) const unsigned char model[] = {', xxd_ret)
    tflm_format = re.sub('unsigned int .*_len', 'const unsigned int model_len', tflm_format)
    tflm_format = ( '#include "model/model_data.h"\n' + 
                   f'char model_name[]="{model_name}";\n' +
                    'char compile_time[]=__TIMESTAMP__;\n') + tflm_format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(tflm_format)
