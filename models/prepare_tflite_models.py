import glob
import requests
import os

for path in glob.glob('**/list.csv', recursive=True):
    benchmark_path = path.rstrip('list.csv')
    with open(path, 'r') as f:
        model_list = [{'name': line.split(sep=',')[0], 'download_link': line.split(sep=',')[1].rstrip()}
                            for line in f.readlines()]
        for model in model_list:
            model_path = f"{benchmark_path}/{model['name']}/{model['name']}.tflite"
            if os.path.exists(model_path):
                continue
            print(f"{model_path} downloaded")
            r = requests.get(url=model['download_link'], allow_redirects=True)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model_file = open(model_path, 'wb')
            model_file.write(r.content)
            model_file.close()

