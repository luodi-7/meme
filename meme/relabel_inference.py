from logging import exception
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
import csv
import json
import os

def load_csv_data(path):
    with open(path, 'r', encoding = "utf-8") as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

def get_list_from_file(data_path):
    with open(f'/mnt/afs/xueyingyi/meme/prompt_relabel.txt', 'r') as file:
        PROMPT = file.read()

    image_path = '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/'
    data = load_csv_data(data_path)
    image_url_list = []
    prompt_list = []
    name_list = []
    for id, d in enumerate(data):
        name, senti_cate, senti_deg, intent, offense, meta_occur, meta_cate, target, source, target_mod, source_mod = d
        # we only need name, senti_cate and intent
        sentiment_category = senti_cate[2:-1]
        intention_detection = intent[2:-1]

        prompt_list.append(PROMPT + f'\n\nSentiment_category:{sentiment_category}\nIntention_detection:{intention_detection}\n')
        image_url_list.append(image_path+name)
        name_list.append(name)

    return image_url_list, prompt_list, name_list

def build_list_from_IIbench(data_path):
    with open(f'/mnt/afs/xueyingyi/meme/prompt_classification_emo_off_meta.txt', 'r') as file:
        PROMPT = file.read()
    image_path_list = []
    prompt_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            image_path_list.append(file_path)
            prompt_list.append(PROMPT)

    return image_path_list, prompt_list

def postpreprocess_relabel(response, name_list, path):
    for id, r in enumerate(response):
        sentiment_relabel = None
        intention_relabel = None
        try:
            result_dict = json.loads(r.text)
            if 'sentiment_category' in result_dict.keys():
                if 'author' in result_dict['sentiment_category'] or 'reader' in result_dict['sentiment_category'] or 'character' in result_dict['sentiment_category']:
                    sentiment_relabel = result_dict['sentiment_category']
            if 'intention_detection' in result_dict.keys():
                if 'author' in result_dict['intention_detection'] or 'reader' in result_dict['intention_detection'] or 'character' in result_dict['intention_detection']:
                    intention_relabel = result_dict['intention_detection']

            with open(path, 'a', newline='', encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([name_list[id], sentiment_relabel, intention_relabel])

        except:
            continue

def postpreprocess_II_bench(response, image_url_list):
    # 在csv中保存图片url和reponse
    with open('/mnt/afs/niuyazhe/data/meme/II-Bench/data/classification.csv', 'w') as file:
        writer = csv.writer(file)
        for res, url in zip(response, image_url_list):
            writer.writerow([res.text, url])


def get_result_and_save(prompt_list, image_url_list, name_list=None, path=None, pipe=None):
    prompts = [(prompt, load_image(img_url)) for prompt, img_url in zip(prompt_list, image_url_list)]
    response = pipe(prompts)
    postpreprocess_relabel(response, name_list, path)


def get_result_and_save_II_bench(prompt_list, image_url_list, name_list=None, path=None):
    model_path = '/mnt/afs/niuyazhe/data/meme/checkpoint/InternVL2-8B_en_relabel'
    pipe = pipeline(model_path, backend_config=TurbomindEngineConfig(session_len=8192))

    prompts = [(prompt, load_image(img_url)) for prompt, img_url in zip(prompt_list, image_url_list)]
    response = pipe(prompts)
    postpreprocess_II_bench(response, image_url_list)

def relabel():

    model_path = '/mnt/afs/share/InternVL25-4B'
    pipe = pipeline(model_path, backend_config=TurbomindEngineConfig(session_len=8192),torch_dtype='float16')

    train_data_path = '/mnt/afs/xueyingyi/meme/data/label_C_train.csv'
    eval_data_path = '/mnt/afs/xueyingyi/meme/data/label_C_evaluate.csv'

    save_train_path = '/mnt/afs/xueyingyi/meme/data/label_C_train_relabel.csv'
    save_test_path = '/mnt/afs/xueyingyi/meme/data/label_C_evaluate_relabel.csv'

    image_urls_list_train, prompt_list_train, name_list_train = get_list_from_file(train_data_path)
    image_urls_list_test, prompt_list_test, name_list_test = get_list_from_file(eval_data_path)

    get_result_and_save(prompt_list_train, image_urls_list_train, name_list_train, save_train_path, pipe)
    get_result_and_save(prompt_list_test, image_urls_list_test, name_list_test, save_test_path, pipe)

    count_train_1 = {'character': 0, 'reader': 0, 'author': 0}
    count_train_2 = {'character': 0, 'reader': 0, 'author': 0}
    
    count_test_1 = {'character': 0, 'reader': 0, 'author': 0}
    count_test_2 = {'character': 0, 'reader': 0, 'author': 0}
    with open(save_train_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for key in count_train_1.keys():
                if key in row[1]:
                    count_train_1[key] += 1

            for key in count_train_2.keys():
                if key in row[2]:
                    count_train_2[key] += 1

                
    with open(save_test_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for key in count_test_1.keys():
                if key in row[1]:
                    count_test_1[key] += 1

            for key in count_test_2.keys():
                if key in row[2]:
                    count_test_2[key] += 1


    print('In train data:', count_train_1, count_train_2)
    print('In test data:', count_test_1, count_test_2)


if __name__ == "__main__":
    # image_path = '/mnt/afs/niuyazhe/data/meme/II-Bench/images/dev'
    
    
    # image_path_list, prompt_list = build_list_from_IIbench(image_path)
    # get_result_and_save(prompt_list, image_path_list)
    relabel()