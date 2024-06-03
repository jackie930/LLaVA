## preprocess data
import pandas as pd
#pip install openpyxl
from tqdm import tqdm
import json
import argparse
import os
from PIL import Image

### include only the data is full
def preprocess(data, cols):
    for i in cols:
        data = data[-data[i].isnull()]
    return data

#check if data exists
def check_data(data_root_path, imge_path):
    full_path = os.path.join(data_root_path,imge_path)
    if os.path.isfile(full_path):
        return True
    else:
        return False

#TODO: Add more training data generation methods
def get_single_res(data_root_path,data_fillna,idx):
    #输入: 人工草稿的描述, 标题, 商品图片(1张)
    #输出: 标题+描述
    inputs = data_fillna.iloc[idx,:]
    res = {}
    image_path = inputs["ImagePaths"].split('\n')[0]
    if image_path.startswith('http'):
        image_path = image_path.replace("http://img.yafex.cn","img")
    try:
        #try to open imae file to see if it exists & valid
        image = Image.open(os.path.join(data_root_path, image_path))

        res["id"] = str(idx)
        res['image'] = image_path
        if len(inputs["OriginDescribe"])<10:
            res["conversations"] = [{'from': 'human',
                                     'value': '<image>\nHelp me rephrase the title and listing information for the above picture. \nThe draft title from human is [{}]. What"s the optimized title? '.format(
                                         inputs["OriginTitle"])},
                                    {'from': 'gpt',
                                     'value': 'Rephrased title is [{}].'.format(inputs["Title"])}]
        else:
            res["conversations"] = [{'from': 'human',
                                     'value': '<image>\nHelp me rephrase the title and listing information for the above picture. \nThe draft title from human is [{}]. \n The draft listing from human is [{}]. What"s the optimized title and listing? '.format(
                                         inputs["OriginTitle"], inputs["OriginDescribe"])},
                                    {'from': 'gpt',
                                     'value': 'Rephrased title is [{}] \n Rephrased listing is [{}] '.format(
                                         inputs["Title"], inputs["itemDescription"])}]

    except:
        res = {}

    return res

def main(data_path,output_folder):
    data = pd.read_excel(data_path)
    print ("<<< load excel data!")
    data_root_path = os.path.dirname(data_path)
    data_fillna = preprocess(data, ['ImagePaths', 'OriginDescribe', 'OriginTitle', 'Title', 'itemDescription'])
    res_json = []
    for i in tqdm(range(len(data_fillna))):
    #for i in tqdm(range(60)):
        res = get_single_res(data_root_path,data_fillna,i)
        if res!={}:
            res_json.append(res)

    #output json, train/test split
    total_len = len(res_json)
    split_idx = int(total_len*0.9)
    train_json = res_json[:split_idx]
    test_json = res_json[split_idx:]

    # Open a file for writing (text mode with UTF-8 encoding)
    with open(os.path.join(output_folder, 'data.json'), 'w', encoding='utf-8') as f:
        # Use json.dump() to write the list to the file
        json.dump(train_json, f)  # Optional parameter for indentation
    print('Data written to json')

    with open(os.path.join(output_folder, 'test.json'), 'w', encoding='utf-8') as f:
        # Use json.dump() to write the list to the file
        json.dump(test_json, f)  # Optional parameter for indentation
    print('Data written to json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_folder', type=str)
    args = parser.parse_args()
    main(args.data_path, args.output_folder)

