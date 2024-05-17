## preprocess data
import pandas as pd
#pip install openpyxl
from tqdm import tqdm
import json
import argparse
import os

### include only the data is full
def preprocess(data, cols):
    for i in cols:
        data = data[-data[i].isnull()]
    return data
def get_single_res(data_fillna,idx):
    #输入: 人工草稿的描述, 标题, 商品图片(1张)
    #输出: 标题+描述
    inputs = data_fillna.iloc[idx,:]
    res = {}
    res["id"] = str(idx)
    res["image"] = inputs["ImagePaths"].split('\n')[0]
    res["conversations"] = [{'from':'human',
                             'value':'<image>\nHelp me rephrase the title and listing information for the above picture, the initial title from human is {}, the initial product description from human is {}'.format(inputs["OriginTitle"],inputs["OriginDescribe"])},
                           {'from':'gpt',
                            'value':'title is {}, description is {}'.format(inputs["Title"],inputs["itemDescription"])}]
    return res

def main(data_path,output_folder):
    data = pd.read_excel(data_path)
    print ("<<< load excel data!")
    data_fillna = preprocess(data, ['ImagePaths', 'OriginDescribe', 'OriginTitle', 'Title', 'itemDescription'])
    res_json = []
    #for i in tqdm(range(len(data_fillna))):
    for i in tqdm(range(60)):
        res = get_single_res(data_fillna,i)
        res_json.append(res)

    #output json
    # Open a file for writing (text mode with UTF-8 encoding)
    with open(os.path.join(output_folder, 'data.json'), 'w', encoding='utf-8') as f:
        # Use json.dump() to write the list to the file
        json.dump(res_json, f)  # Optional parameter for indentation
    print('Data written to json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_folder', type=str)
    args = parser.parse_args()
    main(args.data_path, args.output_folder)

