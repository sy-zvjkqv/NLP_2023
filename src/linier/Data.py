import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import BertTokenizer

MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
batch = 2
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def Data_pre(df,x):
    index_train = int(len(df) * 0.8)
    index_val = int(len(df) * 0.1)
    index_test = int(len(df) * 0.1)
    # L = LabelEncoder()
    # df[y] = L.fit_transform(df[y])
    df[x] = df[x].astype(str)
    df[x]=df[x].astype(str)
    df[x] = df[x].str.replace(r'[【】]', ' ')       # 【】の除去
    df[x] = df[x].str.replace(r'[（）()]', ' ')     # （）の除去
    df[x] = df[x].str.replace(r'[［］\[\]]', ' ')   # ［］の除去
    df[x] = df[x].str.replace(r'[@＠]\w+', '')  # メンションの除去
    df[x]= df[x].str.replace(r'_', '')#underscodf[x].str 
    df[x] = df[x].str.replace(r'https?:\/\/.*?[\r\n ]', '')  # URLの除去
    df[x] = df[x].str.replace(r'http:\/\/.*?[\r\n ]', '')  # URLの除去
    df[x] = df[x].str.replace(r'　', ' ')  # 全角空白の除去
    df[x]=df[x].str.replace(' ','')
    df = df.sample(frac=1, random_state=0)
    df_train = df.iloc[0:index_train, 1:]
    df_val = df.iloc[index_train : index_train + index_val, 1:]
    df_test = df.iloc[
        index_train + index_val : index_train + index_val + index_test, 1:
    ]
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return df_train, df_val, df_test


def Dataloader(df, x,y_1,y_2, batch):
    sentences_text = df[x].to_list()
    list_lat = df[y_1].to_list()
    list_lon = df[y_2].to_list()
    dataset_for_loader = []
    i = 0
    for sentence in sentences_text:
        encoding = tokenizer(
            sentence,
            max_length=100,
            pad_to_max_length=True,
            truncation=True,
            # return_tensors = 'pt'
        )
        encoding['latitude'] = list_lat[i]
        encoding['longitude'] = list_lon[i]
        i = i + 1
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        dataset_for_loader.append(encoding)
    dataloader = DataLoader(dataset_for_loader, batch_size=batch)
    return dataloader