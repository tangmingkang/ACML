import os
import pandas as pd
import random
import numpy as np
import tqdm
import pdb

# 读取数据 添加filepath 和 fold
data_dir_2020='images/ISIC2020/'
df_train = pd.read_csv(os.path.join(data_dir_2020, 'train.csv'))
df_test = pd.read_csv(os.path.join(data_dir_2020, 'test.csv'))
df_hair = pd.read_csv(os.path.join('./hair.csv'))
train_fold = pd.read_csv(os.path.join('./fold.csv'))
df_train['hair']=df_hair['label'].fillna(-1).astype(int)
hair_map={-1:0,0:0,1:1,2:2,3:3,4:4}
df_train['hair']=df_train['hair'].map(hair_map)
# fold = []
# for index, row in df_train.iterrows():
#     fold.append(random.randint(0,19))
# df_train['fold']=fold
df_train['fold']=train_fold['fold']
df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir_2020, 'jpeg/train_1024', f'{x}.jpg'))
df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir_2020, 'jpeg/test_1024', f'{x}.jpg'))

# 生成meta数据
def get_meta_data(df_train, df_test):
    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat([df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site') # 获取site one hot编码  dummy_na=True：空值单独编码
    df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)
    # Sex features
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)
    # Age features
    df_train['age_approx'] /= 90
    df_test['age_approx'] /= 90
    df_train['age_approx'] = df_train['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)
    df_train['patient_id'] = df_train['patient_id'].fillna(0)
    # n_image per user
    df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
    df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train.loc[df_train['patient_id'] == 0, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)
    # image size
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(train_images):
        train_sizes[i] = os.path.getsize(img_path)
    
    df_train['image_size'] = np.log(train_sizes)

    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(test_images):
        test_sizes[i] = os.path.getsize(img_path)
    df_test['image_size'] = np.log(test_sizes)
    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')] # n_images:同一个病人的图片数量  imagesize：图片大小
    n_meta_features = len(meta_features)

    return df_train, df_test, meta_features, n_meta_features

df_train, df_test, meta_features, n_meta_features = get_meta_data(df_train, df_test)

# class mapping
diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
'''
{'atypical melanocytic...liferation': 0, 'cafe-au-lait macule': 1, 'lentigo NOS': 2, 'lichenoid keratosis': 3, 'melanoma': 4, 'nevus': 5, 'seborrheic keratosis': 6, 'solar lentigo': 7, 'unknown': 8}
'''
# 二分类
# for key in diagnosis2idx.keys():
#     if key=='melanoma':
#         diagnosis2idx[key]=1
#     else:
#         diagnosis2idx[key]=0
print(diagnosis2idx)
df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)
train_save_columns=['image_name', 'fold','hair','target' ,'sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')] # n_images:同一个病人的图片数量  imagesize：图片大小
test_save_columns=['image_name','sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')] # n_images:同一个病人的图片数量  imagesize：图片大小
df_train[train_save_columns].to_csv('train.csv',sep=',',index=False,header=True)
df_test[test_save_columns].to_csv('test.csv',sep=',',index=False,header=True)