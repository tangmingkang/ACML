import os
import pandas as pd
import random
import numpy as np
import tqdm
import pdb

# 读取数据 添加filepath 和 fold
data_dir_fake='../melanoma/'
data_dict=dict()
data_dict['image_name']=['fake_melanoma_'+str(i+1) for i in range(1000)]
data_dict['fold']=[-1 for i in range(len(data_dict['image_name']))]
data_dict['hair']=[0 for i in range(len(data_dict['image_name']))]
data_dict['target']=[4 for i in range(len(data_dict['image_name']))]
data_dict['sex']=[-1.0 for i in range(len(data_dict['image_name']))]
data_dict['age_approx']=[0.0 for i in range(len(data_dict['image_name']))]
data_dict['n_images']=[np.log1p(1) for i in range(len(data_dict['image_name']))]
for key in ['site_head/neck','site_lower extremity','site_oral/genital','site_palms/soles','site_torso','site_upper extremity']:
    data_dict[key]=[0 for i in range(len(data_dict['image_name']))]
data_dict['site_nan']=[1 for i in range(len(data_dict['image_name']))]
df_fake=pd.DataFrame(data_dict)
df_fake['filepath'] = df_fake['image_name'].apply(lambda x: os.path.join(data_dir_fake,  f'{x}.jpg'))
# 生成meta数据
def get_meta_data(df_fake):
    train_images = df_fake['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(train_images):
        train_sizes[i] = os.path.getsize(img_path)
    
    df_fake['image_size'] = np.log(train_sizes)
    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_fake.columns if col.startswith('site_')] # n_images:同一个病人的图片数量  imagesize：图片大小
    n_meta_features = len(meta_features)

    return df_fake, meta_features, n_meta_features

df_fake, meta_features, n_meta_features = get_meta_data(df_fake)

df_fake_columns=['image_name', 'fold','hair','target' ,'sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_fake.columns if col.startswith('site_')] # n_images:同一个病人的图片数量  imagesize：图片大小
df_fake[df_fake_columns].to_csv('fake.csv',sep=',',index=False,header=True)
