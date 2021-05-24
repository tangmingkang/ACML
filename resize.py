from PIL import Image

import os  
  
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg':  
                L.append(os.path.join(root, file))  
    return L  

L=file_name('/home/data/ISIC/ISIC2020/jpeg/test')
print(len(L))
for path in L:
    pic = Image.open(path)
    pic = pic.resize((1024, 1024))
    pic.save('/home/data/ISIC/ISIC2020/jpeg/test_1024/'+path.split('/')[-1])
print('over')