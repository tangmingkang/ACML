# Retinal Image Classfication
### 使用教程
clone源码并安装依赖包，所有图像放到datasets/image路径下，使用generate_csv.py可以生成相应的标签文件label.csv
```bash
git clone https://github.com/tangmingkang/ACML.git
cd ACML
pip install -r requirements.txt
```
### 运行说明
数据
```bash
python resize.py # 将数据resize到1024*1024 加快计算速度 非必须（自行修改地址）
cd datasets
ln -s /home/data/ISIC images # 设置软链接 非必须
```
训练模型(注意修改数据路径,运行时DEBUG设置为false)
5.26 发现一个bug，会导致多卡训练时fold0训练结束后，在fold1卡住，暂时未解决，设置train-fold为单个fold或使用main.py避免train.py中训练多个fold
```bash
# chmod a+x ./scripts/train.sh
# ./scripts/train.sh
nohup python main.py >main.log 2>&1 & # 注意保证lock.txt 运行之前开锁（为0）
```
**注：kernel_type指定参数文件与日志文件名称，未设置时使用默认命名**
**注：单机多卡代码必须使用torch.distributed.launch启动，nproc_per_node指定GPU数量**
