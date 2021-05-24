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
```bash
chmod a+x ./scripts/train.sh
./scripts/train.sh
```
**注：kernel_type指定参数文件与日志文件名称，未设置时使用默认命名**
