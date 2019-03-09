# 该文件从原始文件中提取出训练的部分文件，由于原始文件太大，已经移动到服务器上，本地只保存了部分文件
# 该文件调用后不需要再调用
# 完整版 https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/generative_examples/image_captioning_with_attention.ipynb

import os, json, pickle
from sklearn.utils import shuffle

# 
annotation_file = 'annotations/captions_train2014.json'
assert os.path.exists(annotation_file)

with open(annotation_file, "r") as f:
	annotations = json.load(f)           # dict_keys(['info', 'images', 'licenses', 'annotations'])

# 读取全部的 caption 和对应的图片位置
all_captions = []
all_img_name_vector = []

for annot in annotations["annotations"]:
	# annot是字典，例: {'image_id': 318556, 'id': 48, 'caption': 'A very clean and well decorated empty bathroom'}
	caption = "<start> " + annot["caption"] + " <end>"
	full_coco_image_path = "COCO_train2014_" + "%012d.jpg" % (annot["image_id"])
	#
	all_img_name_vector.append(full_coco_image_path)
	all_captions.append(caption)

# 输入为两个list,经过shuffle后，两个list顺序被打乱，但是其元素之间的对应关系不变
train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

# 截取 30000 条数据用作训练
num_examples = 30000
train_captions = train_captions[:num_examples]
image_name_vector = img_name_vector[:num_examples]

# 将选定的 captions 写入文件
f = open('train_captions.dat','wb')
pickle.dump(train_captions, f)

# 将选定的 image 写入文件
f1 = open('image_name_vector.dat','wb')
pickle.dump(image_name_vector, f1)

# 将选定的 30000 条数据移动到另外一个文件夹下
import shutil
for img in image_name_vector:
	newpath = os.path.join("train_2014_small", img)
	shutil.copyfile(os.path.join("train2014", img), newpath)
