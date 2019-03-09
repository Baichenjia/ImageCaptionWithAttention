import pickle, os
import numpy as np
import tensorflow as tf 
tf.enable_eager_execution()

def load_caption():
	f_caption = open("annotations_small/train_captions.dat", 'rb')
	train_captions = pickle.load(f_caption)

	f_img_name = open("annotations_small/image_name_vector.dat", 'rb')
	img_name = pickle.load(f_img_name)

	img_name_vector = []
	for n in img_name:
		name = os.path.join("train_2014_small", n)
		assert os.path.exists(name)
		img_name_vector.append(name)

	print("total captions:", len(train_captions), ", total images:", len(img_name_vector))
	return train_captions, img_name_vector

# 调用
train_captions, img_name_vector = load_caption()

def load_image(image_path):
	# 导入图片，预处理，返回一个 tensor
	img = tf.read_file(image_path)
	img = tf.image.decode_jpeg(img, channels=3)
	img = tf.image.resize_images(img, (299, 299))
	img = tf.keras.applications.inception_v3.preprocess_input(img)  # 减去均值除以标准差
	return img, image_path

def Inception_feature():
	# 构建 inception 模型，输出为 8*8*2048 维
	image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
	model = tf.keras.Model(image_model.input, image_model.layers[-1].output)
	return model

# 特征提取模型
model = Inception_feature()

# 读入数据，提取特征
encode_train = sorted(set(img_name_vector))

# !! 这里的用法有所不同. 首先使用了map函数，map函数中返回的不仅有图像，还有图像的路径
# 返回: <BatchDataset shapes: ((?, 299, 299, 3), (?,)), types: (tf.float32, tf.string)>
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train).map(load_image).batch(16)

for img, path in image_dataset:
	batch_features = model(img)    # 提取特征, 返回值的 shape=(16, 8, 8, 2048)
	# reshape，原因是将每个图像的输出构成 64 个向量，在这些向量中进行attention, shape=(16,64,2048)
	batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
	# 构成 图像numpy、目录字符串 之间的关系
	for bf, p in zip(batch_features, path):
		# 循环处理每个图像, bf.shape=(64,2048), p是路径相关的字符串
		path_of_feature = p.numpy().decode("utf-8")
		np.save(os.path.join("train_2014_small_inception_feature", os.path.split(path_of_feature)[-1]), bf.numpy())












