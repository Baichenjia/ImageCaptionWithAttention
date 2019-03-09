# -*- coding: utf-8 -*-
import pickle, os
import numpy as np
import tensorflow as tf 
tf.enable_eager_execution()
from sklearn.model_selection import train_test_split

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

# 处理captions，建立字典. 
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>", 
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
# tokenizer.word_index是一个字典，按照词频对所有单词进行编码。键为单词，值为序号. 这里仅需保留序号小于5000的值
tokenizer.word_index = {key:value for key, value in tokenizer.word_index.items() if value <= top_k}
# tokenizer.word_index[tokenizer.oov_token] = top_k + 1   # 将 'unk' 的值变为 top_k+1
tokenizer.word_index['<pad>'] = 0
index_word = {value: key for key, value in tokenizer.word_index.items()}  # 逆向字典

# 对所有caption进行编码处理. len(train_seqs)=30000, 每个元素是一个list，表示编码后的序列
train_seqs = tokenizer.texts_to_sequences(train_captions)
# 对每个句子进行padding，在后方补0. 使每个句子的长度相等，都为49
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# 训练数据：图像路径为 img_name_vector, caption为cap_vector
# 将训练数据分成训练集和验证集
img_name_train, img_name_val, cap_train, cap_val = train_test_split(
                    img_name_vector, cap_vector, test_size=0.2, random_state=0)
# 24000 24000 6000 6000
print("Total samples:", len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))

# 训练超参数
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
hidden_dim = 256
units = 512
vocab_size = len(tokenizer.word_index)
feature_shape = 2048
attention_feature_shape = 64

# 用 tf.data 导入训练数据
def map_func(img_name, cap):
    path = os.path.join("train_2014_small_inception_feature", os.path.split(img_name.decode("utf-8"))[-1]+".npy")
    img_tensor = np.load(path) 
    return img_tensor, cap

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
dataset = dataset.map(lambda item1, item2: tf.py_func(map_func,                  # 此处调用了map_func函数
                [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=4)
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(1)

# 构建模型
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features.shape = (batch,64,embedding_dim), hidden.shape = (batch,hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)   # shape=(bathc,1,hidden_size)

        # score.shape = (batch, 64, hidden_size) 
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # 规约. attention_weights.shape = (batch,64,1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # 构造context
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)   # (batch, embedding_dim)

        return context_vector, attention_weights

# Encoder
class CNN_Encoder(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(hidden_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.gru = tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')
        # 
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # 该函数只进行一步的attention和RNN过程
        # attention
        context_vector, attention_weights = self.attention(features, hidden)
        
        # x.shape=(batch, 1, emebdding_dim)
        x = self.embedding(x)
        
        # 将 context 和 单词输入合并. x.shape=(batch, 1, emebdding_dim+hidden_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # gru
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)                      # 输出为(batch, vocab_size), 用于计算softmax损失
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.train.AdamOptimizer()

def loss_function(real, pred):
    # real.shape=(batch,), pred.shape(batch,5001)
    # 为0的地方是pad的结果，此处不计算损失
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

loss_plot = []
EPOCHS = 1

for epoch in range(EPOCHS):
    total_loss = 0
    for (batch, (img_tensor, target)) in enumerate(dataset):
        loss = 0
        # img_tensor.shape=(batch, 64, 2048), target.shape=(batch,seq_len)
        # hidden.shape = (batch, units=512)
        hidden = decoder.reset_state(batch_size=target.shape[0])
        # 
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)  # (batch,1)

        with tf.GradientTape() as tape:
            # feautre.shape=(64,64,256)
            features = encoder(img_tensor)
            #
            for i in range(1, target.shape[1]):
                predictions, hidden, _ = decoder(dec_input, features, hidden)
                loss += loss_function(target[:, i], predictions)
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss += (loss / int(target.shape[1]))   # 除以序列长度

        variables = encoder.variables + decoder.variables
        gradients = tape.gradient(loss, variables) 
        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
        if batch % 1 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss.numpy() / int(target.shape[1])))
    
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / len(cap_vector))
    print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/len(cap_vector)))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# def evaluate(image):
#     attention_plot = np.zeros((max_length, attention_features_shape))


#     # 读取输入图片，经过inception特征提取
#     temp_input = tf.expand_dims(load_image(image)[0], 0)
#     img_tensor_val = image_features_extract_model(temp_input)
#     img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
#     features = encoder(img_tensor_val)

#     # Decoder的第一个输入
#     dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
#     # 初始化decoder
#     hidden = decoder.reset_state(batch_size=1)

#     result = []
#     for i in range(max_length):   # 循环
#         predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

#         attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

#         # 此处是一个采样过程
#         predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
#         result.append(index_word[predicted_id])

#         if index_word[predicted_id] == '<end>':
#             return result, attention_plot
#         # 采样的结果作为下一个时间步的输入
#         dec_input = tf.expand_dims([predicted_id], 0)

#     attention_plot = attention_plot[:len(result), :]
#     return result, attention_plot

# def plot_attention(image, result, attention_plot):
#     temp_image = np.array(Image.open(image))

#     fig = plt.figure(figsize=(10, 10))
    
#     len_result = len(result)
#     for l in range(len_result):
#         temp_att = np.resize(attention_plot[l], (8, 8))
#         ax = fig.add_subplot(len_result//2, len_result//2, l+1)
#         ax.set_title(result[l])
#         img = ax.imshow(temp_image)
#         ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

#     plt.tight_layout()
#     plt.show()














