import tensorflow as tf
import Lenet
import numpy as np
import os
import matplotlib.pyplot as plt


# 读取训练数据的路径-标签文件，返回路径数组，标签数组
def load_data_label(path, s: bool = False):
    """
        Create a function that read path_label file.

    :param path: file path
    :param s: to shuffle the image and label array
    :return: Image path and corresponding label
    """
    f = open(path, "r")
    lines = f.readlines()
    # [img_number, 2 : (path, label)]
    tmp = [["0", 1] for j in range(len(lines))]
    for n, i in enumerate(lines):
        tmp[n][0] = i.strip().split()[0]
        tmp[n][1] = int(i.strip().split()[1])
    # to ndarray
    L = np.array(tmp)
    if s is True:
        np.random.shuffle(L)
    f.close()
    # return img, label
    return L[:, 0], L[:, 1]


def load_and_preprocess_image(path):
    """
        Create a function that read images.
        decode ,resize, normalize.
    :param path: Absolute path
    :return: images
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image /= 255.0  # normalize to [0,1] range

    return image


def to_tfDataSet(input_img, input_label):
    """
        Create a function that make tf data set.
    :param input_img: image path array
    :param input_label: label array
    :return: images and label DataSet
    """
    # for train
    # 路径数组切片，转化为tf  Data set 格式
    path_ds = tf.data.Dataset.from_tensor_slices(input_img)
    # map函数映射，使图片路径数组 转化为可输入网络的 图片数组
    image_ds = path_ds.map(load_and_preprocess_image)
    # label数组切片并且转化为int64格式，生成用于网络输入的标签数组
    label_ds = tf.data.Dataset.from_tensor_slices(input_label.astype(np.int64))
    # 将 image data set 与 label data set 打包 形成[images, label]格式
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_label_ds


img, label = load_data_label("./train_label.txt", True)
# 数据集大小
num = len(img)
# 测试集：验证集 = 8：2
train_img = img[:int(num * 0.8)]
train_label = label[:int(num * 0.8)]

val_img = img[int(num * 0.8):]
val_label = label[int(num * 0.8):]

# 训练集数据 与 验证集数据 转化为tf输入格式
train_ds = to_tfDataSet(train_img, train_label)
val_ds = to_tfDataSet(val_img, val_label)

# 设置batch_size
BATCH_SIZE = 32
# 定义网络
model = Lenet.LeNet_model
# 对 train_ds 数组打乱
train = train_ds.shuffle(buffer_size=len(train_img))
# 生成无限数据
train = train.repeat()
# 对数据集进行划分
train = train.batch(BATCH_SIZE)

val = val_ds.shuffle(buffer_size=int(60000 * 0.2))
val = val.batch(BATCH_SIZE)

# 创建 回调文件夹 模型保存文件夹
checkpoint_path = "./training_1/cp.ckpt"
save_model_path = "./saved_model"
checkpoint_dir = os.path.dirname(checkpoint_path)
save_model_dir = os.path.dirname(save_model_path)

# 创建一个保存模型权重的回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# train
history = model.fit(train,
                    epochs=5,                       # epoch
                    steps_per_epoch=1800,           # 每个epoch 输入多少 划分后的train
                    callbacks=[cp_callback],        # callback
                    validation_data=val             # 验证集数据
                    )
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
# print(history.history.keys())

model.save('saved_model/my_model')

# 可视化
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
