import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model('saved_model/my_model')

new_model.summary()


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


img, label = load_data_label("./test_label.txt", True)
num = len(img)
print(num)
test_ds = to_tfDataSet(img, label)

test_ds = test_ds.shuffle(buffer_size=num)
test_ds = test_ds.batch(32)

loss, acc = new_model.evaluate(test_ds)
print(loss, acc)

