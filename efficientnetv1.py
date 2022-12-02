import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import math

"""
https://blog.csdn.net/dgvv4/article/details/123553351

2.1 深度可分离卷积
MobileNetV1 中主要使用了深度可分离卷积模块，大大减少了参数量和计算量。

普通卷积是一个卷积核处理所有的通道，输入特征图有多少个通道，卷积核就有几个通道，一个卷积核生成一张特征图。

深度可分离卷积 可理解为 深度卷积 + 逐点卷积

深度卷积只处理长宽方向的空间信息；逐点卷积只处理跨通道方向的信息。能大大减少参数量，提高计算效率

深度卷积： 一个卷积核只处理一个通道，即每个卷积核只处理自己对应的通道。输入特征图有多少个通道就有多少个卷积核。将每个卷积核处理后的特征图堆叠在一起。输入和输出特征图的通道数相同。

由于只处理长宽方向的信息会导致丢失跨通道信息，为了将跨通道的信息补充回来，需要进行逐点卷积。

逐点卷积： 是使用1x1卷积对跨通道维度处理，有多少个1x1卷积核就会生成多少个特征图。



2.2 逆转残差模块
逆转残差模块流程如下。输入图像，先使用1x1卷积提升通道数；然后在高维空间下使用深度卷积；再使用1x1卷积下降通道数，降维时采用线性激活函数(y=x)。当步长等于1且输入和输出特征图的shape相同时，使用残差连接输入和输出；当步长=2（下采样阶段）直接输出降维后的特征图。

对比 ResNet 的残差结构。输入图像，先使用1x1卷积下降通道数；然后在低维空间下使用标准卷积，再使用1x1卷积上升通道数，激活函数都是ReLU函数。当步长等于1且输入和输出特征图的shape相同时，使用残差连接输入和输出；当步长=2（下采样阶段）直接输出降维后的特征图。



2.3 SE注意力机制
（1）先将特征图进行全局平均池化，特征图有多少个通道，那么池化结果（一维向量）就有多少个元素，[h, w, c]==>[None, c]。

（2）然后经过两个全连接层得到输出向量。在EfficientNet中，第一个全连接层降维，输出通道数等于该逆转残差模块的输入图像的通道数的1/4；第二个全连接层升维，输出通道数等于全局平均池化前的特征图的通道数。

（3）全连接层的输出向量可理解为，向量的每个元素是对每张特征图进行分析得出的权重关系。比较重要的特征图就会赋予更大的权重，即该特征图对应的向量元素的值较大。反之，不太重要的特征图对应的权重值较小。

（4）经过两个全连接层得到一个由channel个元素组成的向量，每个元素是针对每个通道的权重，将权重和原特征图的像素值对应相乘，得到新的特征图数据

以下图为例，特征图经过两个全连接层之后，比较重要的特征图对应的向量元素的值就较大。将得到的权重和对应特征图中的所有元素相乘，得到新的输出特征图


"""


# swish激活函数
def swish(x):
    return x * tf.nn.sigmoid(x)


def conv_block(input_tensor, filters, kernel_size, strides, activation=True):
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False # 有BN层就不要偏置
    )(input_tensor)

    x = layers.BatchNormalization()(x)

    if activation:
        x = swish(x)

    return x


# squeeze_excitation SE注意力机制
# 为了减少计算量，SE注意力机制中的全连接层可以换成1*1卷积层。这里要注意，第一个卷积层降维的通道数，
# 是MBConv模块的输入特征图通道数的1/4，也就是在逆转残差模块中1*1卷积升维之前的特征图通道数的1/4
def se_block(input_tensor, inputs_channel):
    # 通道数下降为输入该MBConv的特征图的1/4
    squeeze = inputs_channel // 4
    # 通道数上升为深度卷积的输出特征图个数
    excitation = input_tensor.shape[-1]

    # 全局平均池化 [h,w,c]==>[None,c]
    x = layers.GlobalAveragePooling2D()(input_tensor)
    # [None,c]==>[1,1,c]
    x = layers.Reshape(target_shape=[1, 1, x.shape[-1]])(x)

    # 1*1卷积降维，通道数变为输入MBblock模块的图像的通道数的1/4
    x = layers.Conv2D(
        filters=squeeze,
        kernel_size=[1, 1],
        strides=1,
        padding='same')(x)

    x = swish(x)

    # 1*1卷积升维，通道数变为深度卷积的输出特征图个数
    x = layers.Conv2D(
        filters=excitation,
        kernel_size=[1, 1],
        strides=1,
        padding='same')(x)

    x = tf.nn.sigmoid(x)

    # 将深度卷积的输入特征图的每个通道和SE得到的针对每个通道的权重相乘
    x = layers.multiply([input_tensor, x])

    return x


# 逆转残差模块
# 以基本模块为例(stride=1)。如果需要提升特征图的通道数，那么先经过1x1卷积上升通道数；
# 然后在高纬空间下使用深度卷积；再经过SE注意力机制优化特征图数据；
# 再经过1x1卷积下降通道数（使用线性激活函数，y=x）；
# 若此时输入特征图的shape和输出特征图的shape相同，那么对1x1卷积降维后的特征图加一个Dropout层，防止过拟合；
# 最后残差连接输入和输出。
def MBConv(x, expansion, out_channel, kernel_size, strides, drop_rate):
    """
    expansion代表第一个1*1卷积上升的通道数是输入图像通道数的expansion倍
    out_channel代表MBConv模块输出通道数个数，即第二个1*1卷积的卷积核个数
    dropout_rate代表dropout层随机杀死神经元的概率
    """

    # 残差边
    residual = x

    # 输入的特征图的通道数
    input_channel = x.shape[-1]

    # 1 若expansion==1，1*1卷积升维就不用执行
    if expansion != 1:
        x = conv_block(
            input_tensor=x,
            filters=input_channel*expansion,
            kernel_size=[1, 1],
            strides=1,
            activation=True
        )

    # 2 深度卷积
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False # 有BN层就不用偏置
    )(x)

    x = layers.BatchNormalization()(x)
    x = swish(x)

    # 3 SE注意力机制，传入深度卷积输出的tensor，和输入至MBConv模块的特征图通道数
    x = se_block(input_tensor=x, inputs_channel=input_channel)

    # 4 1*1卷积下降通道数，使用线性激活，即卷积+BN
    x = conv_block(
        input_tensor=x,
        filters=out_channel, # 1*1卷积输出通道数就是MBConv模块输出通道数
        kernel_size=[1, 1],
        strides=1,
        activation=False
    )

    # 5. 只有使用残差连接，并且dropout_rate>0时才会使用Dropout层
    if strides == 1 and residual.shape == x.shape:
        if drop_rate > 0:
            x = layers.Dropout(rate=drop_rate)(x)

        # 残差连接输入和输出
        x = layers.Add()([residual, x])
        return x

    # 如果步长=2，直接输出1*1降维的结果
    return x


# 一个stage模块是由多个MBConv模块组成
def stage(x, n, expansion, out_channel, kernel_size, strides, drop_rate):
    for _ in range(n):
        # x, expansion, out_channel, kernel_size, strides, drop_rate
        x = MBConv(x, expansion, out_channel, kernel_size, strides, drop_rate)

    return x


# 通道数乘维度因子后，取8的倍数
def round_filters(filters, width_coefficient, divisor=8):
    # 通道数乘宽度因子
    filters = filters * width_coefficient

    # 新的通道数是距离远通道数最近的8的倍数
    new_filters = max(divisor, int(filters + divisor/2) // divisor * divisor)

    if new_filters <= 0.9 * filters:
        new_filters += filters

    return new_filters


# 深度乘上深度因子后，向上取整
def round_repeat(repeats, depth_coefficient):
    # 求得每一个卷积模块重复执行的次数
    # 向上取整后小数部分=0，int()舍弃小数部分
    repeats = int(math.ceil(repeats * depth_coefficient))
    return repeats


# 主干模型结构
def efficientnet(input_shape, classes, width_coefficient, depth_coefficient, drop_rate):
    """
    width_coefficient，通道维度上的倍率因子。与卷积核个数相乘，取整到离它最近的8的倍数
    depth_coefficient，深度维度上的倍率因子。和模块重复次数相乘，向上取整
    dropout_rate，dropout层杀死神经元的概率
    """

    inputs = layers.Input(shape=input_shape)

    # 标准卷积[224, 224, 3] == > [112, 112, 32]
    # 维度因子改变卷积核个数
    x = conv_block(
        input_tensor=inputs,
        filters=round_filters(32, width_coefficient), # 维度因子改变卷积核个数
        kernel_size=[3, 3],
        strides=2
    )

    # [112,112,32]==>[112,112,16]
    x = stage(
        x,
        n=round_repeat(1, depth_coefficient),
        expansion=1,
        out_channel=round_filters(16, width_coefficient),
        kernel_size=[3, 3],
        strides=1,
        drop_rate=drop_rate
    )

    # [112,112,16]==>[56,56,24]
    x = stage(
        x,
        n=round_repeat(2, depth_coefficient),
        expansion=6,
        out_channel=round_filters(24, width_coefficient),
        kernel_size=[3, 3],
        strides=2,
        drop_rate=drop_rate
    )

    # [56,56,24]==>[28,28,40]
    x = stage(
        x,
        n=round_repeat(2, depth_coefficient),
        expansion=6,
        out_channel=round_filters(40, width_coefficient),
        kernel_size=[5, 5],
        strides=2,
        drop_rate=drop_rate
    )

    # [28,28,40]==>[14,14,80]
    x = stage(
        x,
        n=round_repeat(3, depth_coefficient),
        expansion=6,
        out_channel=round_filters(80, width_coefficient),
        kernel_size=[3, 3],
        strides=2,
        drop_rate=drop_rate
    )

    # [14,14,80]==>[14,14,112]
    x = stage(
        x,
        n=round_repeat(3, depth_coefficient),
        expansion=6,
        out_channel=round_filters(112, width_coefficient),
        kernel_size=[5, 5],
        strides=1,
        drop_rate=drop_rate
    )

    # [14,14,112]==>[7,7,192]
    x = stage(
        x,
        n=round_repeat(4, depth_coefficient),
        expansion=6,
        out_channel=round_filters(192, width_coefficient),
        kernel_size=[5, 5],
        strides=2,
        drop_rate=drop_rate
    )

    # [7,7,192]==>[7,7,320]
    x = stage(
        x,
        n=round_repeat(1, depth_coefficient),
        expansion=6,
        out_channel=round_filters(320, width_coefficient),
        kernel_size=[3, 3],
        strides=1,
        drop_rate=drop_rate
    )

    # [7,7,320]==>[7,7,1280]
    x = layers.Conv2D(
        filters=1280,
        kernel_size=[1, 1],
        strides=1,
        padding='same',
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = swish(x)

    # [7,7,1280]==>[None,1280]
    x = layers.GlobalAveragePooling2D()(x)

    # [None,1280]==>[None,1000]
    x = layers.Dropout(rate=drop_rate)(x)
    x = layers.Dense(classes)(x)

    model = models.Model(inputs=inputs, outputs=x)
    model.summary()

    return model


if __name__ == '__main__':
    efficientnet = efficientnet(
        input_shape=[224, 224, 3],
        classes=1000,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        drop_rate=0.2
    )
