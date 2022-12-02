import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import math


# swish激活函数
def swish(x):
    return x * tf.nn.sigmoid(x)


# 标准卷积块
def conv_block(x, filters, kernel_size, strides, activation=True):
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    if activation:
        x = swish(x)

    return x


# SE注意力机制
def se_block(in_tensor, in_channel, ratio=0.25):
    '''
    in_tensor: 深度卷积层的输出特征图
    in_channel: MBConv模块的输入特征图的通道数
    ratio: 第一个全连接层的通道数下降为MBConv输入特征图的几倍
    '''

    # 第一个FC降低通道数个数
    squeeze = max(1, int(in_channel * ratio))
    # 第二个FC上升通道数个数
    excitation = in_tensor.shape[-1]

    # 全局平均池化[h, w, c] == > [None, c]
    x = layers.GlobalAveragePooling2D()(in_tensor)
    # [None,c]==>[1,1,c]
    x = layers.Reshape(target_shape=[1, 1, x.shape[-1]])(x)

    # [1,1,c]==>[1,1,c/4]
    x = layers.Conv2D(
        filters=squeeze, # 通道数下降1/4
        kernel_size=[1, 1],
        strides=1,
        padding='same'
    )(x)

    x = swish(x)

    # [1,1,c/4]==>[1,1,c]
    x = layers.Conv2D(
        filters=excitation, # 通道数上升至原来
        kernel_size=[1, 1],
        strides=1,
        padding='same'
    )(x)

    # sigmoid激活，权重归一化
    x = tf.nn.sigmoid(x)

    # [h,w,c] * [1,1,c] ==> [h,w,c]
    x = layers.multiply([x, in_tensor])
    return x


# 逆转残差模块
def MBConv(x, expansion, out_channel, kernel_size, strides, drop_rate):
    '''
    expansion: 第一个卷积层特征图通道数上升的倍数
    kernel_size: 深度卷积层的卷积核size
    stride: 深度卷积层的步长
    out_channel: 第二个卷积层下降的通道数
    dropout_rate: Dropout层随机丢弃输出层的概率，直接将输入接到输出
    '''

    # 残差边
    residual = x
    # 输入特征图的通道数
    in_channel = x.shape[-1]

    if expansion != 1:
        # 1, 1*1标准卷积升维
        x = conv_block(
            x=x,
            filters=in_channel*expansion, # 上升通道数为expansion倍
            kernel_size=[1, 1],
            strides=1,
            activation=True
        )

    # 2, 3*3深度卷积
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = swish(x)

    # SE注意力机制，输入特征图x，和MBConv模块输入图像的通道数
    x = se_block(x, in_channel)

    # 1*1标准卷积降维，使用线性激活
    x = conv_block(
        x=x,
        filters=out_channel,
        kernel_size=[1, 1],
        strides=1,
        activation=False
    )

    # 只有步长=1且输入等于输出shape，才使用残差连接输入和输出
    if strides == 1 and x.shape == residual.shape:
        if drop_rate > 0:
            # 参数noise_shape一定的概率将某一层的输出丢弃
            x = layers.Dropout(rate=drop_rate, noise_shape=[None, 1, 1, 1])(x)

        x = layers.Add()([x, residual])

    return x


# Fused-MBConv模块
def Fused_MBConv(x, expansion, out_channel, kernel_size, strides, drop_rate):
    # 残差边
    residual = x
    # 输入特征图的通道数
    in_channel = x.shape[-1]

    # 如果通道扩展倍数expansion==1，就不需要升维
    if expansion != 1:
        # 3*3标准卷积升维
        x = conv_block(
            x=x,
            filters=in_channel*expansion, # 通道数上升为原来的expansion倍
            kernel_size=kernel_size,
            strides=strides,
            activation=True
        )

    #
    #     x = conv_block(
    #         x=x,
    #         filters=out_channel,
    #         kernel_size=[1, 1],
    #         strides=1,
    #         activation=False
    #     )
    # else:
    #
    #     x = conv_block(
    #         x=x,
    #         filters=out_channel,
    #         kernel_size=kernel_size,
    #         strides=strides,
    #         activation=True
    #     )

    se_block(x, in_channel)

    # expansion!=1, 变成1*1卷积+BN，步长为1
    # expansion==1，变成3*3卷积+BN+激活
    x = conv_block(
        x=x,
        filters=out_channel,
        kernel_size=[1, 1] if expansion != 1 else kernel_size,
        strides=1 if expansion != 1 else strides,
        activation=False if expansion != 1 else True
    )

    # 当步长=1且输入输出shape相同时残差连接
    if strides == 1 and x.shape == residual.shape:
        if drop_rate > 0:
            x = layers.Dropout(
                rate=drop_rate,
                # 代表不是杀死神经元，是丢弃输出层
                noise_shape=[None, 1, 1, 1]
            )(x)

        x = layers.Add()([x, residual])

    return x


def Fused_stage(x, n, expansion, out_channel, kernel_size, strides, drop_rate):
    for _ in range(n):
        x = Fused_MBConv(x, expansion, out_channel, kernel_size, strides, drop_rate)

    return x


def stage(x, n, expansion, out_channel, kernel_size, strides, drop_rate):
    for _ in range(n):
        x = MBConv(x, expansion, out_channel, kernel_size, strides, drop_rate)

    return x


def round_filters(filters, width_coefficient, depth_divisor=8):
    filters = filters * width_coefficient
    new_filters = max(depth_divisor, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += filters

    return new_filters


def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(repeats * depth_coefficient))


def efficientnetv2(in_shape, classes, drop_rate, width_coefficient, depth_coefficient):
    inputs = layers.Input(shape=in_shape)

    # 标准卷积层[224,224,3]==>[112,112,24]
    x = conv_block(
        x=inputs,
        filters=round_filters(24, width_coefficient),
        kernel_size=[3, 3],
        strides=2,
        activation=True
    )

    # [112,112,24]==>[112,112,24]
    x = Fused_stage(
        x=x,
        n=round_repeats(2, depth_coefficient),
        expansion=1,
        out_channel=round_filters(24, width_coefficient),
        kernel_size=[3, 3],
        strides=1,
        drop_rate=drop_rate
    )

    # [112,112,24]==>[56,56,48]
    x = Fused_stage(
        x=x,
        n=round_repeats(4, depth_coefficient),
        expansion=4,
        out_channel=round_filters(48, width_coefficient),
        kernel_size=[3, 3],
        strides=2,
        drop_rate=drop_rate
    )

    # [56,56,48]==>[32,32,64]
    x = Fused_stage(
        x=x,
        n=round_repeats(4, depth_coefficient),
        expansion=4,
        out_channel=round_filters(64, width_coefficient),
        kernel_size=[3, 3],
        strides=2,
        drop_rate=drop_rate
    )

    # [32,32,64]==>[16,16,128]
    x = stage(
        x=x,
        n=round_repeats(6, depth_coefficient),
        expansion=4,
        out_channel=round_filters(128, width_coefficient),
        kernel_size=[3, 3],
        strides=2,
        drop_rate=drop_rate
    )

    # [16,16,128]==>[16,16,160]
    x = stage(
        x=x,
        n=round_repeats(9, depth_coefficient),
        expansion=6,
        out_channel=round_filters(160, width_coefficient),
        kernel_size=[3, 3],
        strides=1,
        drop_rate=drop_rate
    )

    # [16,16,160]==>[8,8,256]
    x = stage(
        x=x,
        n=round_repeats(15, depth_coefficient),
        expansion=6,
        out_channel=round_filters(256, width_coefficient),
        kernel_size=[3, 3],
        strides=2,
        drop_rate=drop_rate
    )

    # [8,8,256]==>[8,8,1280]
    x = conv_block(
        x=x,
        filters=1280,
        kernel_size=[1, 1],
        strides=1,
        activation=True
    )

    # [8,8,1280]==>[None,1280]
    x = layers.GlobalAveragePooling2D()(x)

    if drop_rate > 0:
        x = layers.Dropout(rate=drop_rate)(x)

    # [None,1280]==>[None,classes]
    x = layers.Dense(classes)(x)

    model = models.Model(inputs=inputs, outputs=x)
    model.summary()

    return model


if __name__ == '__main__':
    model = efficientnetv2([224, 224, 3], 1000, 0, 1.0, 1.0)
