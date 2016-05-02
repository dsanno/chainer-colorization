import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L

class Colorization(chainer.Chain):
    wscale=1
    def __init__(self):
        super(Colorization, self).__init__(
            class1=L.Linear(7 * 7 * 512, 1024, wscale=self.wscale),
            class_bn1=L.BatchNormalization(1024),
            class2=L.Linear(1024, 512, wscale=self.wscale),
            class_bn2=L.BatchNormalization(512),
            class3=L.Linear(512, 256, wscale=self.wscale),
            class_bn3=L.BatchNormalization(256),
            feature1=L.Convolution2D(256, 512, 3, stride=1, pad=1, wscale=self.wscale),
            feature_bn1=L.BatchNormalization(512),
            feature2=L.Convolution2D(512, 256, 3, stride=1, pad=1, wscale=self.wscale),
            feature_bn2=L.BatchNormalization(256),
            fusion=L.Convolution2D(512, 256, 1, stride=1, wscale=self.wscale),
            fusion_bn=L.BatchNormalization(256),
            color1=L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=self.wscale),
            color_bn1=L.BatchNormalization(128),
            color2=L.Convolution2D(128, 64, 3, stride=1, pad=1, wscale=self.wscale),
            color_bn2=L.BatchNormalization(64),
            color3=L.Deconvolution2D(64, 64, 4, stride=2, pad=1, wscale=self.wscale),
            color_bn3=L.BatchNormalization(64),
            color4=L.Convolution2D(64, 32, 3, stride=1, pad=1, wscale=self.wscale),
            color_bn4=L.BatchNormalization(32),
            color5=L.Convolution2D(32, 2, 3, stride=1, pad=1, wscale=self.wscale),
        )
    def __call__(self, x1, x2, train=True):
        h1_1 = F.leaky_relu(self.class_bn1(self.class1(x1), test=not train))
        h1_2 = F.leaky_relu(self.class_bn2(self.class2(h1_1), test=not train))
#        h1_1 = F.leaky_relu(self.class1(x1))
#        h1_2 = F.leaky_relu(self.class2(h1_1))
        h1_3 = F.leaky_relu(self.class_bn3(self.class3(h1_2), test=not train))
        h1_4 = F.reshape(h1_3, h1_3.data.shape + (1, 1))
        h2_1 = F.leaky_relu(self.feature_bn1(self.feature1(x2), test=not train))
#        h2_1 = F.leaky_relu(self.feature1(x2))
        h2_2 = F.leaky_relu(self.feature_bn2(self.feature2(h2_1), test=not train))
        h3_1 = F.concat((h2_2, (F.broadcast_to(h1_4, h2_2.data.shape))), axis=1)
        h3_2 = F.leaky_relu(self.fusion_bn(self.fusion(h3_1), test=not train))
        h4_1 = F.leaky_relu(self.color_bn1(self.color1(h3_2), test=not train))
#        h4_1 = F.leaky_relu(self.color1(h3_2))
        h4_2 = F.leaky_relu(self.color_bn2(self.color2(h4_1), test=not train))
        h4_3 = F.leaky_relu(self.color_bn3(self.color3(h4_2), test=not train))
#        h4_3 = F.leaky_relu(self.color3(h4_2))
        h4_4 = F.leaky_relu(self.color_bn4(self.color4(h4_3), test=not train))
        return F.tanh(self.color5(h4_4))

class Colorization2(chainer.Chain):
    wscale=1
    def __init__(self):
        super(Colorization2, self).__init__(
            conv1=L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=self.wscale),
            bn1=L.BatchNormalization(64),
            conv2=L.Convolution2D(64, 128, 3, stride=1, pad=1, wscale=self.wscale),
            bn2=L.BatchNormalization(128),
            conv3=L.Convolution2D(128, 128, 4, stride=2, pad=1, wscale=self.wscale),
            bn3=L.BatchNormalization(128),
            conv4=L.Convolution2D(128, 256, 3, stride=1, pad=1, wscale=self.wscale),
            bn4=L.BatchNormalization(256),
            conv5=L.Convolution2D(256, 256, 4, stride=2, pad=1, wscale=self.wscale),
            bn5=L.BatchNormalization(256),
            conv6=L.Convolution2D(256, 512, 3, stride=1, pad=1, wscale=self.wscale),
            bn6=L.BatchNormalization(512),
            conv7=L.Convolution2D(512, 512, 4, stride=2, pad=1, wscale=self.wscale),
            bn7=L.BatchNormalization(512),
            conv8=L.Convolution2D(512, 512, 3, stride=1, pad=1, wscale=self.wscale),
            bn8=L.BatchNormalization(512),
            conv9=L.Convolution2D(512, 512, 4, stride=2, pad=1, wscale=self.wscale),
            bn9=L.BatchNormalization(512),
            conv10=L.Convolution2D(512, 512, 3, stride=1, pad=1, wscale=self.wscale),
            bn10=L.BatchNormalization(512),

            class1=L.Linear(7 * 7 * 512, 1024, wscale=self.wscale),
            class_bn1=L.BatchNormalization(1024),
            class2=L.Linear(1024, 512, wscale=self.wscale),
            class_bn2=L.BatchNormalization(512),
            class3=L.Linear(512, 256, wscale=self.wscale),
            class_bn3=L.BatchNormalization(256),
            feature1=L.Convolution2D(512, 512, 3, stride=1, pad=1, wscale=self.wscale),
            feature_bn1=L.BatchNormalization(512),
            feature2=L.Convolution2D(512, 256, 3, stride=1, pad=1, wscale=self.wscale),
            feature_bn2=L.BatchNormalization(256),
            fusion=L.Convolution2D(512, 256, 1, stride=1, wscale=self.wscale),
            fusion_bn=L.BatchNormalization(256),
            color1=L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=self.wscale),
            color_bn1=L.BatchNormalization(128),
            color2=L.Convolution2D(128, 64, 3, stride=1, pad=1, wscale=self.wscale),
            color_bn2=L.BatchNormalization(64),
            color3=L.Deconvolution2D(64, 64, 4, stride=2, pad=1, wscale=self.wscale),
            color_bn3=L.BatchNormalization(64),
            color4=L.Convolution2D(64, 32, 3, stride=1, pad=1, wscale=self.wscale),
            color_bn4=L.BatchNormalization(32),
            color5=L.Convolution2D(32, 2, 3, stride=1, pad=1, wscale=self.wscale),
        )
    def __call__(self, x, train=True):
        h0_1 = F.leaky_relu(self.bn1(self.conv1(x), test=not train))
        h0_2 = F.leaky_relu(self.bn2(self.conv2(h0_1), test=not train))
        h0_3 = F.leaky_relu(self.bn3(self.conv3(h0_2), test=not train))
        h0_4 = F.leaky_relu(self.bn4(self.conv4(h0_3), test=not train))
        h0_5 = F.leaky_relu(self.bn5(self.conv5(h0_4), test=not train))
        h0_6 = F.leaky_relu(self.bn6(self.conv6(h0_5), test=not train))
        h0_7 = F.leaky_relu(self.bn7(self.conv7(h0_6), test=not train))
        h0_8 = F.leaky_relu(self.bn8(self.conv8(h0_7), test=not train))
        h0_9 = F.leaky_relu(self.bn9(self.conv9(h0_8), test=not train))
        h0_10 = F.leaky_relu(self.bn10(self.conv10(h0_9), test=not train))
        h1_1 = F.leaky_relu(self.class_bn1(self.class1(h0_10), test=not train))
        h1_2 = F.leaky_relu(self.class_bn2(self.class2(h1_1), test=not train))
        h1_3 = F.leaky_relu(self.class_bn3(self.class3(h1_2), test=not train))
        h1_4 = F.reshape(h1_3, h1_3.data.shape + (1, 1))
        h2_1 = F.leaky_relu(self.feature_bn1(self.feature1(h0_6), test=not train))
        h2_2 = F.leaky_relu(self.feature_bn2(self.feature2(h2_1), test=not train))
        h3_1 = F.concat((h2_2, (F.broadcast_to(h1_4, h2_2.data.shape))), axis=1)
        h3_2 = F.leaky_relu(self.fusion_bn(self.fusion(h3_1), test=not train))
        h4_1 = F.leaky_relu(self.color_bn1(self.color1(h3_2), test=not train))
        h4_2 = F.leaky_relu(self.color_bn2(self.color2(h4_1), test=not train))
        h4_3 = F.leaky_relu(self.color_bn3(self.color3(h4_2), test=not train))
        h4_4 = F.leaky_relu(self.color_bn4(self.color4(h4_3), test=not train))
        return F.tanh(self.color5(h4_4))

class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
        )
        self.mean = np.asarray([104, 117, 124], dtype=np.float32)

    def preprocess(self, image, input_type='BGR'):
        if input_type == 'RGB':
            image = image[:,:,::-1]
        return np.rollaxis(image - self.mean, 2)

    def postprocess(self, image, output_type='RGB'):
        image = np.transpose(image, (1, 2, 0)) + self.mean
        if input_type == 'RGB':
            return image[:,:,::-1]
        else:
            return image


    def __call__(self, x):
        y1 = F.relu(self.conv1_2(F.relu(self.conv1_1(x))))
        h = F.max_pooling_2d(y1, 2, stride=2)
        y2 = F.relu(self.conv2_2(F.relu(self.conv2_1(h))))
        h = F.max_pooling_2d(y2, 2, stride=2)
        y3 = F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h))))))
        h = F.max_pooling_2d(y3, 2, stride=2)
        y4 = F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(h))))))
        h = F.max_pooling_2d(y4, 2, stride=2)
        y5 = F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(h))))))
        return (y5, y3)
