import argparse
from PIL import Image
import io
import numpy as np
import os
import cPickle as pickle
import six
import chainer
from chainer import cuda
from chainer import Variable
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import net
from image_util import ImageUtil

image_util = ImageUtil()
image_size = 224
train_image_size = image_size / 2

def forward(feature, colorization, x_batch, train=True):
    x = Variable(x_batch, volatile=True)
    f1, f2 = feature(x)
    if train:
        f1 = Variable(f1.data)
        f2 = Variable(f2.data)
    return colorization(f1, f2, train=train)

def train_one(feature, colorization, optimizer, x_batch, y_batch):
    colorization.zerograds()
    y = forward(feature, colorization, x_batch)
    t = Variable(y_batch)
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()
    return float(loss.data)

def train_epoch(feature, colorization, optimizer, images, batch_size, gpu_device):
    total_loss = 0
    if gpu_device == None:
        xp = np
    else:
        xp = cuda.cupy
    image_len = len(images)
    perm = np.random.permutation(image_len)
    x_batch = np.zeros((batch_size, 3, train_image_size, train_image_size), dtype=np.float32)
    y_batch = np.zeros((batch_size, 2, train_image_size, train_image_size), dtype=np.float32)
    for i in six.moves.range(0, (image_len // batch_size)):
        for j, image_index in enumerate(perm[i * batch_size:(i + 1) * batch_size]):
            with io.BytesIO(images[image_index]) as b:
                image = np.asarray(image_util.rgb_to_lab(Image.open(b).resize((train_image_size, train_image_size))))
            image_l = image[:,:,:1]
            image_ab = image[:,:,1:]
            image_ab = np.where(image_ab, image_ab >= 128, image_ab - 128, image_ab + 128)
            x_batch[j,:] = np.float32(feature.preprocess(np.repeat(image_l, 3, axis=2), input_type='RGB'))
            y_batch[j,:] = np.float32(np.rollaxis(image_ab, 2)) / 127.5 - 1
        total_loss += train_one(feature, colorization, optimizer, xp.asarray(x_batch), xp.asarray(y_batch))
        if (i + 1) % 100 == 0:
            print('batch: {} done loss: {}'.format(i + 1, total_loss / (i + 1)))

    return total_loss / (image_len // batch_size)

def train(feature, colorization, optimizer, train_images, test_images, batch_size, epoch_num, gpu_device, out_image_dir=None):
    for epoch in six.moves.range(0, epoch_num):
        loss = train_epoch(feature, colorization, optimizer, train_images, batch_size, gpu_device)
        serializers.save_hdf5('{0}_{1:03d}.model'.format(args.output, epoch), colorization)
        serializers.save_hdf5('{0}_{1:03d}.state'.format(args.output, epoch), optimizer)
        if out_image_dir is not None:
            if gpu_device == None:
                xp = np
            else:
                xp = cuda.cupy
            test_batch_size = 10
            x_batch = np.zeros((test_batch_size, 3, train_image_size, train_image_size), dtype=np.float32)
            y_batch = np.zeros((test_batch_size, 2, train_image_size, train_image_size), dtype=np.float32)
            image_ls = np.zeros((test_batch_size, train_image_size, train_image_size, 1), dtype=np.uint8)
            image_abs = np.zeros((test_batch_size, train_image_size, train_image_size, 2), dtype=np.uint8)
            for j in six.moves.range(10):
                with io.BytesIO(train_images[j]) as b:
                    image = np.asarray(image_util.rgb_to_lab(Image.open(b).resize((train_image_size, train_image_size))))
                image_l = image[:,:,:1]
                image_ab = image[:,:,1:]
                image_ab = np.where(image_ab, image_ab >= 128, image_ab - 128, image_ab + 128)
                image_ls[j,:] = image_l[:,:,:]
                image_abs[j,:] = image[:,:,1:]
                x_batch[j,:] = np.float32(feature.preprocess(np.repeat(image_l, 3, axis=2), input_type='RGB'))
                y_batch[j,:] = np.float32(np.rollaxis(image_ab, 2)) / 127.5 - 1
            y = forward(feature, colorization, xp.asarray(x_batch), train=True)
            y = np.uint8((cuda.to_cpu(y.data).transpose(0, 2, 3, 1) + 1) * 127.5)
            y = np.where(y >= 128, y - 128, y + 128)
            for j in six.moves.range(10):
                image = np.concatenate((image_ls[j], y[j]), axis=2)
                file_name = 'out_{0:04d}_{1:03d}.jpg'.format(epoch, j)
                image_util.lab_to_rgb(Image.fromarray(image, mode='LAB')).save(os.path.join(out_image_dir, file_name))
                image = np.concatenate((image_ls[j], image_abs[j]), axis=2)
                file_name = 'out_{0:04d}_{1:03d}_org.jpg'.format(epoch, j)
                image_util.lab_to_rgb(Image.fromarray(image, mode='LAB')).save(os.path.join(out_image_dir, file_name))
            for j in six.moves.range(10):
                with io.BytesIO(test_images[j]) as b:
                    image = np.asarray(image_util.rgb_to_lab(Image.open(b).resize((train_image_size, train_image_size))))
                image_l = image[:,:,:1]
                image_ab = image[:,:,1:]
                image_ab = np.where(image_ab, image_ab >= 128, image_ab - 128, image_ab + 128)
                image_ls[j,:] = image_l[:,:,:]
                image_abs[j,:] = image[:,:,1:]
                x_batch[j,:] = np.float32(feature.preprocess(np.repeat(image_l, 3, axis=2), input_type='RGB'))
                y_batch[j,:] = np.float32(np.rollaxis(image_ab, 2)) / 127.5 - 1
            y = forward(feature, colorization, xp.asarray(x_batch), train=True)
            y = np.uint8((cuda.to_cpu(y.data).transpose(0, 2, 3, 1) + 1) * 127.5)
            y = np.where(y >= 128, y - 128, y + 128)
            for j in six.moves.range(10):
                image = np.concatenate((image_ls[j], y[j]), axis=2)
                file_name = 'out_{0:04d}_test_{1:03d}.jpg'.format(epoch, j)
                image_util.lab_to_rgb(Image.fromarray(image, mode='LAB')).save(os.path.join(out_image_dir, file_name))
                image = np.concatenate((image_ls[j], image_abs[j]), axis=2)
                file_name = 'out_{0:04d}_test_{1:03d}_org.jpg'.format(epoch, j)
                image_util.lab_to_rgb(Image.fromarray(image, mode='LAB')).save(os.path.join(out_image_dir, file_name))
            print('epoch: {} done loss: {}'.format(epoch + 1, loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGAN trainer for ETL9')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='vgg16.model', type=str,
                        help='image feature model file path')
    parser.add_argument('--input', '-i', default=None, type=str,
                        help='input model file path without extension')
    parser.add_argument('--output', '-o', required=True, type=str,
                        help='output model file path without extension')
    parser.add_argument('--iter', default=100, type=int,
                        help='number of iteration')
    parser.add_argument('--batch_size', default=48, type=int,
                        help='batch size')
    parser.add_argument('--out_image_dir', default=None, type=str,
                        help='output directory to output images')
    parser.add_argument('--dataset', '-d', default='dataset/images.pkl', type=str,
                        help='dataset file path')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    args = parser.parse_args()

    vgg_net = net.VGG()
    colorization_net = net.Colorization()
    colorization_optimizer = optimizers.Adam(alpha=args.lr, beta1=0.9)
    colorization_optimizer.setup(colorization_net)

    serializers.load_hdf5(args.model, vgg_net)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        vgg_net.to_gpu()
        colorization_net.to_gpu()
    if args.input != None:
        serializers.load_hdf5(args.input + '.model', colorization_net)
        serializers.load_hdf5(args.input + '.state', colorization_optimizer)

    if args.out_image_dir != None:
        if not os.path.exists(args.out_image_dir):
            try:
                os.mkdir(args.out_image_dir)
            except:
                print'cannot make directory {}'.format(args.out_image_dir)
                exit()
        elif not os.path.isdir(args.out_image_dir):
            print 'file path {} exists but is not directory'.format(args.out_image_dir)
            exit()
    with open(args.dataset, 'rb') as f:
        images = pickle.load(f)
    train_images = images[:-100]
    test_images = images[-100:]

    train(vgg_net, colorization_net, colorization_optimizer, train_images, test_images, args.batch_size, args.iter, gpu_device=args.gpu, out_image_dir=args.out_image_dir)
