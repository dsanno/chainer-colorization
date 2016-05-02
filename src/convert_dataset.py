import io
import os
import sys
import cPickle as pickle
from PIL import Image
import argparse

def open_and_reshape_image(image_path, shape):
    image = Image.open(image_path).convert('RGB')
    out_w, out_h = shape
    image_w, image_h = image.size
    if image_w * out_h > image_h * out_w:
        w = out_w * image_h // out_h
        h = image_h
        x = (image_w - w) // 2
        y = 0
    else:
        w = image_w
        h = out_h * image_w // out_w
        x = 0
        y = (image_h - h) // 2
    return image.crop((x, y, x + w, y + h)).resize(shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert images to dataset file')
    parser.add_argument('in_dir', type=str, help='directory that contains input image files')
    parser.add_argument('out_file', type=str, help='output file path')
    parser.add_argument('--file_num', '-n', type=int, default=None, help='the number of files to be stored in dataset')
    args = parser.parse_args()

    image_shape = (224, 224)
    in_dir = args.in_dir
    out_file = args.out_file
    files = os.listdir(in_dir)
    n = 0
    if args.file_num is not None:
        files = files[:args.file_num]

    images = []
    for file_name in files:
        name, ext = os.path.splitext(file_name)
        if not ext in ['.jpg', '.jpeg', '.png', '.gif']:
            continue
        try:
            image = open_and_reshape_image(os.path.join(in_dir, file_name), image_shape)
        except IOError:
            print('cannot open file: {}'.format(file_name))
            continue
        with io.BytesIO() as b:
            image.save(b, format='jpeg')
            images.append(b.getvalue())
        n += 1
        if n >= args.file_num:
            break

    with open(out_file, 'wb') as f:
        pickle.dump(images, f)
