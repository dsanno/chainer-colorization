from PIL import Image
from PIL import ImageCms

class ImageUtil(object):
    def __init__(self):
        srgb_profile = ImageCms.createProfile('sRGB')
        lab_profile  = ImageCms.createProfile('LAB')
        self.rgb_to_lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, 'RGB', 'LAB')
        self.lab_to_rgb_transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, 'LAB', 'RGB')

    def rgb_to_lab(self, image):
        return ImageCms.applyTransform(image, self.rgb_to_lab_transform)

    def lab_to_rgb(self, image):
        return ImageCms.applyTransform(image, self.lab_to_rgb_transform)
