from tensorflow.keras.utils import Sequence
import cv2 as cv
import numpy as np
import json
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from random import randint,randrange,random
from RandomAugmetationGen import RandomAugmetationGen
from sklearn.utils import shuffle

def rot():
    angle=randint(0,50)
    p_rot = (randint(-1,1)*randint(1,50),randint(-1,1)*randint(1,50))
    scale=1-random()*0.1
    def f (image):
        height,width = image.shape[:2]
        rotation_matrix =  cv.getRotationMatrix2D((width//2 + p_rot[0],height//2 + p_rot[1]),angle,scale)
        rotated_image = cv.warpAffine(image,rotation_matrix,(width,height)) 
        return rotated_image
    return f

def brightness():
    decrease_b=bool(randint(0,1))
    bright_constant=randint(10,80)
    def f(image):
        bright = np.ones(image.shape,dtype='uint8')*bright_constant
        if decrease_b:
            return cv.subtract(image,bright)
        return cv.add(image,bright)
    return f

def flips():
    flip_mode=randint(-1,1)
    to_flip = bool(randint(0,1))

    def f(image):
        if to_flip:
            return cv.flip(image,flip_mode)
        return image

    return f


class CustomDataGen(Sequence):
    def __init__(self, x_paths,
                 y_paths,
                 batch_size,
                 input_size=(224, 224, 3),
                 shuffle_=True, load_images_func=None,data_augmentation=False):
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle_
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.read_image = load_images_func
        self.data_augmentation = data_augmentation

        self.n = len(self.x_paths)
        if shuffle_:
            self.x_paths,self.y_paths = shuffle(self.x_paths,self.y_paths)

    def on_epoch_end(self):
        pass

    def __load_image(self,image_path,shape,gray=False,mods=None):
        if gray:
            image = load_img(image_path,color_mode='grayscale')
        else:
            image = load_img(image_path)

        image = image.resize(shape[:2])
        image = img_to_array(image)

        if len(image.shape)<3 and gray:
            image  = np.reshape(image,image.shape+(1,))
        
        
        if mods is not None:
            for f in mods:
                image = f(image)

        image = image* 1./ 255
        # print("image shape", image.shape)
        # cv.imshow('Image',image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return image

    # def __load_images(self,image_paths,shape,gray=False):
    #     return np.array([self.__load_image(str(path),shape,gray)for path in image_paths])

    def __getitem__(self, index):
        batches_x = self.x_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batches_y = self.y_paths[index * self.batch_size:(index + 1) * self.batch_size]

        if self.read_image is None:
            
            X ,y = [],[]
            shape = self.input_size
            # print("Input_shape in getItem",shape)
            if self.data_augmentation:
                augm = RandomAugmetationGen(self.input_size[:2],rotation_range=40,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest',
                                    brightness_range=(10,80))

                for x_path,y_path in zip(batches_x,batches_y):
                    modification_set_of_params_f = augm.generate_random_transformation_f()
                    modification_fx = modification_set_of_params_f(apply_brightness_mod=True)
                    # modification_fy = modification_set_of_params_f(apply_brightness_mod=False)

                    # y_mod = [modification_fy]
                    x_mod = [modification_fx]
                    X.append(self.__load_image(x_path,shape,gray=self.input_size[-1]==1,mods=x_mod))
                    y.append(self.__load_image(y_path,shape,gray=self.input_size[-1]==1,mods=x_mod))
            else:
                for x_path,y_path in zip(batches_x,batches_y):
                    X.append(self.__load_image(x_path,shape,gray=self.input_size[-1]==1))
                    y.append(self.__load_image(y_path,shape,gray=self.input_size[-1]==1))


        else:
            batch = [self.read_image(img_path,label_path,self.input_size[0:2]) for img_path,label_path in zip(batches_x,batches_y)]

            X,y = zip(*batch)
        X,y = np.array(X),np.array(y)

        return X, y

    def __len__(self):
        return self.n // self.batch_size


