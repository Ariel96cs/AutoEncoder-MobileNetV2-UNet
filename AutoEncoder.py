from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Conv2D,Dropout,Dense,Conv2DTranspose, BatchNormalization,Input,Flatten,concatenate,Reshape
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from CustomGenerator import CustomDataGen
from sklearn.model_selection import train_test_split
import json
from tensorflow.keras import backend as K
import numpy as np

class MobileNetV2_UNet:
    def __init__(self,input_shape=(224,224,3),read_image_func=None,encoder_output_size=None):
        super().__init__()
        self.dice_coefficient = False
        self.model = None
        self.input_shape = input_shape
        self.read_image_func = read_image_func
        self.encoder_output_size = encoder_output_size

        if encoder_output_size is not None:
            self.encoder_last_layer_name = "latent-space-rep"
        else:
            self.encoder_last_layer_name = "block_16_project_BN"

        
    def build_model(self,nodes):
        mobileNet = MobileNetV2(input_shape=self.input_shape,include_top=False)
        inputs = mobileNet.inputs

        c5 = [layer for layer in mobileNet.layers if layer.name == 'block_16_project_BN'][0].output
        c4 = [layer for layer in mobileNet.layers if layer.name == 'block_12_add'][0].output
        c3 = [layer for layer in mobileNet.layers if layer.name == 'block_5_add'][0].output
        c2 = [layer for layer in mobileNet.layers if layer.name == 'block_2_add'][0].output
        c1 = [layer for layer in mobileNet.layers if layer.name == 'expanded_conv_project_BN'][0].output

        # c5 = [layer for layer in mobileNet.layers if layer.name == 'block_16_project'][0].output
        # c4 = [layer for layer in mobileNet.layers if layer.name == 'block_12_project'][0].output
        # c3 = [layer for layer in mobileNet.layers if layer.name == 'block_5_project'][0].output
        # c2 = [layer for layer in mobileNet.layers if layer.name == 'block_2_project'][0].output
        # c1 = [layer for layer in mobileNet.layers if layer.name == 'Conv1'][0].output

        if self.encoder_output_size is not None:
            volumeSize = K.int_shape(c5)
            x = Flatten()(c5)
            x = Dense(self.encoder_output_size,name="latent-space-rep")(x)
            x = Dense(np.prod(volumeSize[1:]))(x)
            c5 = Reshape((volumeSize[1],volumeSize[2],volumeSize[3]))(x)

        # DECODER Unet

        u6 = Conv2DTranspose(nodes*8, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(nodes*8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(nodes*8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)

        u7 = Conv2DTranspose(nodes*4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(nodes*4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(nodes*4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)

        u8 = Conv2DTranspose(nodes*2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(nodes*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(nodes*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)

        u9 = Conv2DTranspose(nodes, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = BatchNormalization()(c9)

        u10 = Conv2DTranspose(nodes, (2, 2), strides=(2, 2), padding='same',name='last_block_transponse')(c9)
#         u10 = concatenate([u9, c1], axis=3)
        c10 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u10)
        c10 = BatchNormalization()(c10)
        c10 = Dropout(0.1)(c10)
        c10 = Conv2D(nodes, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c10)
        c10 = BatchNormalization()(c10)


        outputs = Conv2D(3, (1, 1), activation='sigmoid')(c10)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        return self.model
    def train_model(self, x_train, y_train, early_stopping_patience=None, reduce_lr_callback=True,epochs=60, checkpoint_filepath='./checkpoints/',
                    save_best_only=True,validation_split=0.1, verbose=1,
                    batch_size=32, use_custom_generator_training=True,
                    save_distribution=False, initial_epoch=0,
                    x_val=None,y_val=None):
        callbacks = []
        if early_stopping_patience is not None:
            early_stopping = EarlyStopping(patience=early_stopping_patience,verbose=verbose)
            callbacks.append(early_stopping)
        if reduce_lr_callback:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=0.001)
            callbacks.append(reduce_lr)
        if checkpoint_filepath is not None:
            name = 'MobileNetV2_Unet_{epoch:02d}-{val_loss:.4f}.h5'
            if self.dice_coefficient:
                name = 'MobileNetV2_Unet_{epoch:02d}-{val_loss:.4f}_dice_coeff-{val_dice_coeff:.4f}.h5'
            save_model_path = f'{checkpoint_filepath}/{name}'
            check_pointer = ModelCheckpoint(save_model_path, verbose=verbose,save_best_only=save_best_only)

            callbacks.append(check_pointer)
        if use_custom_generator_training:
            if x_val is None or y_val is None:
                X_train,X_val,y_train,y_val = train_test_split(x_train,y_train,test_size=validation_split,shuffle=True,
                                                            random_state=42)
            else:
                X_train,X_val,y_train,y_val = x_train,x_val,y_train,y_val

            if save_distribution:
                print("saving train and validation distribution")
                with open('train_val_distribution.json','w') as file:
                    json.dump({'X_train':X_train,'X_val':X_val},file)

            print("Train:",len(X_train))
            print("Validation:", len(X_val))
            traingen = CustomDataGen(X_train,y_train,batch_size,self.input_shape,load_images_func=self.read_image_func,data_augmentation=True)
            valgen = CustomDataGen(X_val, y_val, batch_size, self.input_shape,load_images_func=self.read_image_func)

            history = self.model.fit(traingen, validation_data=valgen,epochs=epochs,batch_size=batch_size,
                                     callbacks=callbacks,initial_epoch=initial_epoch)
        else:
            history = self.model.fit(x_train,y_train,validation_split=validation_split,batch_size=batch_size,
                                     epochs=epochs,
                                     callbacks=callbacks,initial_epoch=initial_epoch)
        return history

    def compile_model(self,loss_function='mse',show_metrics=False,metrics=None):
        print("Compiling model")
        if not show_metrics:
            metrics = []
        if loss_function == 'dice_loss':
            self.dice_coefficient = True
            self.model.compile(optimizer='adam', loss=self.bce_dice_loss, metrics=metrics)
        else:
            self.model.compile(optimizer=Adam(1e-3), loss=loss_function, metrics=metrics)
        
    def load_model(self, model_path):
        self.model = load_model(model_path,custom_objects={'bce_dice_loss':self.bce_dice_loss,'dice_coeff':self.dice_coeff})
        return self.model

    def save_autoencoder(self,path):
        if self.model is not None:
            output_layer = [layer.output for layer in self.model.layers if layer.name == self.encoder_last_layer_name][0]
            model = Model(self.model.input,output_layer,name='autoencoder')
            model.save(path)
            return True
        return False
    def model_is_compiled(self):
        return self.model._is_compiled

    def dice_coeff(self,y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score
    
    def dice_loss(self,y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss
    
    def bce_dice_loss(self,y_true, y_pred):
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss




