from pathlib import Path
from argparse import ArgumentParser
from AutoEncoder import MobileNetV2_UNet
import pandas as pd
import json
from os import makedirs

from sklearn.model_selection import train_test_split

BATCH_SIZE = 64
INPUT_SHAPE = (320,320,3)
LATENT_SPACE_SIZE = 64

def load_paths(csv_path):
    x_paths = []
    
    df = pd.read_csv(csv_path)
    x_paths = [row.filePath for _,row in df.iterrows()]

    return x_paths



def main():
    parser = ArgumentParser()
    parser.add_argument('-csv','--csv_path')
    parser.add_argument('-ck','--checkpoints_path')
    args = parser.parse_args()

    model = MobileNetV2_UNet(input_shape=INPUT_SHAPE,encoder_output_size=LATENT_SPACE_SIZE)
    model.build_model(nodes=2)
    model.compile_model()
    model.model.summary()

    x_paths = load_paths(args.csv_path)

    if not Path(args.checkpoints_path).exists():
        makedirs(args.checkpoints_path)
    model.train_model(x_paths,x_paths,early_stopping_patience=100,reduce_lr_callback=False,epochs=600,
                    checkpoint_filepath=args.checkpoints_path,batch_size=BATCH_SIZE,use_custom_generator_training=True,save_distribution=True)



main()