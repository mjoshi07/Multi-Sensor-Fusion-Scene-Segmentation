import os
import pandas as pd
import tensorflow as tf

import Config as cf
import Network as nw
import DataGenerator as dg


def train():

    if not os.path.exists(cf.MODEL_DIR):
        os.makedirs(cf.MODEL_DIR)

    rgb_images, depth_images = dg.load_data(cf.DATA_DIR)

    data = {"image": rgb_images, "mask": depth_images}
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42)

    train_samples = int(df["image"].size * cf.TRAIN_PERCENT)
    train_loader = dg.Generator(data=df[:train_samples].reset_index(drop="true"), batch_size=cf.BATCH_SIZE, dim=(cf.HEIGHT, cf.WIDTH))
    validation_loader = dg.Generator(data=df[train_samples:].reset_index(drop="true"), batch_size=cf.BATCH_SIZE, dim=(cf.HEIGHT, cf.WIDTH))

    model = nw.SegmentationModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=cf.LR)
    model.compile(optimizer)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    # [NOTE]: this saved model weights will be loaded as load_weights and not load_model
    model_chkpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(cf.MODEL_DIR, "intermediate"), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    callback_list = [early_stopping, model_chkpt]

    model.fit(train_loader, epochs=cf.EPOCHS, validation_data=validation_loader, callbacks=callback_list)

    model.save(os.path.join(cf.MODEL_DIR, cf.MODEL_CHKPT_NAME), save_format="tf")


if __name__ == "__main__":
    train()
