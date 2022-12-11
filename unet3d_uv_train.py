#%%
import keras
from unet3d import get_model
import numpy as np
import keras.backend as K
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#%%
fpath = '../3Deddy_uvts_file_new/seperated/seperated_'
num_classes = 4
u = np.load(f'{fpath}u_2000.npy')
v = np.load(f'{fpath}v_2000.npy')
spd = np.load(f'{fpath}spd_2000.npy')
label = np.load(f'{fpath}eddy_label_2000.npy')

for yyyy in range(2001, 2008):
    u_tmp = np.load(f'{fpath}u_{yyyy}.npy')
    v_tmp = np.load(f'{fpath}v_{yyyy}.npy')
    spd_tmp = np.load(f'{fpath}spd_{yyyy}.npy')
    label_tmp = np.load(f'{fpath}eddy_label_{yyyy}.npy')

    u = np.concatenate([u,u_tmp])
    v = np.concatenate([v,v_tmp])
    spd = np.concatenate([spd, spd_tmp])
    label = np.concatenate([label,label_tmp])

#%%
u = np.expand_dims(u,axis=-1)
v = np.expand_dims(v,axis=-1)
spd = np.expand_dims(spd, axis=-1)
# label = np.where(label==4, 3, label)

trainX = np.concatenate([u,v,spd], axis=-1)
trainX = np.swapaxes(trainX, 1,3) # (N, 16, 32, 32 ,3) -> (N, 32, 32, 16, 3)
trainy = np.expand_dims(np.swapaxes(label, 1, 3), axis=-1)  # 1234 -> 0123

#%%
model = get_model((32, 32, 16, 3), num_classes)
# model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("log/stcc_3d_uv_v2_0512.h5", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',patience=30)
]
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=['sparse_categorical_accuracy'])
model.fit(trainX, trainy, epochs=10000, batch_size=72, validation_split=0.1, callbacks=callbacks)
