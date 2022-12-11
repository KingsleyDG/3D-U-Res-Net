#%%
from unet3d import get_model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import scipy.io as scio

#%%
def find_cnt(img, ce=3, ae=4):
    """
    ce:cyclonic eddy
    ae:anticyclonic eddy
    """
    cnt1=[]
    if np.sum(img==ce):
        binary1 = np.where(img==ce,ce,0).astype('uint8')
        cnt1, _ = cv2.findContours(binary1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    cnt2=[]
    if np.sum(img==ae):
        binary2 = np.where(img==ae,ae,0).astype('uint8')
        cnt2, _ = cv2.findContours(binary2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    return cnt1, cnt2

#%%

fpath = '../3Deddy_uvts_file/seperated/seperated_'
model_path = 'log/stcc_3d_uvt_0504.h5'
year = 2008
save_file = f'result/clean_uvt{year}_v2.mat'

level = np.array([   0.,   10.,   20.,   30.,   50.,   75.,  100.,  125.,  150.,
         200.,  250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,
        1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000., 2500.,
        3000., 3500., 4000., 4500., 5000., 5500.])
u = np.load(f'{fpath}u_{year}.npy')
v = np.load(f'{fpath}v_{year}.npy')
spd = np.load(f'{fpath}spd_{year}.npy')
t = np.load(f'{fpath}t_{year}.npy')
label = np.load(f'{fpath}eddy_label_{year}.npy')
label = np.swapaxes(label,1,3)
u = np.swapaxes(np.expand_dims(u,axis=-1), 1, 3)
v = np.swapaxes(np.expand_dims(v,axis=-1), 1, 3)
t = np.swapaxes(np.expand_dims(t,axis=-1), 1, 3)
spd = np.swapaxes(np.expand_dims(spd, axis=-1), 1, 3)

testX = np.concatenate([u,v,spd,t], axis=-1)


model = get_model((32, 32, 16, 4), 4)
model.load_weights(model_path)
pred = model.predict(testX)

pred_label = np.squeeze(np.argmax(pred,axis=-1))

#%%
aes = []
ces = []
for s in range(pred_label.shape[0]):
    aed=[]
    ced=[]
    for d in range(16):
        label_tmp = pred_label[s,...,d]
        cnt1, cnt2 = find_cnt(label_tmp)

        ae = []
        ce = []
        if cnt1:
            for cnt in cnt1:
                cntp = np.concatenate([cnt,cnt[0:1]], axis=0) 
                ae.append(np.squeeze(cntp))
        if cnt2:
            for cnt in cnt2:
                cntp = np.concatenate([cnt,cnt[0:1]], axis=0) 
                ce.append(np.squeeze(cntp))
        aed.append(ae)
        ced.append(ce)
    aes.append(aed)
    ces.append(ced)

scio.savemat(save_file,
{
    'u':u,'v':v,'spd':spd,
    'label':label,'pred':pred_label,
    'aes':aes,'ces':ces
    }
    )
#%%
x = np.linspace(0, 1, 32)
y = np.linspace(0, 1, 32)
# z = np.arange(0,16)

X, Y =  np.meshgrid(x,y)

# %%
plot_list = [0, 12, 15]
zz = level[plot_list]
X3, Y3, Z3 = np.meshgrid(x, y, -zz)
ww = np.zeros((32, 32, len(plot_list)))

ztick = [0, -0.4, -0.7]
x, y, z = np.meshgrid(np.linspace(0, 1, 11),
                      np.linspace(0, 1, 11),
                      np.array(ztick))
#%%
rnd = np.random.choice(pred_label.shape[0])
example = pred_label[rnd]
spd_p = np.squeeze(spd[rnd])
u_p = np.squeeze(u[rnd])
v_p = np.squeeze(v[rnd])
ax = plt.figure(dpi=300).add_subplot(projection='3d')
for sl, zt in zip(plot_list, ztick):
    # ax.contourf(X,Y,spd_p[...,sl]-0.01, zdir='z', 
    # offset=zt, cmap='rainbow', alpha=0.6)
    result = np.squeeze(pred_label[rnd,...,sl])
    cnt1, cnt2 = find_cnt(result)
    if cnt1:
        for cnt in cnt1:
            cntp = np.concatenate([cnt,cnt[0:1]], axis=0) / 32
            ax.plot(cntp[:,0,0], cntp[:,0,1], zt*np.ones(cntp.shape[0]),c='tab:red',lw=2)

    if cnt2:
        for cnt in cnt1:
            cntp = np.concatenate([cnt,cnt[0:1]], axis=0) / 32
            ax.plot(cntp[:,0,0], cntp[:,0,1], zt*np.ones(cntp.shape[0]),c='tab:blue',lw=2)

ax.quiver(x,y,z,u_p[::3,::3,plot_list], v_p[::3,::3,plot_list], 
ww[::3,::3],length=0.05,normalize=True,color='k', lw=0.5)
ax.set(xlim=(0,1), ylim=(0,1), zlim=(-0.7,0), 
xlabel='X', ylabel='Y', zlabel='Depth')
plt.savefig('img/3d.png')

# %%
