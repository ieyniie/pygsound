import numpy as np
import pygsound as ps
from wavefile import WaveWriter, Format
import random
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm


custom_array = np.array(
[[0, 0.035, 0], [-0.0303, 0.0175, 0], [-0.0303, -0.0175, 0], [0, -0.035, 0], [0.0303, -0.0175, 0],
    [0.0303, 0.0175, 0]])
r = 0.6
sound_a = np.array([0.5,0.6,0.6,0.7,0.75,0.8,0.9,0.9])
s = 0.1
# mesh1 = ps.loadobj("cube.obj",r,s)
mesh2 = ps.createbox(40, 50, 10, sound_a, s)
def compute_array(mesh, src_coord, lis_coord, micarray,num):
    ctx = ps.Context()
    ctx.diffuse_count = 20000
    ctx.specular_count = 2000
    ctx.channel_type = ps.ChannelLayoutType.mono
    scene = ps.Scene()
    scene.setMesh(mesh)

    res = {}
    rate = 0
    res['samples'] = []
    for j in range(2):
        src = ps.Source(src_coord[:,j])
        src.radius = 0.01
        res_buffer = []
        for offset in micarray:
            lis = ps.Listener((offset + lis_coord).tolist())
            lis.radius = 0.002

            res_ch = scene.computeIR([src], [lis], ctx)
            rate = res_ch['rate']
            sa = res_ch['samples'][0][0][0]
            res['rate'] = rate
            res_buffer.append(sa)
        res_temp = np.zeros((len(res_buffer), np.max([len(ps) for ps in res_buffer])))
        for i, c in enumerate(res_buffer):
             res_temp[i, :len(c)] = c
        res['samples'].append(res_temp)
    np.save('/data/datasets/gsound_rir3/rir_' + str(num) + '.npy',res)
    return res

def main(num):
    phi1 = random.randint(0,210)
    deta_phi = random.randint(-60,60)
    phi2 = int(phi1 + deta_phi)
    mic_centerx = random.randint(3,20)
    mic_centery = random.randint(3,20)
    distance1 = round(random.uniform(0.05,1.40),2)
    distance2 = round(random.uniform(0.05,1.40),2)
    x_s1 = mic_centerx+distance1*np.cos(np.deg2rad(phi1))
    y_s1 = mic_centery+distance1*np.sin(np.deg2rad(phi1))
    x_s2 = mic_centerx+distance2*np.cos(np.deg2rad(phi2))
    y_s2 = mic_centery+distance2*np.sin(np.deg2rad(phi2))
    h1 = round(random.uniform(1.50,1.80),2)
    h2 = round(random.uniform(1.50,1.80),2)
    spk1_pos = np.array([x_s1, y_s1, h1])
    spk2_pos = np.array([x_s2, y_s2, h2])
    spk_pos = np.c_[spk1_pos,spk2_pos]

    # src_coord = [x_s1, y_s1, 1.7]
    lis_coord = [mic_centerx, mic_centery, 1.68]

    res = compute_array(mesh2, spk_pos, lis_coord, custom_array,num)

    # with WaveWriter('array_rir0.wav', channels=np.shape(res['samples'][0])[0], samplerate=int(res['rate'])) as w:
    #     w.write(np.array(res['samples'][0]))

if __name__ == '__main__':
    mp.set_start_method('spawn')

    cpu_num = 16
    train_idx = list(range(0,30000))
    # main(0)
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(main, train_idx), total=len(train_idx)))

