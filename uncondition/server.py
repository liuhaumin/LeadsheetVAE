import os
import time
from flask import Flask, request, Response
from flask_cors import CORS
import numpy as np
import json
import pypianoroll
from pypianoroll import Multitrack, Track
import torch
import torch.utils.data as Data
import vae_4bar as model

app = Flask(__name__)
app.config['ENV'] = 'development'
CORS(app)
USE_CUDA = torch.cuda.is_available()

'''
load model
'''
path = '/home/paul115236/NAS/paul115236/leadsheetvae/deploy/presets/'
# path = '/home/vibertthio/local_dir/vibertthio/leadsheetvae/server/presets/'
checkpt = [ m for m in os.listdir(path) if '.pt' in m ][0]
songfiles = [m for m in os.listdir(path) if '.midi' in m]
print(songfiles)
vae = model.VAE(hidden_m=256, hidden_c=48, bar=4).to(model.device)
if USE_CUDA:
    vae.load_state_dict(torch.load(path + checkpt))
else:
    vae.load_state_dict(torch.load(path + checkpt, map_location='cpu'))

'''
global variables
'''
### two songs for interpolation
UNIT_LEN = 4 # the length for encode and decode
INTERP_NUM = 7 # number of interpolated samples between
TOTAL_LEN = (INTERP_NUM+2)*4
RHYTHM_THRESHOLD = 0.8 # number: 0~1
TEMPO = np.array([100, 100])
SONGNAME = np.array(['pay phone.mid','some one like you.mid'])
# m, c = load_midi(SONG1, SONG2, UNIT_LEN)
# m_roll, c_roll = interp_sample(vae, SONG1, SONG2, m, c, INTERP_NUM)

'''
load preset
'''
# preset_file = './static/payphone2someonelikeyou.npy' # type bool (48*total_len, 128, 2)
# seed = np.load(preset_file)
# seed_m = seed[:, :, 0]
# seed_c = seed[:, :, 1]

'''
utils
'''
### numpytojson
def numpy2json(m, c, z, tempo, songnames, theta):
    out_melody = m.tolist() # (n, 4, 48)
    out_chord = c.tolist() #(n, 4, 4)
    out_z = z.tolist() # (n, 304)
    out_tempo = tempo.tolist() #(2)
    out_songnames = songnames.tolist() #(2)
    out_theta = theta #(1)

    #print(type(out_melody))
    response = {
        'melody': out_melody,
        'chord': out_chord,
        'z': out_z,
        'tempo': out_tempo,
        'songnames': out_songnames,
        'theta': out_theta,
    }
    response_pickled = json.dumps(response)
    return response_pickled


'''
api route
'''
@app.route('/static', methods=['GET'], endpoint='static_1')
def static():
    with torch.no_grad():
        global UNIT_LEN
        global INTERP_NUM
        global TOTAL_LEN
        global path
        global RHYTHM_THRESHOLD
        global TEMPO
        global SONGNAME
        
        song1 = path + songfiles[42] # 
        song2 = path + songfiles[51] # someonelikeyou

        # print('song1:',song1)
        # print('song2:',song2)
        m, c, tempo = model.load_midi(song1, song2, UNIT_LEN)
        m_seq, c_seq, z = model.interp_sample(vae, m, c, INTERP_NUM, RHYTHM_THRESHOLD)
        print("z", np.shape(z))
        TEMPO = tempo
        SONGNAME = np.array([songfiles[42],songfiles[51]]) 

    response_pickled = numpy2json(m_seq, c_seq, z, TEMPO, SONGNAME, RHYTHM_THRESHOLD)
    return Response(response=response_pickled, status=200, mimetype="application/json")


@app.route('/static/<s1>/<s2>', methods=['GET'], endpoint='static_twosong_1', defaults={'num': 7})
@app.route('/static/<s1>/<s2>/<num>', methods=['GET'], endpoint='static_twosong_1')
def static_twosong(s1, s2, num):
    with torch.no_grad():
        global UNIT_LEN
        global INTERP_NUM
        global TOTAL_LEN
        global path
        global RHYTHM_THRESHOLD
        global TEMPO
        global SONGNAME
        INTERP_NUM = num # number of interp group
        # TOTAL_LEN = (INTERP_NUM + 2)*4 # number of group * 4bar = total bars
        
        song1 = path + songfiles[int(s1)]
        song2 = path + songfiles[int(s2)]

        # print('song1:',song1)
        # print('song2:',song2)
        m, c, tempo = model.load_midi(song1, song2, UNIT_LEN)
        # print(m.shape)
        # print(c.shape)
        # print(tempo.shape)
        m_seq, c_seq, z = model.interp_sample(vae, m, c, INTERP_NUM, RHYTHM_THRESHOLD)
        print("z", np.shape(z))
        TEMPO = tempo
        SONGNAME = np.array([songfiles[int(s1)],songfiles[int(s2)]])    

    response_pickled = numpy2json(m_seq, c_seq, z, TEMPO, SONGNAME, RHYTHM_THRESHOLD)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/api/melody_chord', methods=['POST'])
def melody_chord():
    r = request.json
    if (type(r) == dict):
        r_json = r
    else:
        r_json = json.loads(r)
        
    m1 = r_json['m_seq_1']
    c1 = r_json['c_seq_1']
    m2 = r_json['m_seq_2']
    c2 = r_json['c_seq_2']

    m_seq1 = np.asarray(m1).astype(int)
    c_seq1 = np.asarray(c1)
    m_seq2 = np.asarray(m2).astype(int)
    c_seq2 = np.asarray(c2)

    with torch.no_grad():
        m, c = model.load_seq(m_seq1, c_seq1, m_seq2, c_seq2)
        m_seq, c_seq, z = model.interp_sample(vae, m, c, INTERP_NUM, RHYTHM_THRESHOLD)

    response_pickled = numpy2json(m_seq, c_seq, z, TEMPO, SONGNAME, RHYTHM_THRESHOLD)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/api/theta', methods=['POST'])
def theta():
    global RHYTHM_THRESHOLD
    global SONGNAME
    r = request.json
    if (type(r) == dict):
        r_json = r
    else:
        r_json = json.loads(r)

    theta_temp = r_json['theta']

    theta = np.float(theta_temp)

    with torch.no_grad():

        RHYTHM_THRESHOLD = theta

        song1 = path + SONGNAME[0] # heyjude
        song2 = path + SONGNAME[1] # someonelikeyou

        # print('song1:',song1)
        # print('song2:',song2)
        m, c, tempo = model.load_midi(song1, song2, UNIT_LEN)
        m_seq, c_seq, z = model.interp_sample(vae, m, c, INTERP_NUM, RHYTHM_THRESHOLD)

    response_pickled = numpy2json(m_seq, c_seq, z, TEMPO, SONGNAME, RHYTHM_THRESHOLD)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/api/tempos', methods=['POST'])
def tempos():
    global TEMPO
    global SONGNAME
    r = request.json
    if (type(r) == dict):
        r_json = r
    else:
        r_json = json.loads(r)

    t1 = r_json['tempo_1']
    t2 = r_json['tempo_2']

    tempo_1 = np.asarray(t1)
    tempo_2 = np.asarray(t2)

    TEMPO = np.array([tempo_1, tempo_2])

    song1 = path + SONGNAME[0] # heyjude
    song2 = path + SONGNAME[1] # someonelikeyou

    m, c, tempo = model.load_midi(song1, song2, UNIT_LEN)
    m_seq, c_seq, z = model.interp_sample(vae, m, c, INTERP_NUM, RHYTHM_THRESHOLD)

    response_pickled = numpy2json(m_seq, c_seq, z, TEMPO, SONGNAME, RHYTHM_THRESHOLD)
    return Response(response=response_pickled, status=200, mimetype="application/json")

'''
start app
'''
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)