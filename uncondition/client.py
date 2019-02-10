import requests
import numpy as np
import json



'''
global variables
'''
melody_seq = []
chord_seq = []
tempo_seq = []
theta = 0

'''
show leadsheet sequence in console
'''
def printchord_seq(seq):
	trans = np.flip(seq, 0)
	for r_i, r in enumerate(trans):
		print('[{}]'.format(r_i), end='')
		print() 
		print('[{}]'.format(r), end='') 
		print()
		# print('[{}]'.format(r_i), end='') 
		# for i, w in enumerate(r):
		#     if i > 0 and i % 12 == 0:
		#         print('|', end='')
		#     if w == 0:
		#         print('_', end='')
		#     else:
		#         print('[{}]'.format(w), end='')
		# print()

def printmelody_seq(seq):
	trans = np.flip(seq, 0)
	for r_i, r in enumerate(trans):
		print('[{}]'.format(r_i), end='')
		print() 
		print('[{}]'.format(r), end='') 
		print() 


### GET
def get_static(addr, SONG1, SONG2):
    global melody_seq
    global chord_seq
    global tempo_seq
    global songnames_seq
    global theta
    
    # test_url = addr + '/static'
    test_url = addr + '/static' + '/'+SONG1+'/'+SONG2

    response = requests.get(
    test_url,
    headers=headers)

    print(response)

    r_json = json.loads(response.text)
    chord_seq = r_json['chord']
    melody_seq = r_json['melody']
    tempo_seq = r_json['tempo']
    songnames_seq = r_json['songnames']
    theta = r_json['theta']

    # for i, d in enumerate(chord_seq):
    #     print('({})'.format(i))
    #     printchord_seq(d)

    # for i, d in enumerate(melody_seq):
    #     print('({})'.format(i))
    #     printmelody_seq(d

    for i in enumerate(tempo_seq):
        print('({})'.format(i))

    for i in enumerate(songnames_seq):
        print('({})'.format(i))

    print('theta:'+'({})'.format(theta)) 

# ### POST
def post(addr): 
    global melody_seq
    global chord_seq
    global tempo_seq
    global songnames_seq
    global theta

    # test_url = addr + '/api/melody_chord'
    # test_url = addr + '/api/theta'
    test_url = addr + '/api/tempos'

    m1 = melody_seq[0]
    c1 = chord_seq[0]
    m2 = melody_seq[-1]
    c2 = chord_seq[-1]
    t1 = 80
    t2 = 200
    theta = 0.9

    melcho_data = {
                   'm_seq_1': m1,
                   'c_seq_1': c1,
                   'm_seq_2': m2,
                   'c_seq_2': c2,
                  }

    tempo_data = {
                  'tempo_1': t1,
                  'tempo_2': t2,
                  }

    theta_data = {
                  'theta': theta,
                  }

    response = requests.post(
    test_url,
    json=json.dumps(tempo_data),
    headers=headers)
    # print(response.text)

    r_json = json.loads(response.text)
    chord_seq = r_json['chord']
    melody_seq = r_json['melody']
    tempo_seq = r_json['tempo']
    songnames_seq = r_json['songnames']
    theta = r_json['theta']

    # for i, d in enumerate(chord_seq):
    #     print('({})'.format(i))
    #     printchord_seq(d)

    # for i, d in enumerate(melody_seq):
    #     print('({})'.format(i))
    #     printmelody_seq(d)

    for i in enumerate(tempo_seq):
        print('({})'.format(i))

    for i in enumerate(songnames_seq):
        print('({})'.format(i))

    print('theta:'+'({})'.format(theta)) 


addr = 'http://localhost:5001'
SONG1 = '1'
SONG2 = '4'
content_type = 'application/json'
headers = {'content-type': content_type}

'''
main function
'''
get_static(addr, SONG1, SONG2)
post(addr)

