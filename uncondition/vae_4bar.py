import numpy as np
import pretty_midi
import pypianoroll
from pypianoroll import Multitrack, Track
import torch
import torch.utils.data as Data
import os
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt



'''
check the GPU usage
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


'''
chord convertion info
'''
chord_name=([
            "Cdim",
            "Cm",
            "C",
            "Caug",
            "Cm7",
            "C7",
            "CM7",

            "C#dim",
            "C#m",
            "C#",
            "C#aug",
            "C#m7",
            "C#7",
            "C#M7",

            "Ddim",
            "Dm",
            "D",
            "Daug",
            "Dm7",
            "D7",
            "DM7",

            "D#dim",
            "D#m",
            "D#",
            "D#aug",
            "D#m7",
            "D#7",
            "D#M7",

            "Edim",
            "Em",
            "E",
            "Eaug",
            "Em7",
            "E7",
            "EM7",

            "Fdim",
            "Fm",
            "F",
            "Faug",
            "Fm7",
            "F7",
            "FM7",

            "F#dim",
            "F#m",
            "F#",
            "F#aug",
            "F#m7",
            "F#7",
            "F#M7",

            "Gdim",
            "Gm",
            "G",
            "Gaug",
            "Gm7",
            "G7",
            "GM7",

            "G#dim",
            "G#m",
            "G#",
            "G#aug",
            "G#m7",
            "G#7",
            "G#M7",

            "Adim",
            "Am",
            "A",
            "Aaug",
            "Am7",
            "A7",
            "AM7",

            "A#dim",
            "A#m",
            "A#",
            "A#aug",
            "A#m7",
            "A#7",
            "A#M7",

            "Bdim",
            "Bm",
            "B",
            "Baug",
            "Bm7",            
            "B7",            
            "BM7",

			# "no event",
            "Nothing"
])

chord_name_v2=([
            "Co",
            "Cm",
            "CM",
            "CM#5",
            "Cm7",
            "C7",
            "CMaj7",

            "Dbo",
            "Dbm",
            "DbM",
            "DbM#5",
            "Dbm7",
            "Db7",
            "DbMaj7",

            "Do",
            "Dm",
            "DM",
            "DM#5",
            "Dm7",
            "D7",
            "DMaj7",

            "Ebo",
            "Ebm",
            "EbM",
            "EbM#5",
            "Ebm7",
            "Eb7",
            "EbMaj7",

            "Eo",
            "Em",
            "EM",
            "EM#5",
            "Em7",
            "E7",
            "EMaj7",

            "Fo",
            "Fm",
            "FM",
            "FM#5",
            "Fm7",
            "F7",
            "FMaj7",

            "Gbo",
            "Gbm",
            "GbM",
            "GbM#5",
            "Gbm7",
            "Gb7",
            "GbMaj7",

            "Go",
            "Gm",
            "GM",
            "GM#5",
            "Gm7",
            "G7",
            "GMaj7",

            "Abo",
            "Abm",
            "AbM",
            "AbM#5",
            "Abm7",
            "Ab7",
            "AbMaj7",

            "Ao",
            "Am",
            "AM",
            "AM#5",
            "Am7",
            "A7",
            "AMaj7",

            "Bbo",
            "Bbm",
            "BbM",
            "BbM#5",
            "Bbm7",
            "Bb7",
            "BbMaj7",

            "Bo",
            "Bm",
            "BM",
            "BM#5",
            "Bm7",            
            "B7",            
            "BMaj7",

			# "no event",
            "Nothing"
])

chord_composition=np.array([
    [1,0,0,1,0,0,1,0,0,0,0,0],#Cdim
    [1,0,0,1,0,0,0,1,0,0,0,0],#Cm
    [1,0,0,0,1,0,0,1,0,0,0,0],#C
    [1,0,0,0,1,0,0,0,1,0,0,0],#Caug
    [1,0,0,1,0,0,0,1,0,0,1,0],#Cm7
    [1,0,0,0,1,0,0,1,0,0,1,0],#C7
    [1,0,0,0,1,0,0,1,0,0,0,1],#CM7

    [0,1,0,0,1,0,0,1,0,0,0,0],#C#dim
    [0,1,0,0,1,0,0,0,1,0,0,0],#C#m
    [0,1,0,0,0,1,0,0,1,0,0,0],#C#
    [0,1,0,0,0,1,0,0,0,1,0,0],#C#aug
    [0,1,0,0,1,0,0,0,1,0,0,1],#C#m7
    [0,1,0,0,0,1,0,0,1,0,0,1],#C#7
    [1,1,0,0,0,1,0,0,1,0,0,0],#C#M7

    [0,0,1,0,0,1,0,0,1,0,0,0],#Ddim
    [0,0,1,0,0,1,0,0,0,1,0,0],#Dm
    [0,0,1,0,0,0,1,0,0,1,0,0],#D
    [0,0,1,0,0,0,1,0,0,0,1,0],#Daug
    [1,0,1,0,0,1,0,0,0,1,0,0],#Dm7
    [1,0,1,0,0,0,1,0,0,1,0,0],#D7
    [0,1,1,0,0,0,1,0,0,1,0,0],#DM7

    [0,0,0,1,0,0,1,0,0,1,0,0],#D#dim
    [0,0,0,1,0,0,1,0,0,0,1,0],#D#m
    [0,0,0,1,0,0,0,1,0,0,1,0],#D#
    [0,0,0,1,0,0,0,1,0,0,0,1],#D#aug
    [0,1,0,1,0,0,1,0,0,0,1,0],#D#m7
    [0,1,0,1,0,0,0,1,0,0,1,0],#D#7
    [0,0,1,1,0,0,0,1,0,0,1,0],#D#M7

    [0,0,0,0,1,0,0,1,0,0,1,0],#Edim
    [0,0,0,0,1,0,0,1,0,0,0,1],#Em
    [0,0,0,0,1,0,0,0,1,0,0,1],#E
    [1,0,0,0,1,0,0,0,1,0,0,0],#Eaug
    [0,0,1,0,1,0,0,1,0,0,0,1],#Em7
    [0,0,1,0,1,0,0,0,1,0,0,1],#E7
    [0,0,0,1,1,0,0,0,1,0,0,1],#EM7

    [0,0,0,0,0,1,0,0,1,0,0,1],#Fdim
    [1,0,0,0,0,1,0,0,1,0,0,0],#Fm
    [1,0,0,0,0,1,0,0,0,1,0,0],#F
    [0,1,0,0,0,1,0,0,0,1,0,0],#Faug
    [1,0,0,1,0,1,0,0,1,0,0,0],#Fm7
    [1,0,0,1,0,1,0,0,0,1,0,0],#F7
    [1,0,0,0,1,1,0,0,0,1,0,0],#FM7

    [1,0,0,0,0,0,1,0,0,1,0,0],#F#dim
    [0,1,0,0,0,0,1,0,0,1,0,0],#F#m
    [0,1,0,0,0,0,1,0,0,0,1,0],#F#
    [0,0,1,0,0,0,1,0,0,0,1,0],#F#aug
    [0,1,0,0,1,0,1,0,0,1,0,0],#F#m7
    [0,1,0,0,1,0,1,0,0,0,1,0],#F#7
    [0,1,0,0,0,1,1,0,0,0,1,0],#F#M7

    [0,1,0,0,0,0,0,1,0,0,1,0],#Gdim
    [0,0,1,0,0,0,0,1,0,0,1,0],#Gm
    [0,0,1,0,0,0,0,1,0,0,0,1],#G
    [0,0,0,1,0,0,0,1,0,0,0,1],#Gaug
    [0,0,1,0,0,1,0,1,0,0,1,0],#Gm7
    [0,0,1,0,0,1,0,1,0,0,0,1],#G7
    [0,0,1,0,0,0,1,1,0,0,0,1],#GM7

    [0,0,1,0,0,0,0,0,1,0,0,1],#G#dim
    [0,0,0,1,0,0,0,0,1,0,0,1],#G#m
    [1,0,0,1,0,0,0,0,1,0,0,0],#G#
    [1,0,0,0,1,0,0,0,1,0,0,0],#G#aug
    [0,0,0,1,0,0,1,0,1,0,0,1],#G#m7
    [1,0,0,1,0,0,1,0,1,0,0,0],#G#7
    [1,0,0,1,0,0,0,1,1,0,0,0],#G#M7

    [1,0,0,1,0,0,0,0,0,1,0,0],#Adim
    [1,0,0,0,1,0,0,0,0,1,0,0],#Am
    [0,1,0,0,1,0,0,0,0,1,0,0],#A
    [0,1,0,0,0,1,0,0,0,1,0,0],#Aaug
    [1,0,0,0,1,0,0,1,0,1,0,0],#Am7
    [0,1,0,0,1,0,0,1,0,1,0,0],#A7
    [0,1,0,0,1,0,0,0,1,1,0,0],#AM7

    [0,1,0,0,1,0,0,0,0,0,1,0],#A#dim
    [0,1,0,0,0,1,0,0,0,0,1,0],#A#m
    [0,0,1,0,0,1,0,0,0,0,1,0],#A#
    [0,0,1,0,0,0,1,0,0,0,1,0],#A#aug
    [0,1,0,0,0,1,0,0,1,0,1,0],#A#m7
    [0,0,1,0,0,1,0,0,1,0,1,0],#A#7
    [0,0,1,0,0,1,0,0,0,1,1,0],#A#M7

    [0,0,1,0,0,1,0,0,0,0,0,1],#Bdim
    [0,0,1,0,0,0,1,0,0,0,0,1],#Bm
    [0,0,0,1,0,0,1,0,0,0,0,1],#B
    [0,0,0,1,0,0,0,1,0,0,0,1],#Baug
    [0,0,1,0,0,0,1,0,0,1,0,1],#Bm7
    [0,0,0,1,0,0,1,0,0,1,0,1],#B7
    [0,0,0,1,0,0,1,0,0,0,1,1],#BM7

	# [1,1,1,1,1,1,1,1,1,1,1,1],#no event
    [0,0,0,0,0,0,0,0,0,0,0,0] #Nothing
])
'''
data form convertion functions
'''
def parse_data(bar):
	beat = bar*4
	m = np.load("X.npy").astype(int)
	c = np.load("y.npy").astype(int)

	mr = m[:,-4:]
	m = m[:,:-4]

	m = m.reshape(int(m.shape[0]/beat), 4, int(m.shape[1]*beat/4))
	mr = mr.reshape(int(mr.shape[0]/beat), 4, int(mr.shape[1]*beat/4))
	m = np.concatenate([m, mr], 2)
	c = c.reshape(int(c.shape[0]/beat), 4, int(c.shape[1]*beat/4))

	ratio = 0.1
	T = int(m.shape[0]*ratio)

	train_m = m[:-T]
	train_c = c[:-T]
	test_m = m[-T:]
	test_c = c[-T:]

	train_m = torch.from_numpy(train_m).type(torch.FloatTensor)
	train_c = torch.from_numpy(train_c).type(torch.FloatTensor)
	test_m = torch.from_numpy(test_m).type(torch.FloatTensor)
	test_c = torch.from_numpy(test_c).type(torch.FloatTensor)

	return train_m, train_c, test_m, test_c

def midi2numpy(filename, b): # b: how many bars per group
    midi = Multitrack(filename, beat_resolution=12)
    tempo = int(midi.tempo[0])

    m = midi.tracks[0].pianoroll
    c = midi.tracks[1].pianoroll

    print('m:',m.shape)
    print('c:',c.shape)

    bar = int(m.shape[0]/12/4)
    bar = int(bar - bar%4) # only take 4*n bars

    unit = 3
    x1 = np.zeros((bar*16, 49)) # 48~96
    x2 = np.zeros((bar*16, 1))
    for i in range(x1.shape[0]):
        
        if np.sum(m[i*unit+1]>0) > 0:
            x1[i, np.where(m[i*unit+1]>0)[0][0] - 48] = 1
        else:
            x1[i, 48] = 1

        if i != 0 and np.sum(m[i*unit+1]==m[i*unit-1]) == 128:
            x2[i,0] = 0
        else:
            x2[i,0] = 1

    unit = 12
    y = np.zeros((bar*4, 12))
    for i in range(y.shape[0]):
        if np.sum(c[i*unit+1]>0) > 0:
            # print(np.where(c[i*unit+1]>0)[0]%12)
            for note in np.where(c[i*unit+1]>0)[0]%12:
                y[i, note] = 1

    # 4bar
    x1 = x1.reshape((int(bar/b), b, 49*16)) # (49 pitches x 16 timesteps) / bar
    x2 = x2.reshape((int(bar/b), b, 1*16)) # (1 onset x 16 timesteps) /bar
    x = np.concatenate([x1, x2], 2)
    y = y.reshape((int(bar/b), b, 12*4)) # (12 chroma x 4 timesteps) / bar

    return x, y, tempo

def numpy2midi(m, c, theta, filename):
	resolution = 12
	ratio = int(resolution/4) # 3
	bar = int(m.shape[0]/4)

	mr = m[:,-4:].flatten()
	m = np.argmax(m[:,:-4].reshape(m.shape[0]*4, 49), 1)
	midi_m = np.zeros((resolution*bar*4, 128))
	for i in range(len(m)):
		if m[i] == 48: # stop
			continue

		if i+1 != len(m):
			if mr[i+1] > theta and m[i+1]==m[i]:
				midi_m[i*ratio:(i+1)*ratio - 1, m[i]+48] = 100
			else:
				midi_m[i*ratio:(i+1)*ratio, m[i]+48] = 100
		else: #i+1 != len(m) and mr[i+1] == 0:
			midi_m[i*ratio:(i+1)*ratio, m[i]+48] = 100
		# else: #i+1 == len(m):
			# midi_m[i*ratio:(i+1)*ratio - 1, m[i]+48] = 100

	midi_c = np.zeros((resolution*bar*4, 128))
	nextchord = -1
	for i in range(len(c)):
		# round
		# midi_c[i*resolution:(i+1)*resolution-1, np.where(np.round(c[i])==1)[0]+48] = 100

		# dot
		if np.sum(c[i]) == 0:
			chord = len(chord_composition)-1
		else:
			chord = np.argmax( np.dot(chord_composition, c[i])/(np.linalg.norm(chord_composition, axis=1)+1e-5)/(np.linalg.norm(c)+1e-5) )
		if i < len(c)-1 and i%4!=3:
			nextchord = np.argmax( np.dot(chord_composition, c[i+1])/(np.linalg.norm(chord_composition, axis=1)+1e-5)/(np.linalg.norm(c)+1e-5) )
		else:
			nextchord = -1

		main = int(chord/7)
		for j in np.where(chord_composition[chord]==1)[0]:
			if j < main:
				if chord == nextchord:
					midi_c[i*resolution:(i+1)*resolution, j+60] = 100
				else:
					midi_c[i*resolution:(i+1)*resolution-1, j+60] = 100
			else:
				if chord == nextchord:
					midi_c[i*resolution:(i+1)*resolution, j+48] = 100
				else:
					midi_c[i*resolution:(i+1)*resolution-1, j+48] = 100

	track_m = Track(pianoroll=midi_m, program=0, is_drum=False)
	track_c = Track(pianoroll=midi_c, program=0, is_drum=False)
	multitrack = Multitrack(tracks=[track_m, track_c], tempo=80.0, beat_resolution=resolution)
	pypianoroll.write(multitrack, filename+".mid")

def numpy2pianoroll(m, c): ### output form to pianoroll form
	resolution = 12
	ratio = int(resolution/4) # 3
	bar = int(m.shape[0]/4)

	mr = m[:,-4:].flatten()
	m = np.argmax(m[:,:-4].reshape(m.shape[0]*4, 49), 1)
	midi_m = np.zeros((resolution*bar*4, 128))
	for i in range(len(m)):
		if m[i] == 48: # stop
			continue

		if i+1 != len(m):
			if mr[i+1] > 0.5:
				midi_m[i*ratio:(i+1)*ratio - 1, m[i]+48] = 100
			else:
				midi_m[i*ratio:(i+1)*ratio, m[i]+48] = 100
		else: #i+1 != len(m) and mr[i+1] == 0:
			midi_m[i*ratio:(i+1)*ratio - 1, m[i]+48] = 100
		# else: #i+1 == len(m):
			# midi_m[i*ratio:(i+1)*ratio - 1, m[i]+48] = 100

	midi_c = np.zeros((resolution*bar*4, 128))
	nextchord = -1
	for i in range(len(c)):
		# round
		# midi_c[i*resolution:(i+1)*resolution-1, np.where(np.round(c[i])==1)[0]+48] = 100

		# dot
		if np.sum(c[i]) == 0:
			chord = len(chord_composition)-1
		else:
			chord = np.argmax( np.dot(chord_composition, c[i])/(np.linalg.norm(chord_composition, axis=1)+1e-5)/(np.linalg.norm(c)+1e-5) )
		if i < len(c)-1 and i%4!=3:
			nextchord = np.argmax( np.dot(chord_composition, c[i+1])/(np.linalg.norm(chord_composition, axis=1)+1e-5)/(np.linalg.norm(c)+1e-5) )
		else:
			nextchord = -1

		main = int(chord/7)
		for j in np.where(chord_composition[chord]==1)[0]:
			if j < main:
				if chord == nextchord:
					midi_c[i*resolution:(i+1)*resolution, j+60] = 100
				else:
					midi_c[i*resolution:(i+1)*resolution-1, j+60] = 100
			else:
				if chord == nextchord:
					midi_c[i*resolution:(i+1)*resolution, j+48] = 100
				else:
					midi_c[i*resolution:(i+1)*resolution-1, j+48] = 100

	return midi_m, midi_c # pianoroll type

def midi2pianoroll(filename):
	### Create a `pypianoroll.Multitrack` instance
	multitrack = Multitrack(filepath=filename+'.mid', tempo=120.0, beat_resolution=12)
	### save pypianoroll
	pypianoroll.save(filename+'.npz', multitrack, compressed=True)
	data = pypianoroll.load(filename+'.npz')
	data_tracks = data.get_stacked_pianorolls()
	data_bool = data_tracks.astype(bool)
	np.save(filename+'.npy',data_bool)

def numpy2seq(m, c, theta): ### output form to sequence form
	# theta: 0 ~ 1 as the threshold for melody rhythm (mr)
	resolution = 12
	ratio = int(resolution/4) # 3
	bar = int(m.shape[0]/4)

	'''
	melody processing
	'''
	mr = m[:,-4:].flatten()
	m = np.argmax(m[:,:-4].reshape(m.shape[0]*4, 49), 1)
	midi_m = np.zeros((resolution*bar*4, 128))
	for i in range(len(m)):
		if m[i] == 48: # stop
			continue

		if i+1 != len(m):
			if mr[i+1] > theta and m[i+1]==m[i]:
				midi_m[i*ratio:(i+1)*ratio - 1, m[i]+48] = 100
			else:
				midi_m[i*ratio:(i+1)*ratio, m[i]+48] = 100
		else: #i+1 != len(m) and mr[i+1] == 0:
			midi_m[i*ratio:(i+1)*ratio, m[i]+48] = 100
		# else: #i+1 == len(m):
			# midi_m[i*ratio:(i+1)*ratio - 1, m[i]+48] = 100
	
	### turn midi_m (m, 128)--> m_seq(n,4,48)
	m_seq = np.zeros((np.shape(midi_m)[0]))
	for i in range(np.shape(midi_m)[0]):
		idx = np.where(midi_m[i, :]==100)
		#print(idx)
		if len(idx[0])==0:
			m_seq[i] = -1
		else:
			m_seq[i] = idx[0][0]		
		#print('m_seq:',m_seq[i])
	m_seq = m_seq.reshape((-1,4,48))
	
	'''
	chord processing
	'''
	c_seq = np.chararray(shape = (len(c)), itemsize = 5, unicode = True)
	#nextchord = -1
	for i in range(len(c)):
		if np.sum(c[i]) == 0:
			chord = len(chord_composition)-1
		else:
			chord = np.argmax( np.dot(chord_composition, c[i])/(np.linalg.norm(chord_composition, axis=1)+1e-5)/(np.linalg.norm(c)+1e-5) )
		# if i < len(c)-1 and i%4!=3:
		# 	nextchord = np.argmax( np.dot(chord_composition, c[i+1])/(np.linalg.norm(chord_composition, axis=1)+1e-5)/(np.linalg.norm(c)+1e-5) )
		# else:
		# 	nextchord = -1
		c_seq[i] = chord_name_v2[chord]
	
	c_seq = c_seq.reshape((-1,4,4))
	# print('c_seq:',np.shape(c_seq))
	# print('m_seq:',np.shape(m_seq))

	return m_seq, c_seq # pianoroll type	

def seq2numpy(m_seq, c_seq):
	m_seq = m_seq.reshape(-1) # (4,48)-->(192)
	c_seq = c_seq.reshape(-1) # (4,4)-->(16)
	bar = 4
	b = 4 # how many bars per group
	unit = 3
	x1 = np.zeros((bar*16, 49)) # 48~96
	x2 = np.zeros((bar*16, 1))

	for i in range(x1.shape[0]):
		if m_seq[i*unit]==-1:
			x1[i, 48] = 1
		else:
			x1[i, m_seq[i*unit]-48] = 1

		if i != 0 and (m_seq[i*unit]==m_seq[i*unit-1]):
			x2[i,0] = 0
		else:
			x2[i,0] = 1
	
	y = np.zeros((bar*4, 12))
	for i in range(y.shape[0]):
		chord = chord_name_v2.index(c_seq[i])
		y[i] = chord_composition[chord]

	# 4bar
	x1 = x1.reshape((int(bar/b), b, 49*16)) # (49 pitches x 16 timesteps) / bar
	x2 = x2.reshape((int(bar/b), b, 1*16)) # (1 onset x 16 timesteps) /bar
	x = np.concatenate([x1, x2], 2)
	y = y.reshape((int(bar/b), b, 12*4)) # (12 chroma x 4 timesteps) / bar

	return x, y
	
def c_seq2pianoroll(c): 
	c = c.reshape(-1)
	resolution = 12
	ratio = int(resolution/4) # 3
	bar = int(c.shape[0]/4)

	midi_c = np.zeros((resolution*bar*4, 128))
	nextchord = -1
	for i in range(len(c)):
		# round
		# midi_c[i*resolution:(i+1)*resolution-1, np.where(np.round(c[i])==1)[0]+48] = 100

		# dot
		chord = chord_name_v2.index(c[i])
		if i < len(c)-1 and i%4!=3:
			nextchord = chord_name_v2.index(c[i+1])
		else:
			nextchord = -1
		
		main = int(chord/7)
		for j in np.where(chord_composition[chord]==1)[0]:
			if j < main:
				if chord == nextchord:
					midi_c[i*resolution:(i+1)*resolution, j+60] = 100
				else:
					midi_c[i*resolution:(i+1)*resolution-1, j+60] = 100
			else:
				if chord == nextchord:
					midi_c[i*resolution:(i+1)*resolution, j+48] = 100
				else:
					midi_c[i*resolution:(i+1)*resolution-1, j+48] = 100

	return midi_c # pianoroll type	
	x1 = np.zeros((bar*16, 49)) # 48~96
	x2 = np.zeros((bar*16, 1))
def m_roll2seq(m_roll): 
	### turn m_roll(m, 128)-->m_seq(n,4,48)
	m_seq = np.zeros((np.shape(m_roll)[0]))
	for i in range(np.shape(m_roll)[0]):
		idx = np.where(m_roll[i, :]==100)
		#print(idx)
		if len(idx[0])==0:
			m_seq[i] = -1
		else:
			m_seq[i] = idx[0][0]		
		#print('m_seq:',m_seq[i])
	m_seq = m_seq.reshape((-1,4,48))
	print(np.shape(m_seq))
	return m_seq


'''
interpolation function
'''
def slerp(a, b, steps):
	aa =  np.squeeze(a/np.linalg.norm(a))
	bb =  np.squeeze(b/np.linalg.norm(b))
	ttt = np.sum(aa*bb)
	omega = np.arccos(ttt)
	so = np.sin(omega)
	step_deg = 1 / (steps+1)
	step_list = []

	for idx in range(1, steps+1):
		t = step_deg*idx
		tmp = np.sin((1.0-t)*omega) / so * a + np.sin(t*omega)/so * b
		step_list.append(tmp)
	return step_list

class VAE(nn.Module):
    """Class that defines the model."""
    def __init__(self, hidden_m, hidden_c, bar):
        super(VAE, self).__init__()

        self.hidden_m = hidden_m
        self.hidden_c = hidden_c
        self.bar = bar
        self.timestep = 4
        self.Bi = 2

        self.BGRUm 		= nn.GRU(input_size=800, hidden_size=self.hidden_m, num_layers=2, batch_first=True, bidirectional=True)
        self.BGRUm2 	= nn.GRU(input_size=self.hidden_m*self.Bi*self.timestep, hidden_size=self.hidden_m*self.Bi*self.timestep, num_layers=1, batch_first=True, bidirectional=False)
        self.BGRUc 		= nn.GRU(input_size=48, hidden_size=self.hidden_c, num_layers=2, batch_first=True, bidirectional=True)
        self.BGRUc2 	= nn.GRU(input_size=self.hidden_c*self.Bi*self.timestep, hidden_size=self.hidden_c*self.Bi*self.timestep, num_layers=1, batch_first=True, bidirectional=False)

        self.hid2mean	= nn.Linear((self.hidden_m+self.hidden_c)*self.Bi*self.timestep, self.hidden_m+self.hidden_c)
        self.hid2var	= nn.Linear((self.hidden_m+self.hidden_c)*self.Bi*self.timestep, self.hidden_m+self.hidden_c)

        self.lat2hidm	= nn.Linear(self.hidden_m+self.hidden_c, self.hidden_m*self.Bi*self.timestep)
        self.lat2hidc	= nn.Linear(self.hidden_m+self.hidden_c, self.hidden_c*self.Bi*self.timestep)
        
        self.outm		= nn.Linear(self.hidden_m*self.Bi*self.timestep, 800)
        self.outc		= nn.Linear(self.hidden_c*self.Bi*self.timestep, 48)

    def encode(self, m, c):
        batch_size = m.shape[0]

        m, _ = self.BGRUm(m)
        c, _ = self.BGRUc(c)

        h1 = m.contiguous().view(batch_size, self.hidden_m*self.Bi*self.timestep)
        h2 = c.contiguous().view(batch_size, self.hidden_c*self.Bi*self.timestep)
        h = torch.cat([h1, h2], 1)
        mu = self.hid2mean(h)
        var = self.hid2var(h)

        return mu, var

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(mu.shape).cuda()
        z = eps * std + mu
        return z

    def decode(self, z):
        melody = torch.zeros((z.shape[0], self.timestep, 800))
        if use_cuda:
            melody = melody.cuda()	

        m = self.lat2hidm(z)
        m = m.view(m.shape[0], 1, m.shape[1])
        for i in range(self.timestep):
            m, _ = self.BGRUm2(m)
            melody[:,i,:] = torch.sigmoid(self.outm(m[:,0,:]))

        chord = torch.zeros((z.shape[0], self.timestep, 48))
        if use_cuda:
            chord = chord.cuda()		

        c = self.lat2hidc(z)
        c = c.view(c.shape[0], 1, c.shape[1])
        for i in range(self.timestep):
            c, _ = self.BGRUc2(c)
            chord[:,i,:] = torch.sigmoid(self.outc(c[:,0,:]))

        return melody, chord

    def forward(self, m, c):
        mu, logvar = self.encode(m.view(-1, self.timestep, 800), c.view(-1, self.timestep, 48))
        
        ### reparameter
        z = self.reparameterize(mu, logvar)

        ### interpolation 4bar
        # a = 0
        # b = 1
        # z = z.cpu().detach().numpy()
        # interp = z[a].reshape(1, z.shape[1])
        # interp = np.concatenate([interp, np.array(slerp(z[a], z[b], 6))])
        # interp = np.concatenate([interp, z[b].reshape(1, z.shape[1])])
        # z = torch.from_numpy(interp).cuda()

        ### interpolation 8bar
        # z = z.cpu().detach().numpy()
        # orif4 = z[0:2].reshape(2, z.shape[1])
        # oris4 = z[2:4].reshape(2, z.shape[1])
        
        # f4 = np.array(slerp(z[0], z[2], 6))
        # s4 = np.array(slerp(z[1], z[3], 6))

        # interp = np.concatenate([orif4,f4,s4,oris4], 0)
        # z = torch.from_numpy(interp).cuda()

        ### random sample
        # z = torch.randn(m.shape[0], self.hidden_m+self.hidden_c).cuda()
        
        m, c = self.decode(z)
        return m, c, mu, logvar

    def interpolation(self, m, c, interp_num):
        ### encode
        mu, logvar = self.encode(m.view(-1, self.timestep, 800), c.view(-1, self.timestep, 48))
        ### only take mean from encoder
        z = mu
        print(z.shape)
        a = 0
        b = 1
        z = z.cpu().detach().numpy()
        interp = z[a].reshape(1, z.shape[1]) # a: the start clip
        interp = np.concatenate([interp, np.array(slerp(z[a], z[b], interp_num))]) # passing clips
        interp = np.concatenate([interp, z[b].reshape(1, z.shape[1])]) #b: the end clip
        if (use_cuda):
            z = torch.from_numpy(interp).cuda()	
        else:
            z = torch.from_numpy(interp)
        
        ### decode	
        m, c = self.decode(z)
        return m, c, mu, logvar, z

def loss_vae(predict_m, test_m, predict_c, test_c, mu, logvar):
	# loss	
	BCEm = F.binary_cross_entropy(predict_m.view(-1, bar*4*200), test_m.view(-1, bar*4*200), size_average=False)
	BCEc = F.binary_cross_entropy(predict_c.view(-1, bar*4*12), test_c.view(-1, bar*4*12), size_average=False)
	# MSE = criterion(predict_m, test_m)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	loss = BCEm + BCEc + KLD
	
	# accuracy
	m = torch.argmax(test_m[:,:,:-16].reshape(test_m.shape[0], test_m.shape[1]*16, 49), 2)
	pm = torch.argmax(predict_m[:,:,:-16].reshape(predict_m.shape[0], predict_m.shape[1]*16, 49), 2)
	accm = torch.sum(m==pm)/m.shape[1]

	mr = test_m[:,:,-16:].reshape(test_m.shape[0], test_m.shape[1]*16)
	pmr = torch.round(predict_m[:,:,-16:]).reshape(predict_m.shape[0], predict_m.shape[1]*16)
	accmr = torch.sum(mr==pmr)/mr.shape[1]

	c = test_c.reshape(test_c.shape[0], test_c.shape[1]*test_c.shape[2])
	pc = torch.round(predict_c).reshape(predict_c.shape[0], predict_c.shape[1]*predict_c.shape[2])
	accc = torch.sum(c==pc)/c.shape[1]

	return loss, accm, accmr, accc

def train_vae(model, epoch):
	model.train()
	train_loss=0; train_accm=0; train_accmr=0; train_accc=0
	for batch_idx, (m, c) in enumerate(train_loader):
		batch_m = Variable(m).cuda()
		batch_c = Variable(c).cuda()

		predict_m, predict_c, mu, logvar = model(batch_m, batch_c)
		loss, accm, accmr, accc = loss_vae(predict_m, batch_m, predict_c, batch_c, mu, logvar)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		train_accm += accm.item()
		train_accmr += accmr.item()
		train_accc += accc.item()

		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch_m), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(batch_m)))

	history_train = [[] for i in range(4)]
	train_loss /= len(train_loader.dataset)
	train_accm /= len(train_loader.dataset)
	train_accmr /= len(train_loader.dataset)
	train_accc /= len(train_loader.dataset)
	history_train[0] += [train_loss]
	history_train[1] += [train_accm]
	history_train[2] += [train_accmr]
	history_train[3] += [train_accc]
	print('====> Epoch: {:3d} Average loss: {:.4f} Acc: {:.4f}, {:.4f}, {:.4f}'.format(epoch, 
		train_loss, train_accm, train_accmr, train_accc))

def test_vae(model, epoch):
	model.eval()
	test_loss = 0; test_accm=0; test_accmr=0; test_accc=0
	for i, (m, c) in enumerate(test_loader):
		batch_m = Variable(m).cuda()
		batch_c = Variable(c).cuda()
		predict_m, predict_c, mu, logvar = model(batch_m, batch_c)		

		loss, accm, accmr, accc = loss_vae(predict_m, batch_m, predict_c, batch_c, mu, logvar)
		test_loss += loss.item()
		test_accm += accm.item()
		test_accmr += accmr.item()
		test_accc += accc.item()

	history_test = [[] for i in range(4)]
	test_loss /= len(test_loader.dataset)
	test_accm /= len(test_loader.dataset)
	test_accmr /= len(test_loader.dataset)
	test_accc /= len(test_loader.dataset)
	history_test[0] += [test_loss]
	history_test[1] += [test_accm]
	history_test[2] += [test_accmr]
	history_test[3] += [test_accc]
	print('====> Test set loss: {:.4f} Acc: {:.4f}, {:.4f}, {:.4f}'.format(test_loss,           
		test_accm, test_accmr, test_accc))  # accuracy_melody, accuracy_melody_rhythm, accuracy_chord


def load_midi(filename1, filename2, unit_len):
    x1, y1, tempo1 = midi2numpy(filename1, unit_len) # [2,4,200]
    x2, y2, tempo2 = midi2numpy(filename2, unit_len) # [4,4,200]
    print('load midi')
    print('x1:',x1.shape)
    print('x2:',x2.shape)
    x1 = x1[0:1]
    y1 = y1[0:1]
    x2 = x2[0:1]
    y2 = y2[0:1]

    x = np.concatenate([x1,x2], 0)
    x = torch.from_numpy(x).type(torch.FloatTensor)
    y = np.concatenate([y1,y2], 0)
    y = torch.from_numpy(y).type(torch.FloatTensor)
    tempo = np.array([tempo1,tempo2])
    tempo = torch.from_numpy(tempo).type(torch.FloatTensor)
    return x, y, tempo

def load_seq(m_seq1, c_seq1, m_seq2, c_seq2):
	x1, y1 = seq2numpy(m_seq1, c_seq1)
	x2, y2 = seq2numpy(m_seq2, c_seq2)

	x = np.concatenate([x1,x2], 0)
	x = torch.from_numpy(x).type(torch.FloatTensor)
	y = np.concatenate([y1,y2], 0)
	y = torch.from_numpy(y).type(torch.FloatTensor)
	
	return x, y
	

def interp_sample(model, x, y, interp_num, theta):
    if use_cuda:
        m, c, mu, var, z = model.interpolation(x.cuda(), y.cuda(), interp_num)
    else:
        m, c, mu, var, z = model.interpolation(x, y, interp_num)
    
    m = m.cpu().detach().numpy() # [8, 4, 800]: bar level
    c = c.cpu().detach().numpy() # [8, 4, 48]: bar level
    z = z.cpu().detach().numpy() 

    mr = m[:,:,-16:]
    m = m[:,:,:-16] 
    m = m.reshape(m.shape[0], m.shape[1]*4, int(m.shape[2]/4)) # m: [8, 16, 196]: beat level
    mr = mr.reshape(mr.shape[0], mr.shape[1]*4, int(mr.shape[2]/4)) # mr: [8, 16, 4]: beat level
    m = np.concatenate([m, mr], 2)

    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy() 
    xr = x[:,:,-16:] 
    x = x[:,:,:-16] 
    x = x.reshape(x.shape[0], x.shape[1]*4, int(x.shape[2]/4)) # m: [8, 16, 196]: beat level
    xr = xr.reshape(xr.shape[0], xr.shape[1]*4, int(xr.shape[2]/4)) # mr: [8, 16, 4]: beat level
    x = np.concatenate([x, xr], 2)	

    TOTAL_LEN = interp_num + 2 # start + end + passing clips number 
    m = np.concatenate([x[0:1],m[1:TOTAL_LEN-1],x[1:]],0)
    c = np.concatenate([y[0:1],c[1:TOTAL_LEN-1],y[1:]],0)
    m_seq, c_seq = numpy2seq(m[0:TOTAL_LEN].reshape((TOTAL_LEN*16,200)), c[0:TOTAL_LEN].reshape((TOTAL_LEN*16,12)), theta)		

    # tempo_seq = np.linspace(tempo[0], tempo[1], num=(interp_num+2)).round().astype(int)
    # m_roll, c_roll = numpy2pianoroll(m[0:TOTAL_LEN].reshape((TOTAL_LEN*16,200)), c[0:TOTAL_LEN].reshape((TOTAL_LEN*16,12)))		
    # numpy2midi(m[0:TOTAL_LEN].reshape((TOTAL_LEN*16,200)), c[0:TOTAL_LEN].reshape((TOTAL_LEN*16,12)),theta, './interp_output/'+'test.mid')
        
    # midi2pianoroll('./interp_output/'+filename1+'2'+filename2)

    #print(tempo_seq)
    #print(len(tempo_seq))

    return m_seq, c_seq, z# m_seq(n, 4, 48), c_seq(n, 4, 4), tempo_seq(n)

# ### Load training dataset
# print('Load training dataset')

# bar = 4
# train_m, train_c, test_m, test_c = parse_data(bar)

# # torch.manual_seed(42)
# # if torch.cuda.is_available():
# # 	torch.cuda.manual_seed(42)

# train_dataset = Data.TensorDataset(train_m, train_c)
# train_loader = Data.DataLoader(
# 	dataset = train_dataset,
# 	batch_size = 256,
# 	shuffle=True,
# 	num_workers=1
# )
# test_dataset = Data.TensorDataset(test_m, test_c)
# test_loader = Data.DataLoader(
# 	dataset = test_dataset,
# 	batch_size = 256,
# 	shuffle=True,
# 	num_workers=1
# )

# ### Load model
# print('Load model')
# model = VAE(hidden_m=256, hidden_c=48, bar=bar)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# if torch.cuda.is_available():
# 	model.cuda()

# model.load_state_dict(torch.load("./presets/vaernn_4bar220.pt"))
# # for epoch in range(400):
# # 	train_vae(epoch)
# # 	test_vae(epoch)
# # 	if (epoch+1) % 20 == 0:
# # 		print("epoch", epoch+1, "saving model.")
# # 		torch.save(model.state_dict(), "./presets/vaernn_4bar"+str(epoch+1)+".pt")


# ### interpo heyjude & payphone

# UNIT_LEN = 4 # the length for encode and decode
# INTERP_NUM = 6 # number of interpolated samples between
# SONG1 = 'payphone'
# SONG2 = 'whataboutlove'

# print('load_midi')
# m, c = load_midi(SONG1, SONG2, UNIT_LEN)
# print('interpolation sampling')
# m_roll, c_roll = interp_sample(model, SONG1, SONG2, m, c, INTERP_NUM)
# print(np.shape(m_roll))
