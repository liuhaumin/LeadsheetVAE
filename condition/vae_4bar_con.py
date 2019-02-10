import numpy as np
import pretty_midi
import pypianoroll
from pypianoroll import Multitrack, Track
import torch
import torchvision
import torch.utils.data as Data
import os
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable


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

chord_emotion=np.array([
    0, # C 
    -1,
    1,
    1,
    0,
    0,
    0,

    0, # C#
    -1,
    1,
    1,
    0,
    0,
    0,

    0, # D
    -1,
    1,
    1,
    0,
    0,
    0,

    0, # D#
    -1,
    1,
    1,
    0,
    0,
    0,

    0, # E
    -1,
    1,
    1,
    0,
    0,
    0,

    0, # F
    -1,
    1,
    1,
    0,
    0,
    0,

    0, # F#
    -1,
    1,
    1,
    0,
    0,
    0,

    0, # G
    -1,
    1,
    1,
    0,
    0,
    0,

    0, # G#
    -1,
    1,
    1,
    0,
    0,
    0,

    0, # A
    -1,
    1,
    1,
    0,
    0,
    0,
    
    0, # A#
    -1, 
    1,
    1,
    0,
    0,
    0,
    
    0, # B
    -1, 
    1,
    1,
    0,
    0,
    0,

    0 # nothing
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

def midi2numpy(filename, b):
	midi = Multitrack(filename, beat_resolution=12)
	m = midi.tracks[0].pianoroll
	c = midi.tracks[1].pianoroll
	bar = int(m.shape[0]/12/4)
	bar = int(bar - bar%8)

	unit = 3
	x1 = np.zeros((bar*16, 49)) # 48~96
	x2 = np.zeros((bar*16, 1))
	for i in range(x1.shape[0]):
		if np.sum(m[i*unit]>0) > 0:
			x1[i, np.where(m[i*unit]>0)[0][0] - 48] = 1
		else:
			x1[i, 48] = 1

		if i != 0 and np.sum(m[i*unit]==m[i*unit-1]) == 128:
			x2[i,0] = 0
		else:
			x2[i,0] = 1

	unit = 12
	y = np.zeros((bar*4, 12))
	for i in range(y.shape[0]):
		if np.sum(c[i*unit]>0) > 0:
			for note in np.where(c[i*unit]>0)[0]%12:
				y[i, note] = 1

	# 4bar
	x1 = x1.reshape((int(bar/b), 4, 16*49))
	x2 = x2.reshape((int(bar/b), 4, 16))
	x = np.concatenate([x1, x2], 2)
	y = y.reshape((int(bar/b), 4, 12*b))

	return x, y

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
			if mr[i+1] > theta:
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


	track_m = Track(pianoroll=midi_m, program=0, is_drum=False)
	track_c = Track(pianoroll=midi_c, program=0, is_drum=False)
	multitrack = Multitrack(tracks=[track_m, track_c], tempo=80.0, beat_resolution=resolution)
	pypianoroll.write(multitrack, filename)

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
    def __init__(self, hidden_m, hidden_c, bar):
        super(VAE, self).__init__()

        self.hidden_m	= hidden_m
        self.hidden_c	= hidden_c
        self.bar		= bar
        self.beat		= 4
        self.Bi			= 2
        self.Res        = 4

        self.BGRUm 		= nn.GRU(input_size=800, hidden_size=self.hidden_m, num_layers=2, batch_first=True, bidirectional=True)
        self.BGRUm2 	= nn.GRU(input_size=self.hidden_m*self.Bi*self.beat, hidden_size=self.hidden_m*self.Bi*self.beat, num_layers=1, batch_first=True, bidirectional=False)
        self.BGRUc 		= nn.GRU(input_size=48, hidden_size=self.hidden_c, num_layers=2, batch_first=True, bidirectional=True)
        self.BGRUc2 	= nn.GRU(input_size=self.hidden_c*self.Bi*self.beat, hidden_size=self.hidden_c*self.Bi*self.beat, num_layers=1, batch_first=True, bidirectional=False)

        self.hid2mean	= nn.Linear((self.hidden_m+self.hidden_c)*self.Bi*self.beat, (self.hidden_m+self.hidden_c))
        self.hid2var	= nn.Linear((self.hidden_m+self.hidden_c)*self.Bi*self.beat, (self.hidden_m+self.hidden_c))
        
        self.lat2hidm	= nn.Linear((self.hidden_m+self.hidden_c)+12*16+int(48/self.Res)*int(64/self.Res), self.hidden_m*self.Bi*self.beat)
        self.lat2hidc	= nn.Linear((self.hidden_m+self.hidden_c)+12*16+int(48/self.Res)*int(64/self.Res), self.hidden_c*self.Bi*self.beat)
        
        self.outm		= nn.Linear(self.hidden_m*self.Bi*self.beat, 800)
        self.outc		= nn.Linear(self.hidden_c*self.Bi*self.beat, 48)

    def low_res(self, m, r):
        t = int(m.shape[2]/200)
        m = m[:,:,:-t*4]
        m = m.reshape(m.shape[0], m.shape[1]*t*4, 49)
        r = r
        out = torch.zeros(m.shape[0], m.shape[1]/r, (m.shape[2]-1)/r).cuda()
        for i in range(out.shape[1]):
            tmp = torch.sum(m[:,i*r:(i+1)*r,:], 1)
            for j in range(out.shape[2]):
                out[:,i,j] = torch.sum(tmp[:,j*r:(j+1)*r], 1)

        # out = (out>0).type(torch.cuda.FloatTensor)
        # print(out.shape)
        return out

    def encode(self, m, c):
        batch_size = m.shape[0]

        m, _ = self.BGRUm(m)
        c, _ = self.BGRUc(c)

        h1 = m.contiguous().view(batch_size, self.hidden_m*self.Bi*self.beat)
        h2 = c.contiguous().view(batch_size, self.hidden_c*self.Bi*self.beat)
        h = torch.cat([h1, h2], 1)

        mu = self.hid2mean(h)
        var = self.hid2var(h)

        return mu, var

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.05).exp_()
        eps = torch.randn(mu.shape).cuda()
        z = eps * std
        return z

    def decode(self, z):
        melody = torch.zeros((z.shape[0], self.beat, 800)).cuda()
        
        m = self.lat2hidm(z)
        m = m.view(m.shape[0], 1, m.shape[1])
        hn = torch.zeros(1, m.shape[0], self.hidden_m*self.Bi*self.beat).cuda() # init hidden
        for i in range(4):
            m, hn = self.BGRUm2(m, hn)
            melody[:,i,:] = F.sigmoid(self.outm(m[:,0,:]))

        chord = torch.zeros((z.shape[0], self.beat, 48)).cuda()
        
        c = self.lat2hidc(z)
        c = c.view(c.shape[0], 1, c.shape[1])
        hn = torch.zeros(1, c.shape[0], self.hidden_c*self.Bi*self.beat).cuda() # init hidden
        for i in range(4):
            c, hn = self.BGRUc2(c, hn)
            chord[:,i,:] = F.sigmoid(self.outc(c[:,0,:]))

        return melody, chord

    def forward(self, m, c):
        mu, logvar = self.encode(m.view(-1, self.beat, 800), c.view(-1, self.beat, 48))
        ### low resolution
        low_m = self.low_res(m, self.Res)

        ### reparameter
        z = self.reparameterize(mu, logvar)		
        
        ### random
        # z = torch.randn(m.shape[0], self.hidden_m+self.hidden_c).cuda()
        
        z = torch.cat([z, low_m.view(-1, low_m.shape[1]*low_m.shape[2]), c.view(-1, self.beat*48)], 1)
        
        m, c = self.decode(z)
        return m, c, mu, logvar

    def sampling(self, m, c):
        ### encode
        mu, logvar = self.encode(m.view(-1, self.beat, 800), c.view(-1, self.beat, 48))

        ### low resolution
        low_m = self.low_res(m, self.Res)
        print(np.shape(low_m))
        low_m = torch.cat([low_m[200:],low_m[0:200]],0)
        print(np.shape(low_m))
        ### reparameter: only take the mean from encoder
        std = logvar.mul(0.05).exp_()
        eps = torch.randn(mu.shape).cuda()
        # z = eps * std + mu
        z = eps * std

        ### random
        # z = torch.randn(m.shape[0], self.hidden_m+self.hidden_c).cuda()
        z = torch.cat([z, low_m.view(-1, low_m.shape[1]*low_m.shape[2]), c.view(-1, self.beat*48)], 1)
        
        ### decode	
        m, c = self.decode(z)
        return m, c, mu, logvar

def loss_vae(predict_m, test_m, predict_c, test_c, mu, logvar):
	# loss	
	BCEm = F.binary_cross_entropy(predict_m.view(-1, bar*4*200), test_m.view(-1, bar*4*200), size_average=False)
	BCEc = F.binary_cross_entropy(predict_c.view(-1, bar*4*12), test_c.view(-1, bar*4*12), size_average=False)
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

history_train = [[] for i in range(4)]
history_test = [[] for i in range(4)]

def train_vae(epoch):
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

		# if batch_idx % 10 == 0:
		# 	print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
  #               epoch, batch_idx * len(batch_x), len(train_loader.dataset),
  #               100. * batch_idx / len(train_loader),
  #               loss.item() / len(batch_x)))

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

def test_vae(epoch):
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

	test_loss /= len(test_loader.dataset)
	test_accm /= len(test_loader.dataset)
	test_accmr /= len(test_loader.dataset)
	test_accc /= len(test_loader.dataset)
	history_test[0] += [test_loss]
	history_test[1] += [test_accm]
	history_test[2] += [test_accmr]
	history_test[3] += [test_accc]
	print('====> Test set loss: {:.4f} Acc: {:.4f}, {:.4f}, {:.4f}'.format(test_loss,           
		test_accm, test_accmr, test_accc))

def sample(model, x, y, theta):
    TOTAL_BAR = 4
    if use_cuda:
        m, c, mu, var = model.sampling(x.cuda(), y.cuda())
    else:
        m, c, mu, var = model.sampling(x, y)
    m = m.cpu().detach().numpy() # [8, 4, 800]: bar level
    c = c.cpu().detach().numpy() # [8, 4, 48]: bar level

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

    # m = np.concatenate([x,m],0)
    # c = np.concatenate([y,c],0)
    
    m = np.concatenate([x[0:4],m[0:4]],0)
    c = np.concatenate([y[0:4],c[0:4]],0)
    # m_seq, c_seq = numpy2seq(m[0:TOTAL_LEN].reshape((TOTAL_LEN*16,200)), c[0:TOTAL_LEN].reshape((TOTAL_LEN*16,12)), theta)		
    # tempo_seq = np.linspace(tempo[0], tempo[1], num=(interp_num+2)).round().astype(int)
    # m_roll, c_roll = numpy2pianoroll(m[0:TOTAL_LEN].reshape((TOTAL_LEN*16,200)), c[0:TOTAL_LEN].reshape((TOTAL_LEN*16,12)))		
    numpy2midi(m[0:TOTAL_BAR*2].reshape((TOTAL_BAR*2*16,200)), c[0:TOTAL_BAR*2].reshape((TOTAL_BAR*2*16,12)),theta, './output/'+'test.mid')
        
    # midi2pianoroll('./interp_output/'+filename1+'2'+filename2)
    # return m_seq, c_seq # m_seq(n, 4, 48), c_seq(n, 4, 4), tempo_seq(n)

### Load training dataset
print('Load training dataset')
bar = 4
# train_m, train_c, test_m, test_c = parse_data(bar)

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

### Load model
print('Load model')
model = VAE(hidden_m=256, hidden_c=48, bar=bar)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
if torch.cuda.is_available():
	model.cuda()

model.load_state_dict(torch.load("./presets/vaernn_con4bar120.pt")) # no dense, ratio4
# model.load_state_dict(torch.load("./presets/vaernn_con4bar_r8100.pt"))

# for epoch in range(200):
# 	train_vae(epoch)
# 	test_vae(epoch)
# 	if (epoch+1) % 20 == 0:
# 		print("epoch", epoch+1, "saving model.")
# 		torch.save(model.state_dict(), "./presets/vaernn_con4bar_r8"+str(epoch+1)+".pt")

### sample 
test_m, test_c = midi2numpy('./songs/eternal_economy.midi', bar)
test_m = torch.from_numpy(test_m).type(torch.FloatTensor)
test_c = torch.from_numpy(test_c).type(torch.FloatTensor)
sample(model, test_m.cuda(), test_c.cuda(), 0.9)




# m, c, mu, var = model(test_m.cuda(), test_c.cuda())
# m = m.cpu().detach().numpy()
# c = c.cpu().detach().numpy()

# mr = m[:,:,-16:]
# m = m[:,:,:-16]
# m = m.reshape(m.shape[0], m.shape[1]*4, int(m.shape[2]/4))
# mr = mr.reshape(mr.shape[0], mr.shape[1]*4, int(mr.shape[2]/4))
# m = np.concatenate([m, mr], 2)

# bar = 16
# numpy2midi(m[0:bar].reshape((bar*16,200)), c[0:bar].reshape((bar*16,12)), "./output/out1.mid")



