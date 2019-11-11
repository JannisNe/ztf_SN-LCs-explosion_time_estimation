import sncosmo
from numpy import loadtxt

for f in ['g','r','i']:
        dat = loadtxt('/lustre/fs23/group/icecube/necker/ztf_bandpasses/ztf'+f+'.txt')
        band = sncosmo.Bandpass(dat[:,0], dat[:,1], name='ztf'+f)
        sncosmo.registry.register(band)
