from ase.io import read,write
from ase import Atoms,Atom
from subprocess import call
import numpy as np
import os

def regbuild_pair(pair,rcut,rmin=0.0,size=0.05):
    #lengths = np.linspace(rmin,rcut,size)
    lengths = np.arange(rmin,rcut+size,size)
    for il, length in enumerate(lengths):
        atoms = Atoms(list(pair))
        atoms.set_positions([[0.,0.,0.],[0.,0.,length]])
        atoms.set_cell(np.eye(3) * (2.1*rcut))
        atoms.center()
        write('incr_%03d_%s-%s.xyz' %( (il,) + pair),atoms)
        call('python convert_all.py',shell = True)
        if not os.path.isdir('INCR_%s-%s_%03d' %(pair + (il,))):
            os.mkdir('INCR_%s-%s_%03d' % (pair + (il,)))
        #call('mv incr_%03d_%s-%s.xyz ./INCR_%s-%s_%03d' % ( (il,) + pair + pair + (il,) ), shell=True)
        call('mv bad_incr_%03d_%s-%s.json ./INCR_%s-%s_%03d' % ( (il,) + pair + pair + (il,) ), shell=True)

        
    

#rcut = 6.026
#rmin = 1.6
#regbuild_pair(('Ta','Ta'),rcut,rmin)
"""
rcutfac = 5.742
lambda = 1.723
rcinner = 1.595
"""
rcutstr = '5.742 ' #cutoff per bond type
rminstr = '1.595 ' #inner cutoff per bond type
rcuts = [float(k) for k in rcutstr.split()]
rmins = [float(k) for k in rminstr.split()]
#elems = ['H','O']
elems = ['Ta']
import itertools
pairs = [p for p in itertools.product(elems,elems)]
for ip,pair in enumerate(pairs):
    rcut = rcuts[ip]
    rmin = rmins[ip]
    print (pair,rcut,rmin)
    regbuild_pair(pair,rcut,rmin)

