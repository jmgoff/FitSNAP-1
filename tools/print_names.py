from ase.io import read,write
from ase import Atoms,Atom
from subprocess import call
import numpy as np
import os

def print_names(pair,rcut,rmin=0.0,size=0.05):
    #lengths = np.linspace(rmin,rcut,size)
    lengths = np.arange(rmin,rcut+size,size)
    names = []
    for il, length in enumerate(lengths):
        print('INCR_%s-%s_%03d' %(pair + (il,)), '   =    0.0     1.0     1.0       1.0      1.E-12 ')
        names.append('INCR_%s-%s_%03d' %(pair + (il,)))
    print (names)
    print (' '.join(names))
        
    

#rcut = 6.026
#rmin = 1.6
#print_names(('Ta','Ta'),rcut,rmin)

#rcutstr = '2.383  3.221  3.221  4.058' #cutoff per bond type
#rminstr = '0.4  0.6  0.6  0.8' #inner cutoff per bond type
#rcuts = [float(k) for k in rcutstr.split()]
#rmins = [float(k) for k in rminstr.split()]
#elems = ['H','O']

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
    print_names(pair,rcut,rmin)

