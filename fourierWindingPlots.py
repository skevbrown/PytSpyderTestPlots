# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
#print(sys.path)



import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
import math
from scipy import ndimage as ndi
from pandas import DataFrame
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB


def rect(r, theta):
    """theta in radians

    returns tuple; (float, float); (x,y)
    """
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x,y

def polar(x, y):
    """returns r, theta(radians)
    """
    r, theta = cm.polar( complex(x,y) );

    return r, theta



polarvec = np.vectorize(polar); # Vectorize these func's to operate on vectors
rectvec  = np.vectorize(rect)



vecFreq = 2
tiVec = np.arange(0,13*np.pi,13*np.pi/1000)
lenTi = len(tiVec)
ampVec1 = (1/2)* (np.cos(vecFreq*tiVec)+1)
compExp = np.array( [ complex(0,0)] * lenTi )

plt.plot(tiVec,ampVec1,'r',linewidth=0.3)
plt.grid()
#plt.gca().set_aspect('equal', adjustable='box'); plt.draw();
plt.show()
#print(dir(plt.gca()))

#plt.figure()
ceFreq = 8
compExp.real = np.cos(ceFreq*vecFreq*tiVec)
compExp.imag = np.sin(ceFreq*vecFreq*tiVec)

[compMag, compAng] = polarvec(compExp.real,compExp.imag); compMag = compMag * ampVec1;
[compExp.real, compExp.imag ] = rectvec(compMag,compAng )

from shapely.geometry import Polygon
compEmatr = np.zeros([len(compExp),2])
compEmatr[:,0] = compExp.real; compEmatr[:,1] = compExp.imag
#print(compEmatr)
polygon = Polygon(compEmatr)
cent = polygon.centroid.xy
centx = cent[0][0]; centy = cent[1][0]

print("Centroid: {} {}".format(centx,centy))
#print("Polygon area: {}".format(polygon.area) )
#print("Polygon length: {}".format(polygon.length) )


plt.plot(centx,centy,'ro');
plt.plot(compExp.real,compExp.imag,'b',linewidth=0.2)
plt.gca().set_xlim(-1.5,1.5); plt.gca().set_ylim(-1.5,1.5); 
plt.gca().set_aspect('equal', adjustable='box'); plt.grid(); plt.draw();
#print( dir(compExp) )
plt.show();

