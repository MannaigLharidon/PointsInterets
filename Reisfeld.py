# -*- coding: utf-8 -*-
"""
Created on Tuesday Mars  6  2018

@author: Mannaig L'Haridon et Habib Doucouré
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage
from scipy.ndimage import gaussian_filter


"""
#############################################################################
#                                                                           #
#                   TP : EXTRACTION DE POINTS D'INTERETS                    #
#      Partie I : REISFELD (traitement des images en niveaux de gris)       #
#                                                                           #
#############################################################################
"""


####################  CALCULS PRELIMINAIRES  ####################

def derivees(Img,filtre):
    """
    Calcul des dérivées d'une image par un filtre 
    """
    Ix = ndimage.filters.convolve(Img,filtre)
    Iy = ndimage.filters.convolve(Img,np.transpose(filtre))
    return Ix,Iy


def gradient(Ix,Iy):
    """
    Calcul du module du gradient
    """
    G_I = np.sqrt(Ix**2 + Iy**2)
    return G_I


def theta(Ix,Iy,L,C):
    """
    Calcul de la direction du gradient
    """
    Theta_I = np.zeros((L,C))
    for l in range(L):
        for c in range(C):
            if Iy[l][c]==0:
                Theta_I[l][c]=0
            else:
                Theta_I[l][c] = np.arctan(Ix[l][c]/Iy[l][c])
    return Theta_I


def gamma(R):
    """
    Détermination des pixels se trouvant dans un rayon R au pixel de coordonnees (0,0)
    """
    gamma_R = []
    for i in range(-R,R+1):
        for j in range(-R,1):
            if j==0 and i<0:
                continue
            norm = np.sqrt(i**2+j**2)
            if norm<R:
                gamma_R.append([i,j])
                gamma_R.append([-i,-j])
    return gamma_R




####################  CALCUL DE LA CARTE DE SYMETRIE  ####################

def GWF(pi,pj):
    """
    Gradient Weight Function : Pondération de la paire de pixels
    """
    gwf = np.log(1+G_I[pi[0],pi[1]])*np.log(1+G_I[pj[0],pj[1]])
    return gwf


def PWF(pi,pj):
    """
    Phase Weight Function
    """
    if pi[0]==0:
        alpha = 0
    else:
        alpha = np.arctan(pi[1]/pi[0])
    gamma_i = Theta_I[pi[0]][pi[1]]-alpha
    gamma_j = Theta_I[pj[0],pj[1]]-alpha
    pwf_moins = 1-np.cos(gamma_i-gamma_j) 
    pwf_plus = 1-np.cos(gamma_i+gamma_j) 
    pwf = pwf_plus*pwf_moins
    return pwf


def symetrie(l,c,gamma_R):
    """
    Valeur de la symetrie pour un pixel
    """
    s = 0
    taille = int(np.size(gamma_R)/2)
    for i in range(taille):
        if i%2 ==0:
            # Translation d'un vecteur (l,c)
            pi = np.add(gamma_R[i],[l,c])
            pj = np.add(gamma_R[i+1],[l,c])
            s += PWF(pi,pj) * GWF(pi,pj)
    return s




#################### DETERMINATION DES POINTS D'INTERETS ####################        

def convGauss(carteSym,sigma):
    """
    Convolution de la carte de symetrie par une gaussienne
    """
    conv = gaussian_filter(carteSym,sigma)
    return conv


def maxLocaux(conv,seuil,fichier):
    """
    Détection des maximum locaux supérieurs au seuil
    """
    maxL = []
    maxC = []
    for l in range(L):
        for c in range(C):
            if conv[l][c] > seuil:
                maxL.append(l)
                maxC.append(c)
                fichier.write(str(c)+" "+str(l)+"\n")
    MaxL = np.asarray(maxL)
    MaxC = np.asarray(maxC)
    return MaxL,MaxC





if __name__ == "__main__" :
    
    chemin = './'
    I = io.imread('image027.ssech4.tif')
    plt.imshow(I,cmap='gray')
    L,C = np.shape(I)
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sigma = 2
    
    Ix,Iy = derivees(I,sobel)  #Dérivées calculées avec Sobel
    G_I = gradient(Ix,Iy)
    Theta_I = theta(Ix,Iy,L,C)
    R = 3
    gamma_R = gamma(R)
    
    # Calcul de la carte de symetrie
    S = np.zeros((L,C))
    for l in range(R,L-R+1):
        for c in range(R,C-R+1):
            S[l][c] = symetrie(l,c,gamma_R)
    
    # Creation du fichier d'enregistrement des points d'interets
    fichier = open("pointsInterets.txt","w")
    
    # Detection et enregistrement des points d'interets        
    conv = convGauss(S,5)
    seuil = 100
    maxL, maxC = maxLocaux(conv,seuil,fichier)
    
    plt.imshow(conv,cmap='gray')
    plt.plot(maxC,maxL)
    plt.show()
    


