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
from skimage.feature import peak_local_max

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
    G_I = np.zeros((Ix.shape[0],Ix.shape[1]))
    for l in range(Ix.shape[0]):
        for c in range(Iy.shape[1]):
            G_I[l][c] = np.sqrt(Ix[l][c]**2 + Iy[l][c]**2)
    return G_I


def theta(Ix,Iy):
    """
    Calcul de la direction du gradient
    """
    Theta_I = np.zeros((Ix.shape[0],Ix.shape[1]))
    for l in range(Ix.shape[0]):
        for c in range(Ix.shape[1]):
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

def GWF(G_I,pi,pj):
    """
    Gradient Weight Function : Pondération de la paire de pixels
    """
    gwf = np.log(1+G_I[pi[0],pi[1]])*np.log(1+G_I[pj[0],pj[1]])
    return gwf


def PWF(Theta_I,pi,pj):
    """
    Phase Weight Function
    """
    if pi[1]==0:
        alpha = 0
    else:
        alpha = np.arctan(pi[0]/pi[1])
    gamma_i = Theta_I[pi[0]][pi[1]]-alpha
    gamma_j = Theta_I[pj[0],pj[1]]-alpha
    pwf_moins = 1-np.cos(gamma_i-gamma_j) 
    pwf_plus = 1-np.cos(gamma_i+gamma_j) 
    pwf = pwf_plus*pwf_moins
    return pwf


def symetrie(l,c,gamma_R,G_I,Theta_I):
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
            s += PWF(Theta_I,pi,pj) * GWF(G_I,pi,pj)
    return s




#################### DETERMINATION DES POINTS D'INTERETS ####################        

def convGauss(carteSym,sigma):
    """
    Convolution de la carte de symetrie par une gaussienne
    """
    conv = gaussian_filter(carteSym,sigma)
    return conv


def seuil_loc(max_loc,conv):
    """
    Détermination du seuil moyen des maxima locaux
    """
    taille = max_loc.shape[0]
    seuil = 0
    for t in range(taille):
        seuil += conv[max_loc[t][0]][max_loc[t][1]]
    seuil /= taille
    return seuil


def maxLocaux(max_loc,conv,seuil,fichier):
    """
    Détection des maximum locaux supérieurs au seuil
    """
    maxL = []
    maxC = []
    taille = max_loc.shape[0]
    for t in range(taille):
        if conv[max_loc[t][0]][max_loc[t][1]]>seuil:
            maxL.append(max_loc[t][0])
            maxC.append(max_loc[t][1])
            fichier.write(str(max_loc[t][1])+" "+str(max_loc[t][0])+"\n")
    MaxL = np.asarray(maxL)
    MaxC = np.asarray(maxC)
    return MaxL,MaxC







if __name__ == "__main__" :

    # Lecture de l'image à étudier
    I = io.imread('chat.tif')
    I = I/256
    plt.figure()
    plt.title("Image brute")
    plt.imshow(I,cmap='gray')
    
    # Paramètres
    L,C = np.shape(I)
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sigma = 1
    R = 8
    
    
    Ix,Iy = derivees(I,sobel)  #Dérivées calculées avec Sobel
    G_I = gradient(Ix,Iy)  #Module du gradient
    plt.figure()
    plt.title("Module du gradient")
    plt.imshow(G_I)
    
    Theta_I = theta(Ix,Iy)   #Direction du gradient
    plt.figure()
    plt.title("Direction du gradient")
    plt.imshow(Theta_I)
    
    gamma_R = gamma(R)  #Pixels se trouvant dans un rayon R à un pixel de ref.
    
    # Calcul de la carte de symetrie
    S = np.zeros((L,C))
    for l in range(R,L-R+1):
        for c in range(R,C-R+1):
            S[l][c] = symetrie(l,c,gamma_R,G_I,Theta_I)
    plt.figure()
    plt.title("Carte de symétrie")
    plt.imshow(S)
    
    # Creation du fichier d'enregistrement des points d'interets
    fichier = open("pointsInterets.txt","w")
    
    # Detection et enregistrement des points d'interets        
    conv = convGauss(S,sigma)
    plt.figure()
    plt.title("Image convoluée")
    plt.imshow(conv,cmap='gray')
    
    max_loc = peak_local_max(conv,min_distance=10)
    seuil = seuil_loc(max_loc,conv)
    maxL, maxC = maxLocaux(max_loc,conv,seuil,fichier)
    
    plt.figure()
    plt.title("Points d'intérêts de l'image")
    plt.imshow(I,cmap='gray')
    plt.plot(maxC,maxL,'b+')
    plt.show()
    


