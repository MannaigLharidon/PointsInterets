# -*- coding: utf-8 -*-
"""
Created on Tuesday Mars  6  2018

@author: Mannaig L'Haridon et Habib Doucouré
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage

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
    Fonction calculant les dérivées d'une image par un filtre 
    """
    Ix = ndimage.filters.convolve(Img,filtre)
    Iy = ndimage.filters.convolve(Img,np.transpose(filtre))
    return Ix,Iy


def gradient(Ix,Iy):
    """
    Fonction calculant le module du gradient
    """
    G_I = np.sqrt(Ix**2 + Iy**2)
    return G_I


def theta(Ix,Iy,L,C):
    """
    Fonction calculant la direction du gradient
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
    Fonction déterminant les pixels se trouvant dans un rayon R au pixel de coordonnees (0,0)
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

# Pondération de la paire de pixels
def GWF(pi,pj):
    gwf = np.log(1+G_I[pi[0],pi[1]])*np.log(1+G_I[pj[0],pj[1]])
    return gwf

# Phase Weight Function
def PWF(pi,pj):
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

# Valeur de la symetrie pour un pixel
def symetrie(l,c,gamma_R):
    s = 0
    taille = int(np.size(gamma_R)/2)
    for i in range(taille):
        if i%2 ==0:
            # Translation d'un vecteur (l,c)
            pi = np.add(gamma_R[i],[l,c])
            pj = np.add(gamma_R[i+1],[l,c])
            s += PWF(pi,pj) * GWF(pi,pj)
    return s
        
# Convoluer la carte de symetrie avec une gaussienne d'écart-type sigma
def convGauss(carteSym,sigma):
    """
    Finir cette partie : gaussienne et faire tourner le programme
    """
    gaussienne = 0
    #filtrage par une gaussienne
    conv = ndimage.filters.gaussian(gaussienne,S)
    return conv


if __name__ == "__main__" :
    
    chemin = './'
    I = io.imread('image027.ssech4.tif')
    plt.imshow(I)
    L,C = np.shape(I)
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sigma = 0.5
    
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
            
    conv = convGauss(S,sigma)







#Detecter les maxima locaux de R superieurs a un certain seuil
#seuil = 




########## Exportation des points ##########
#Format : une ligne par point detecte avec ses coordonnes (colonne, ligne)

#Creation d'un fichier texte
#fichier = open(chemin,'w')
#fichier.write(str(val1)+" "+str(val2)+"\n")

        



