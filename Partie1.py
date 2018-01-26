import numpy as np
import matplotlib.pyplot as plt
# les arrays sont batis avec les dimensions suivantes:
# pluie, arroseur, watson, holmes
# et chaque dimension : faux, vrai

prob_pluie=np.array([0.8,0.2]).reshape(2,1,1,1)

print("Pr(Pluie)=\n" + str(np.squeeze(prob_pluie)))


prob_arroseur=np.array([0.9,0.1]).reshape(1,2,1,1)

print("Pr(Arroseur)=\n" + str(np.squeeze(prob_arroseur)))



watson=np.array([[0.8,0.2],[0,1]]).reshape(2,1,2,1)

print("Pr(Watson|Pluie)=\n" + str(np.squeeze(watson)))

# TODO: determiner facteurs Holmes 

#holmes=np.array([[0.8,0.2],[0,1]]).reshape(2,2,1,2)

#print("Pr(Holmes|Pluie,arroseur)={}\n".format(np.squeeze(holmes)))


watson[0,:,1,:]
(watson*prob_pluie).sum(0).squeeze()[1]
#holmes[0,1,0,1]
