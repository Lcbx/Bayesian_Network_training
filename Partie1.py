import numpy as np
import matplotlib.pyplot as plt






# les arrays sont batis avec les dimensions suivantes:
# pluie, arroseur, watson, holmes
# et chaque dimension : faux, vrai

prob_pluie=np.array([0.8,0.2]).reshape(2,1,1,1)

print("Pr(Pluie)=\n" + str(np.squeeze(prob_pluie)))

prob_arroseur=np.array([0.9,0.1]).reshape(1,2,1,1)

print("Pr(Arroseur)=\n" + str(np.squeeze(prob_arroseur)))

print

watson=np.array([[0.8,0.2],[0,1]]).reshape(2,1,2,1)

print("Pr(Watson | Pluie)=\n" + str(np.squeeze(watson)))

print

# TODO: determiner facteurs Holmes 

holmes=np.array([[[1,0],[0.1,0.9]],[[0,1],[0,1]]]).reshape(2,2,1,2)

print("Pr(Holmes | Pluie, arroseur)=\n" + str(np.squeeze(holmes)))

print

# Pr(H=1)
prH = np.squeeze((holmes * prob_pluie * prob_arroseur) .sum(0).sum(0) )[1]
print("Pr(Holmes=1)= " + str(prH))

print

# Pr(A=1)
prA = prob_arroseur[:,1,:,:]
# Pr(A=1|H=1) = Pr(H=1|A=1) * Pr(A=1) / Pr(H=1)
prAH = np.squeeze( np.squeeze(( holmes * prob_pluie ) .sum(0)[1])[1]  *prA/prH )
print("Pr(Arroseur=1 | Holmes=1) = " + str(np.squeeze( prAH ) ))

print

# Pr(A = 1|H = 1, W = 1)
print("Pr(Arroseur=1 | Holmes=1, Watson =1) = " + str(prAH) )

