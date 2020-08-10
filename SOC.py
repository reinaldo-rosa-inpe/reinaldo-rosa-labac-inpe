# -*- coding: utf-8 -*-
"""
Matemática Computacional I
Lista de Exercícios (10/10)

3.2 - Self-Organized Criticality (SOC)
        i) Calcule a Taxa Local de Flutuação [ϒi] para cada valor da ST
        ii) Calcule P[ϒi] = counts(ni) / N
        iii) Plot logP[ϒi] x log ni (e ajuste uma lei de potencia).

10.1. Implemente um algoritmo em Python para caracterização de SOC a partir de uma ST.

10.2. Aplique o SOC.py para todas as ST do exercício 6.1.

--------------------------------------
Principais referências:        
    https://www.sciencedirect.com/science/article/pii/S0273117701006135
    http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0103-97332000000100004
    
----------------------------------------
Version by Willian Vieira de Oliveira, 
Camila, Felipe, Adriano, Helvecio, e Paulo.
09/06/19
Python: 3.7.3
----------------------------------------
"""

import numpy as np
import pylab as plt

def SOC(data, n_bins=50):
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    #print("mean: ", mean, " var: ", var)
    """ Computa a Taxa Local de Flutuação para cada valor da ST """
    Gamma = []
    for i in range(0,n):
        #Gamma.append((data[i] - mean)/var)
        Gamma.append((data[i] - mean)/std)

    """ Computa P[Psi_i] """
    # Retorna o número de elementos em cada bin, bem como os delimitares dos bins
    counts, bins = np.histogram(Gamma, n_bins)
    Prob_Gamma = []
    for i in range(0, n_bins):
        Prob_Gamma.append(counts[i]/n)
    #plt.plot(Gamma)
    return Prob_Gamma, counts

""" Inicialização """
nomeArquivo = './S7.csv'
data = np.genfromtxt(nomeArquivo, delimiter = ' ', dtype = 'float32',filling_values = 0)

""" Caracterização de SOC a partir da série """
Prob_Gamma, counts = SOC(data)


""" Ajustando uma lei de potência (y = b * 10^(x*a))"""

x = np.linspace(1, len(counts), len(counts))

log_Prob = np.log10(Prob_Gamma)
log_counts = np.log10(counts)
#coef = np.polyfit(log_Prob,log_counts, 1)

# Parte criada para remover os índices correspondentes a valores zeros e definir
#   dois pontos (neste caso, mínimo e máximo) para ajustar a lei de potencia.
p = np.array(Prob_Gamma)
p = p[np.nonzero(p)]
c = counts[np.nonzero(counts)]
log_p = np.log10(p)
log_c = np.log10(c)

a = (log_p[np.argmax(c)] - log_p[np.argmin(c)]) / (np.max(c) - np.min(c))
b = log_Prob[0]
y = b * np.power(10, (a*counts))

""" Plotagem """
plt.clf()
plt.scatter(np.log10(counts), y, marker=".", color="blue")

plt.title('SOC', fontsize = 16)
plt.plt.xlabel('log(ni)')
plt.plt.ylabel('log(Yi)')
plt.plt.grid()

plt.savefig('s7plot_novo.pdf')

plt.show()



