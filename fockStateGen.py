# Simulation based on Engelkemeier et. al PHYSICAL REVIEW A 102, 023712 (2020)
# Density matrix and measurement operator can be represented as a vector with
# with the coeff in the first element and variable in the second element
# (Pk, xk) = sum(Pk * E(xk)) = rho (Eqn 15)
# We can do the same for Measurement operator = sum(Q_l * E(omega_l))

# This version produces results which fit the paper

import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt
import scipy.special as sp
from matplotlib.widgets import Slider, Button, RadioButtons

random.seed(654)
def IsNPArray(arr):
    # Check if array is np.ndarray
    # If not true, then cast arr to np array.
    if isinstance(arr, (int,float)):
        return np.array([arr])
    else:
        return np.array(arr)

def initDensityMatrix(numTerms):
    # Generating random coeffs such that they sum to unity
    P = np.sort(random.dirichlet(np.ones(numTerms)))[::-1]
    x = random.uniform(size = numTerms)
    return np.stack((P,x), axis=1)

def gainFactor(squeezeParam):
    return np.cosh(squeezeParam)**2

def loopConfigOutput(inputRho, heraldRho, squeezeParam):
    # From Eqn 31
    # squeezeParam is an arrray
    # The first two index of the output matrix represents the resulting density
    # matrix, the third index represents the squeezeParam
    squeezeParam = IsNPArray(squeezeParam)
    inputSize = inputRho.shape # (NUMSAMP, k, 2)
    heraldSize = heraldRho.shape
    numsamp = len(squeezeParam)
    gamma = gainFactor(squeezeParam)
    pre = np.einsum('ij,ik,i->ijk', inputRho[:,:,0], heraldRho[:,:,0], 1/gamma)

    # init variable
    v = lambda xk, zl, gamma: (xk + (gamma-1)*zl)/gamma
    var = np.array([[[v(x, z, gamma[i])
                        for z in heraldRho[i,:,1]]
                        for x in inputRho[i,:,1]]
                        for i in np.arange(numsamp)])

    prefactor = pre.reshape(numsamp, pre.shape[1]*pre.shape[2])
    variable = var.reshape(numsamp, var.shape[1]*var.shape[2])
    return np.stack((prefactor, variable), axis = 2)

def outputAfterTRoundTrip(inputRho, heraldRho, squeezeParam, T):
    # instead of calling loopConfigOutput in main script, user just have to call
    # this function once and the output will be in the form of dictionary.
    squeezeParam = IsNPArray(squeezeParam)
    rho1R = loopConfigOutput(inputRho, heraldRho, squeezeParam)
    rhoOut = {'rho1R':rho1R}
    hash = 'rho{:0.0f}R'
    if T == 1:
        return rhoOut
    for i in np.arange(2,T+1):
        rhoOut[hash.format(i)] = \
        loopConfigOutput(rhoOut[hash.format(i-1)], heraldRho, squeezeParam)
    return rhoOut

def expectationValue(rho, M):
    # the trace of operator acting on a density matric can evaluated with Eqn 20,
    # sum_(k,l)(Pk*Ql/(1-xk*wl))
    PQ = np.einsum('ij,k->ijk',rho[:,:,0], M[:,0])
    xw = np.einsum('ij,k->ijk',rho[:,:,1], M[:,1])
    expVal = np.einsum('ijk,ijk->i', PQ, 1/(1-xw))
    return expVal

def successProb(inputRho, outputRho):
    # Success probability = tr(rho_out)/tr(rho_in) =
    # (rho_out, identity)/(rho_in, identity)
    # (rho, operator) defines the inn-product type functional
    id = np.array([[1,1]])
    successProb = expectationValue(outputRho, id)/expectationValue(inputRho, id)
    return successProb

def fidelity(rho, n):
    # Eqn 17 sum(Pk*xk^n)
    return np.einsum('ij,ij->i',rho[:,:,0], rho[:,:,1]**n)

def lossChannel(rho, quantumEff):
    # Eqn 27
    prefactor = rho[:,:,0]/((1-(1-quantumEff)*rho[:,:,1]))
    variable = quantumEff*rho[:,:,1]/((1-(1-quantumEff)*rho[:,:,1]))
    return np.stack((prefactor, variable), axis=1)

def psuedoPNR(numDetector, kClicks, detectorEff):
    # Generate psuedo PNR measurement operator given by eqn 21
    # kClicks = num of clicks corresponding to photon-number state
    J = np.arange(1,kClicks+1)
    prefactor = sp.binom(numDetector, kClicks)*\
                sp.binom(kClicks, J)*(-1)**(kClicks-J)

    variable = J/numDetector
    detectorPOVM = np.stack((prefactor, variable), axis=1)
    return detectorPOVM

def getSqueezeParam(pumpPower, beamRad, xEff, refractiveIdx, crystalLen, \
pumpWavelength):
    epsilon = 8.8541878128E-12
    C = 299792458
    pumpIntensity = pumpPower/(np.pi*beamRad**2)
    fieldAmp = np.sqrt(pumpIntensity/(2*refractiveIdx*epsilon*C))
    angFreq = (2*np.pi)*(C/pumpWavelength)
    return ((xEff*angFreq)/(refractiveIdx*C))*fieldAmp*crystalLen

###############################################################################
# pump = np.linspace(50, 2000, 20)*1E-3 #W
# plt.plot(pump*1E3, getSqueezeParam(pump, 150E-6, 14E-12, 1.8, 10E-3, 775E-9),
#         '.')
# plt.xlabel('Pump Power (mW)')
# plt.ylabel('$|\zeta|$', fontsize = 20)
# plt.savefig('.\data\zetaVsPump.eps', format='eps', transparent=True)
# plt.show()
