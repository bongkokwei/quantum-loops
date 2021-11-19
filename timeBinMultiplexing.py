# Based on Kaneda and Kwiat, Sci. Adv. 2019;5:eaaw8586
# Vectorised version but need minor bug fixes in the future as single photon

import numpy as np
from numpy import inf
import math
import matplotlib.pyplot as plt
import scipy.special as sp
from matplotlib.widgets import Slider, Button, RadioButtons

def binom(n,k):
    return sp.factorial(n)/(sp.factorial(k)*sp.factorial(n-k))

def downConversionProb(mu, k):
    # The probability that an SPDC source generates k-photon pairs in a time bin
    # with a mean photon number mu
    # mu =  mean photon number
    return  (mu**k)/(1+mu)**(k+1)

def triggerDetProbPrimal(eta, k, d):
    prob = 0
    for l in np.arange(1,k+1):
        prob += (eta**l)*(1-eta)**(k-l)*binom(k,l)*(1/d)**(l-1)
    return prob

def triggerDetProb(eta, k, d):
    # Returns an array of trigger-photon detection probability conditioned by
    # k-photon pairs created. k is an array
    # eta = trigger detection efficiency
    # d = number of detectors
    endsum = k[-1]
    kMatrix = np.tril(np.tile(k.reshape(endsum,1), (1,endsum)))
    lMatrix = np.tril(np.tile(k.reshape(1,endsum), (endsum,1)))
    binomTriangle = np.tril(binom(kMatrix,lMatrix))
    prob = (eta**lMatrix)*(1-eta)**(kMatrix-lMatrix)*binomTriangle*(1/d)**(lMatrix-1)
    return np.sum(prob, axis=1)

def MPhotonEmitProb(ti, tDL, M, N, k):
    # M-photon emission probability conditioned by k-photon pair generation
    # in the j-th time bin
    # tDL = net transmission of adjustable delay line
    # ti = net transmission of other optics
    # N = number of time bin (meshgrid(column, row))
    # k = photon pairs
    jGrid, kGrid = np.meshgrid(np.arange(1,N+1), np.arange(1,k+1))
    return  (ti*tDL**(N-jGrid-1))**M*(1-ti*tDL**(N-jGrid-1))**(kGrid-M)*binom(kGrid,M)

def heraldPhotonPairProb(mu, k, eta, d):
    kPhotonPair = np.arange(1,k+1)
    return downConversionProb(mu,kPhotonPair)*triggerDetProb(eta, kPhotonPair ,d)

def multiplexHeraldProb(mu, eta, d, N, endsum = 100):
    # multiplex heralding probability
    # N = number of multiplexed time binom
    # mu =  mean photon number
    # k-photon pairs crea   xxted.
    # eta = trigger detection efficiency
    # d = number of detectors
    j = np.arange(1,N+1)
    heraldProb = np.sum(heraldPhotonPairProb(mu, endsum, eta, d))
    return 1 - (1 - heraldProb)**(N-j)

def multiplexMPhotonProb(mu, eta, d, N, ti, tDL, M, endsum = 100):
    # The probability of producing an M-photon state after time multiplexing
    j = np.arange(1,N+1)
    noTriggerDetect = (1-np.sum(heraldPhotonPairProb(mu, endsum, eta, d)))**(N-j)
    detectTrigger = np.einsum('i, ij',
                            heraldPhotonPairProb(mu, endsum, eta, d),
                            MPhotonEmitProb(ti, tDL, M, N, endsum))
    return np.sum(noTriggerDetect*detectTrigger)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
meanPhotonNum = 0.18 # mu
triggerDetectEff = 0.53 # eta`
numTrigger = 4 #d, 1E8 simulates infinite number of detectors/PNR detectors
numRoundTrip = np.arange(1,51) # N
delayLineTrans = 0.988
opticsTrans = 0.83
MPhotonNum = 1

# Correct ans: P1 = 0.667, PH = 0.98 for mu = 0.18
# print(MPhotonEmitProb(opticsTrans, delayLineTrans, M=1, N=40, k=100)[-1,-1])
# print(multiplexHeraldProb(meanPhotonNum, triggerDetectEff, numTrigger, 40)[0])
# print(multiplexMPhotonProb(meanPhotonNum, triggerDetectEff, numTrigger, 40, opticsTrans, delayLineTrans, M=1))

P1 = [multiplexMPhotonProb(meanPhotonNum, triggerDetectEff, numTrigger,
                          n, opticsTrans, delayLineTrans, MPhotonNum)
     for n in numRoundTrip]

# Start plot
l,  = plt.plot(numRoundTrip, P1, lw=2)
ax.margins(x=0)
ax.set_ylabel('Single photon probability')
ax.set_xlabel('Number of multiplexed time bin')
ax.set_ylim(ymin=0, ymax=1)

startSlider = 0.01
axcolor = 'lightgoldenrodyellow' # Colour option for slider
mu = plt.axes([0.25, startSlider, 0.65, 0.025], facecolor=axcolor)
eta = plt.axes([0.25, startSlider + 0.05, 0.65, 0.025], facecolor=axcolor)
ti = plt.axes([0.25, startSlider+ 0.10, 0.65, 0.025], facecolor=axcolor)
tdl = plt.axes([0.25, startSlider + 0.15, 0.65, 0.025], facecolor=axcolor)

muSlider = Slider(mu, 'Mean Photon Number', 0, 1.00, valinit=meanPhotonNum)
etaSlider = Slider(eta, 'Trigger Transmission', 0, 1.00, valinit=triggerDetectEff)
tiSlider = Slider(ti, 'Optical Transmission', 0, 1.00, valinit=opticsTrans)
tdlSlider = Slider(tdl, 'Delay line Transmission', 0, 1.00, valinit=delayLineTrans)

def update(val):
    mu = muSlider.val
    eta = etaSlider.val
    ti = tiSlider.val
    tdl = tdlSlider.val
    l.set_ydata([multiplexMPhotonProb(mu, eta, numTrigger, n, ti, tdl, MPhotonNum)
                 for n in numRoundTrip])
    fig.canvas.draw_idle()

# Update values when slider moves
muSlider.on_changed(update)
etaSlider.on_changed(update)
tiSlider.on_changed(update)
tdlSlider.on_changed(update)

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

plt.show()
