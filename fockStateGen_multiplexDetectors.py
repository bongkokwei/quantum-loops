from fockStateGen import *

NUMSAMP = 200
detectorEff = 0.1
numDetector = 2
numClicks = 1
numBins = 10

photonNum = np.arange(numBins+1)
squeezeParam = np.arccosh(np.sqrt(1.04))

def genPhotonNumDistro(photonNum, numDetector, numClicks, squeezeParam, T):
    # Input state
    rhoInput = np.array([[1.00, 0.00]])
    rho0R = np.tile(rhoInput, (1,1,1))
    hash = 'rho{:0.00f}R'

    # Detectos POVM
    PNR = psuedoPNR(numDetector, numClicks, detectorEff)
    rhoHeraldTile = np.tile(PNR, (1,1,1))

    # Loop action
    rhoOut = \
    outputAfterTRoundTrip(rho0R, rhoHeraldTile, squeezeParam, T)
    rho = rhoOut[hash.format(T)]

    # sp = (rho, id), since (rho0R, id) = 1
    sp = successProb(rho0R, rho)
    photonNumDistro = \
    np.array([fidelity(rho, x)/sp for x in photonNum])

    return photonNumDistro.reshape(len(photonNum),)

###############################################################################
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

photonDist = genPhotonNumDistro(photonNum, numDetector, numClicks, squeezeParam, 2)
bars = plt.bar(photonNum, photonDist)
ax.margins(x=0)
ax.set_ylabel('probability')
ax.set_xlabel('photon number')
ax.set_ylim(ymin=0, ymax=1)
plt.xticks(np.arange(numBins+1))

startSlider = 0.01
axcolor = 'lightgoldenrodyellow' # Colour option for slider
N = plt.axes([0.25, 0.01, 0.65, 0.025], facecolor=axcolor)
K = plt.axes([0.25, 0.06, 0.65, 0.025], facecolor=axcolor)
Z = plt.axes([0.25, 0.11, 0.65, 0.025], facecolor=axcolor)
T = plt.axes([0.25, 0.16, 0.65, 0.025], facecolor=axcolor)

NSlider = Slider(N, 'Number of Detectors', 0, 10, valinit=numDetector, valfmt='%0.0f')
KSlider = Slider(K, 'Number of Clicks', 0, 10, valinit=numClicks, valfmt='%0.0f')
ZSlider = Slider(Z, '$\zeta$', 0.005, 0.5, valinit=squeezeParam)
TSlider = Slider(T, 'Number of Round Trip', 0, 10, valinit=2, valfmt='%0.0f')

def update(val):
    N = round(NSlider.val)
    K = round(KSlider.val)
    Z = ZSlider.val
    T = round(TSlider.val)
    pDist = genPhotonNumDistro(photonNum, N, K, Z, T)
    for x in photonNum:
        bars[x].set_height(pDist[x])
    fig.canvas.draw_idle()

# Update values when slider moves
NSlider.on_changed(update)
KSlider.on_changed(update)
ZSlider.on_changed(update)
TSlider.on_changed(update)

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

plt.show()
