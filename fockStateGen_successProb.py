from fockStateGen import *

# init constants
NUMSAMP = 200
detectorEff = 0.8
numDetector = 4
numClicks = 1
# input density matrix
rhoInput = np.array([[1.00, 0.00]])
# Tiling the input density matrix to match required input
rho0R = np.tile(rhoInput, (NUMSAMP,1,1))

# on-off detector POVM [Eq5]
rhoHerald = np.array([[0.9999,0.9999],[-0.9999,1-detectorEff]])
# rhoHeraldTile = np.tile(rhoHerald, (NUMSAMP,1,1))
PNR = psuedoPNR(numDetector, numClicks, detectorEff)
rhoHeraldTile = np.tile(PNR, (NUMSAMP,1,1))

# init squeezing parameter
# squeezeParam = np.sqrt(np.linspace(0.005, 0.5, NUMSAMP))
# modSquareSP = np.linspace(0.005, 0.5, NUMSAMP)

squeezeParam = np.exp(\
np.linspace(np.log(np.sqrt(0.005)), np.log(np.sqrt(0.5)), NUMSAMP))
modSquareSP = squeezeParam**2
id = np.array([[1,1]])

rhoOut = outputAfterTRoundTrip(rho0R, rhoHeraldTile, squeezeParam, 4)

rho1R = rhoOut['rho1R']
rho2R = rhoOut['rho2R']
rho3R = rhoOut['rho3R']
rho4R = rhoOut['rho4R']

# the success probability is given by eqn 32, tr(rho_out)/tr(rho_in)
# tr(rho_in) = (rho_in, id) = 1
sp01 = successProb(rho0R, rho1R)
sp02 = successProb(rho0R, rho2R)
sp03 = successProb(rho0R, rho3R)
sp04 = successProb(rho0R, rho4R)

fid1R = fidelity(rho1R,1)/sp01
fid2R = fidelity(rho2R,2)/sp02
fid3R = fidelity(rho3R,3)/sp03
fid4R = fidelity(rho4R,4)/sp04

###############################################################################
# According to FIG 3, when modSquareSP = 0.01 and T = 1, P = 0.01
plt.figure(1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Squeezing parameter $(|\zeta|^2)$", fontsize=15)
plt.ylabel("$\mathcal{P}$", fontsize=20)
plt.annotate("$T=1$", # this is the text
             (modSquareSP[0],sp01[0]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(-10,0), # distance from text to points (x,y)
             ha='right', # horizontal alignment can be left, right or center
             fontsize=12)
plt.annotate("$T=2$", # this is the text
             (modSquareSP[0],sp02[0]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(-10,0), # distance from text to points (x,y)
             ha='right', # horizontal alignment can be left, right or center
             fontsize=12)
plt.annotate("$T=3$", # this is the text
             (modSquareSP[0],sp03[0]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(-10,0), # distance from text to points (x,y)
             ha='right', # horizontal alignment can be left, right or center
             fontsize=12)
plt.annotate("$T=4$", # this is the text
             (modSquareSP[0],sp04[0]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(-10,0), # distance from text to points (x,y)
             ha='right', # horizontal alignment can be left, right or center
             fontsize=12)
plt.plot(modSquareSP, sp01)
plt.plot(modSquareSP, sp02)
plt.plot(modSquareSP, sp03)
plt.plot(modSquareSP, sp04)
# for 'TkAgg' backend to maximised plot windows
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

# plt.savefig('probability.png', format='png', transparent=True)

plt.figure(2)
plt.xscale('log')
plt.xlabel("Squeezing parameter $(|\zeta|^2)$", fontsize=15)
plt.ylabel("$\mathcal{F}$", fontsize=20)
plt.annotate("$T=1$", # this is the text
             (modSquareSP[-1],fid1R[-1]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(10,0), # distance from text to points (x,y)
             ha='left', # horizontal alignment can be left, right or center
             fontsize=12)
plt.annotate("$T=2$", # this is the text
             (modSquareSP[-1],fid2R[-1]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(10,0), # distance from text to points (x,y)
             ha='left', # horizontal alignment can be left, right or center
             fontsize=12)
plt.annotate("$T=3$", # this is the text
             (modSquareSP[-1],fid3R[-1]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(10,0), # distance from text to points (x,y)
             ha='left', # horizontal alignment can be left, right or center
             fontsize=12)
plt.annotate("$T=4$", # this is the text
             (modSquareSP[-1],fid4R[-1]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(10,0), # distance from text to points (x,y)
             ha='left', # horizontal alignment can be left, right or center
             fontsize=12)
plt.plot(modSquareSP, fid1R)
plt.plot(modSquareSP, fid2R)
plt.plot(modSquareSP, fid3R)
plt.plot(modSquareSP, fid4R)
# for 'TkAgg' backend to maximised plot windows
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

# plt.savefig('fidelity.png', format='png', transparent=True)

plt.show()
###############################################################################
