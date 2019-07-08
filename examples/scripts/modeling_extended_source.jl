# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, SeisIO, LinearAlgebra, PyPlot

## Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v[:,Int(round(end/2)):end] .= 5f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2

# Setup info and model structure
nsrc = 2	# number of sources
model = Model(n, d, o, m)

## Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
time = 1000f0   # receiver recording time [ms]
dt = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

# Source wavelet
f0 = 0.01f0     # MHz
wavelet = ricker_wavelet(time, dt, f0)

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

######################## WITH DENSITY ############################################

# Write shots as segy files to disk
opt = Options()

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)

# Random weights (size of the model)
weights = Array{Any}(undef, 2)
weights[1] = randn(Float32, model.n)
weights[2] = randn(Float32, model.n)

w = judiWeights(weights)

# Create operator for injecting the weights, multiplied by the provided wavelet(s)
Pw = judiLRWF(info, wavelet)

# Model observed data w/ extended source
F = Pr*F*adjoint(Pw)

# Simultaneous observed data
d_sim = F*w

# Adjoint operation
dw = adjoint(F)*d_sim

# Plot results
figure()
subplot(1,2,1)
imshow(d_sim.data[1], vmin=-5e2, vmax=5e2, cmap="gray"); title("Shot no. 1")
subplot(1,2,2)
imshow(d_sim.data[2], vmin=-5e2, vmax=5e2, cmap="gray"); title("Shot no. 2")

figure()
subplot(1,2,1)
imshow(adjoint(dw.weights[1]), vmin=-5e6, vmax=5e6, cmap="gray"); title("Weights 1")
subplot(1,2,2)
imshow(adjoint(dw.weights[2]), vmin=-5e6, vmax=5e6, cmap="gray"); title("Weights 2")