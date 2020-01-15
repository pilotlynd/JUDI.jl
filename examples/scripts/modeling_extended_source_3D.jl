# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, SegyIO, LinearAlgebra, PyPlot

## Set up model structure
n = (100, 100, 80)   # (x,y,z) or (x,z)
d = (10., 10., 10.)
o = (0., 0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v[:, :, Int(round(end/2)):end] .= 3f0
v0 = ones(Float32,n) .+ 0.4f0


# Slowness squared [s^2/km^2]
m0 = (1f0 ./ v0).^2
m = (1f0 ./ v).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 1	# number of sources
model0 = Model(n, d, o, m0)
model = Model(n, d, o, m)

# Receiver geometry
nxrec = 120
nyrec = 100
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = range(100f0, stop=900f0, length=nyrec)
zrec = 50f0

# Construct 3D grid from basis vectors
(xrec, yrec, zrec) = setup_3D_grid(xrec, yrec, zrec)

# receiver sampling and recording time
time = 200f0   # receiver recording time [ms]
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
opt = Options(return_array=true)

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)

# Random weights (size of the model)
weights = randn(Float32, model.n)

# Create operator for injecting the weights, multiplied by the provided wavelet(s)
Pw = judiLRWF(info, wavelet)

# Model observed data w/ extended source
F = Pr*F*adjoint(Pw)

# Simultaneous observed data
d_sim = F*vec(weights)
dw = adjoint(F)*d_sim

# Jacobian
J = judiJacobian(F, judiWeights(weights))
d_lin = J*dm
g = adjoint(J)*d_lin

