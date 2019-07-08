using Statistics, Random, LinearAlgebra
using JUDI.TimeModeling, JUDI.SLIM_optim, HDF5, SeisIO, PyPlot, JOLI

# Load background velocity model
n,d,o,m0 = read(h5open("../../data/overthrust_model.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)
n,d,o,m = read(h5open("../../data/overthrust_model.h5","r"), "n", "d", "o", "m")

# Set source & receiver geometry
nsrc = 369;
xsrc  = convertToCell(range(400f0, stop=9600f0, length=nsrc));
ysrc  = convertToCell(range(0f0, stop=0f0, length=nsrc));
zsrc  = convertToCell(range(50f0, stop=50f0, length=nsrc));
time  = 2000f0   # receiver recording time [ms]
dt    = 4.0f0    # receiver sampling interval [ms]

# Set up source structure
src_Geometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time);
wavelet = ricker_wavelet(src_Geometry.t[1],src_Geometry.dt[1],0.008f0)  # 8 Hz wavelet
num_sources = length(src_Geometry.xloc)

# Convert to single simultaneous source
xsrc = zeros(Float32, num_sources)
ysrc = zeros(Float32, num_sources)
zsrc = zeros(Float32, num_sources)
sim_source = zeros(Float32, src_Geometry.nt[1], num_sources)
for j=1:num_sources
    xsrc[j] = src_Geometry.xloc[j]
    zsrc[j] = src_Geometry.zloc[j]
    sim_source[:, j] = wavelet * randn(1)[1]/sqrt(num_sources)    # wavelet w/ random weight
end

# simultaneous source geometry and JUDI vector
sim_geometry = Geometry(xsrc, ysrc, zsrc; dt=src_Geometry.dt[1], t=src_Geometry.t[1])
q = judiVector(sim_geometry, sim_source)

# Receiver
xrec  = range(400f0, stop=9600f0, length=nsrc);
yrec  = range(0f0, stop=0f0, length=nsrc);
zrec  = range(50f0, stop=50f0, length=nsrc);

# Set up source structure
rec_Geometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time, nsrc=1);

nsrc = 1    # one simultaneous source
ntComp = get_computational_nt(sim_geometry, rec_Geometry, model0)

info = Info(prod(n), nsrc, ntComp) # only need geometry of one shot record

# Setup operators
opt = Options(return_array=true)
Pr = judiProjection(info, rec_Geometry)    # set up 1 simultaneous shot instead of 16
F = judiModeling(info, model0; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, sim_geometry)
J = judiJacobian(Pr*F0*adjoint(Ps), q)

# Topmute
T = judiTopmute(model0.n, 29, 10)
J̄ = J*T

# Linearized modeling + migration
δm = m .- m0;
δm = (δm .-mean(δm)) ./ std(δm);

# # Linearized modeling with on-the-fly Fourier + migration
# J.options.dft_subsampling_factor = 8
# q_dist = generate_distribution(q)
# J.options.frequencies = Array{Any}(undef, nsrc)
# for j=1:nsrc
#     J.options.frequencies[j] = select_frequencies(q_dist; fmin=0.002, fmax=0.04, nf=4)
# end

# Linearized modeling w/ single sim source
δd = J̄*vec(δm);

# RTM w/ single sim source
rtm = transpose(J̄)*vec(δd);

# Linearized Bregman
x = randn(Float32, info.n)
z = randn(Float32, info.n)
S(x::Array{Float64}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float64,lambda), 0.0)
S(x::Array{Float32}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float32,lambda), 0f0)
C = joCurvelet2D(model0.n[1], model0.n[2]; all_crvlts=true, DDT=Float32, RDT=Float64)
lambda = []
maxiter = 50
rnorm = zeros(Float32, maxiter)
gnorm = zeros(Float32, maxiter)
xnorm = zeros(Float32, maxiter)
redraw = true

for j=1:maxiter
    print("Iteration ", j, "\n")

    # Model data residual
    if redraw == true
        for k=1:num_sources
            J.source[1][:, k] = wavelet * randn(1)[1]/sqrt(num_sources)
        end
        r = J̄*(x - vec(δm))    # same as δd = J*δm; r = J*x - δd, but w/ single PDE solve
    else
        r = J̄*x - δd
    end

    g = adjoint(J̄)*r
    t = norm(r)^2/norm(g)^2

    if j==1 && isempty(lambda)
        global lambda = .5f0*norm(C*t*g, Inf)
    end

    global z = z - t*g
    global x = adjoint(C)*S(C*z, lambda)

    rnorm[j] = norm(r)
    gnorm[j] = norm(g)
    xnorm[j] = norm(x - vec(δm))
end


# Plot results
figure(); imshow(reshape(δd, rec_Geometry.nt[1], length(rec_Geometry.xloc[1])), vmin=-2e2, vmax=2e2, cmap="gray")
title("Simultaneous shot")
figure(figsize=(6, 8));
subplot(3,1,1); imshow(transpose(reshape(δm,model0.n)))
title("True image")
subplot(3,1,2); imshow(transpose(reshape(rtm,model0.n)), vmin=-1e5, vmax=1e5)
title("Sim-source RTM")
subplot(3,1,3); imshow(transpose(reshape(T*x,model0.n)), vmin=-4e0, vmax=4e0)
title("Sim-source SPLS-RTM w/ redraws")
