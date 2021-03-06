# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: May 2020
#

using JUDI.TimeModeling, PyCall, LinearAlgebra, PyPlot, Test


# Test padding and its adjoint
function test_padding(ndim)
    n = Tuple(51 for i=1:ndim)
    o = Tuple(0. for i=1:ndim)
    d = Tuple(10. for i=1:ndim)

    model= Model(n, d, o, rand(Float32, n...); nb=10)
    modelPy = devito_model(model, Options())

    v0 = sqrt.(1f0 ./model.m)
    cut = remove_padding(deepcopy(modelPy.vp.data), modelPy.padsizes; true_adjoint=true)
    vpdata = deepcopy(modelPy.vp.data)

    @show dot(v0, cut)/dot(vpdata, vpdata)
    @test isapprox(dot(v0, cut), dot(vpdata, vpdata))
end

for ndim = 2:3
    test_padding(ndim)
end
