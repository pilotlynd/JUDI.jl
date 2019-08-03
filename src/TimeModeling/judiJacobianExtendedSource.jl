############################################################
# judiJacobianExQ ###############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiJacobianExQ, judiJacobianExQException, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections
struct judiJacobianExQ{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    recGeometry::Geometry
    wavelet
    weights
    options::Options
    fop::Function              # forward
    fop_T::Union{Function, Nothing}  # transpose
end


mutable struct judiJacobianExQException <: Exception
    msg :: String
end

############################################################
## Constructor
"""
    judiJacobianExQ(F,q)

Create a linearized modeling operator from the non-linear modeling operator `F` and \\
the source `q`. `F` is a full modeling operator including source/receiver projections.

Examples
========

1) `F` is a modeling operator without source/receiver projections:

    J = judiJacobianExQ(Pr*F*Ps',q)

2) `F` is the combined operator `Pr*F*Ps'`:

    J = judiJacobianExQ(F,q)

"""
function judiJacobian(F::judiPDEextended, weights::judiWeights; DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling w/ extended source

    (DDT == Float32 && RDT == Float32) || throw(judiJacobianExQException("Domain and range types not supported"))
    if typeof(F.recGeometry) == GeometryOOC
        m = sum(F.recGeometry.nsamples)
    else
        m = 0
        for j=1:F.info.nsrc m += length(F.recGeometry.xloc[j])*F.recGeometry.nt[j] end
    end
    n = F.info.n
    if F.info.nsrc > 1
        srcnum = 1:F.info.nsrc
    else
        srcnum = 1
    end

    if F.options.return_array == true
        return J = judiJacobianExQ{Float32,Float32}("linearized wave equation", m, n, F.info, F.model, F.recGeometry, F.wavelet, weights.weights, F.options,
        v -> extended_source_modeling(F.model, F.wavelet, F.recGeometry, nothing, weights.weights, v, srcnum, 'J', 1, F.options),
        w -> extended_source_modeling(F.model, F.wavelet, F.recGeometry, reshape(w, F.recGeometry.nt[1], length(F.recGeometry.xloc[1]), F.info.nsrc), weights.weights, nothing, srcnum, 'J', -1, F.options)
        )
    else
        return J = judiJacobianExQ{Float32,Float32}("linearized wave equation", m, n, F.info, F.model, F.recGeometry, F.wavelet, weights.weights, F.options,
        v -> extended_source_modeling(F.model, F.wavelet, F.recGeometry, nothing, weights.weights, v, srcnum, 'J', 1, F.options),
        w -> extended_source_modeling(F.model, F.wavelet, F.recGeometry, w.data, weights.weights, nothing, srcnum, 'J', -1, F.options)
        )
    end
end

# Constructor if weights are given as 1D vector (for one or multiple extended sources)
function judiJacobian(F::judiPDEextended, weights::Array; DDT::DataType=Float32, RDT::DataType=DDT)
    weights = reshape(weights, F.model.n[1], F.model.n[2], F.info.nsrc)
    weights_cell = Array{Array}(undef, F.info.nsrc)
    for j=1:F.info.nsrc
        weights_cell[j] = weights[:,:,j]
    end
    return judiJacobian(F, judiWeights(weights_cell); DDT=DDT, RDT=RDT)
end

############################################################
## overloaded Base functions

# conj(judiJacobianExQ)
conj(A::judiJacobianExQ{DDT,RDT}) where {DDT,RDT} =
    judiJacobianExQ{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
        A.fop,
        A.fop_T
        )

# transpose(judiJacobianExQ)
transpose(A::judiJacobianExQ{DDT,RDT}) where {DDT,RDT} =
    judiJacobianExQ{DDT,RDT}("adjoint linearized wave equation",A.n,A.m,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
        A.fop_T,
        A.fop
        )

# adjoint(judiJacobianExQ)
adjoint(A::judiJacobianExQ{DDT,RDT}) where {DDT,RDT} =
    judiJacobianExQ{DDT,RDT}("adjoint linearized wave equation",A.n,A.m,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
        A.fop_T,
        A.fop
        )

############################################################
## overloaded Base *(...judiJacobianExQ...)

# *(judiJacobianExQ,vec)
function *(A::judiJacobianExQ{ADDT,ARDT},v::AbstractVector{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiJacobianExQException("shape mismatch"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiJacobianExQ,vec):",A.name,typeof(A),vDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiJacobianExQ,vec):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(judiJacobianExQ,judiVector)
function *(A::judiJacobianExQ{ADDT,ARDT},v::judiVector{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiJacobianExQException("shape mismatch"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiJacobianExQ,judiVector):",A.name,typeof(A),vDT]," / "))
    compareGeometry(A.recGeometry,v.geometry) == true || throw(judiJacobianExQException("Geometry mismatch"))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiJacobianExQ,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(num,judiJacobianExQ)
function *(a::Number,A::judiJacobianExQ{ADDT,ARDT}) where {ADDT,ARDT}
    return judiJacobianExQ{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
                                v1 -> jo_convert(ARDT,a*A.fop(v1),false),
                                v2 -> jo_convert(ADDT,a*A.fop_T(v2),false)
                                )
end

function A_mul_B!(x::judiVector,J::judiJacobianExQ,y::Array)
    z = J*y
    x.data = z.data
end

function Ac_mul_B!(x::Array,J::judiJacobianExQ,y::judiVector)
    x[:] = adjoint(J)*y
end

############################################################
## overloaded Bases +(...judiJacobianExQ...), -(...judiJacobianExQ...)

# +(judiJacobianExQ,num)
function +(A::judiJacobianExQ{ADDT,ARDT},b::Number) where {ADDT,ARDT}
    return judiJacobianExQ{ADDT,ARDT}("("*A.name*"+N)",A.m,A.n,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
                            v1 -> A.fop(v1)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
                            v2 -> A.fop_T(v2)+joConstants(A.n,A.m,b;DDT=ADDT,RDT=ARDT)*v2
                            )
end

# -(judiJacobianExQ,num)
function -(A::judiJacobianExQ{ADDT,ARDT},b::Number) where {ADDT,ARDT}
    return judiJacobianExQ{ADDT,ARDT}("("*A.name*"-N)",A.m,A.n,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
                            v1 -> A.fop(v1)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
                            v2 -> A.fop_T(v2)-joConstants(A.n,A.m,b;DDT=ADDT,RDT=ARDT)*v2
                            )
end

# -(judiJacobianExQ)
-(A::judiJacobianExQ{DDT,RDT}) where {DDT,RDT} =
    judiJacobianExQ{DDT,RDT}("(-"*A.name*")",A.m,A.n,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
                    v1 -> -A.fop(v1),
                    v2 -> -A.fop_T(v2)
                    )

############################################################
## Additional overloaded functions

# Subsample Jacobian
function subsample(J::judiJacobianExQ{ADDT,ARDT}, srcnum) where {ADDT,ARDT}

    recGeometry = subsample(J.recGeometry,srcnum)

    info = Info(J.info.n, length(srcnum), J.info.nt[srcnum])
    Fsub = judiModeling(info, J.model, recGeometry; options=J.options)
    wsub = judiWeigths(J.weights[srcnum])
    return judiJacobianExQ(Fsub, wsub; DDT=ADDT, RDT=ARDT)
end

getindex(J::judiJacobianExQ,a) = subsample(J,a)
