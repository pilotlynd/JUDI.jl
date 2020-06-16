# to import MPIManager
using MPIClusterManagers

# need to also import Distributed to use addprocs()
using Distributed

export mpi_devito_interface

function mympi_do(mgr::MPIManager, expr)
    !mgr.initialized && wait(mgr.cond_initialized)
    jpids = keys(mgr.j2mpi)
    refs = Array{Any}(undef, length(jpids))
    out = Array{Any}(undef, length(jpids))
    for (i, p) in enumerate(jpids)
        refs[i] = remotecall(expr, p)
    end

    @sync begin
        for (i, r) in enumerate(refs)
            resp = remotecall_fetch(r.where, r) do rr
                wrkr_result = rr[]
            end
            out[i] = wrkr_result
        end
    end
    out = filter!(x->x!=nothing, out)
    return out[1]
end

macro mympi_do(mgr, expr)
    quote
        # Evaluate expression in Main module
        thunk = () -> (Core.eval(Main, $(Expr(:quote, expr))))
        mympi_do($(esc(mgr)), thunk)
    end
end

function mpi_devito_interface(model, op, args...)
    options = args[end]

    manager = MPIManager(np=options.mpi)
    workers = addprocs(manager)
    # import back JUDI (yeah that's wierd but needed)
    eval(macroexpand(Distributed, quote @everywhere using JUDI, JUDI.TimeModeling end))
    length(model.n) == 3 ? dims = [3,2,1] : dims = [2,1]

    @mympi_do manager begin
        # Init MPI
        using MPI
        comm = MPI.COMM_WORLD
        # Set up Python model structure
        modelPy = devito_model($model, $options)
        update_m(modelPy, $model.m, $dims)
        # Run devito interface
        argout = devito_interface(modelPy, $model, $(args...))
        # Wait for it to finish
        MPI.Barrier(comm)
        # GAther results
        if MPI.Comm_rank(comm) == 0
            println("Receiving on $(MPI.Comm_rank(comm))")
            out = Array{judiVector}(undef, MPI.Comm_size(comm)-1)
            for i=1:MPI.Comm_size(comm)
                out[i], status = MPI.recv(i, i, comm)
            end
            argout = argout + sum(out)
        else
            println("sending from $(MPI.Comm_rank(comm))")
            MPI.send(argout, 0, MPI.Comm_rank(comm), comm)
        end
        # Make sure gather is finished
        MPI.Barrier(comm)
        # Return result
        if MPI.Comm_rank(comm) == 0
            return argout
        end
    end
end