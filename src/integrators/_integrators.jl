module Integrators

export AbstractIntegrator

export QuantumIntegrator

export QuantumPadeIntegrator
export QuantumStatePadeIntegrator
export UnitaryPadeIntegrator

export UnitaryExponentialIntegrator
export QuantumStateExponentialIntegrator
export DensityOperatorExponentialIntegrator

export DerivativeIntegrator

export jacobian
export hessian_of_the_lagrangian

export get_comps

export nth_order_pade
export fourth_order_pade
export sixth_order_pade
export eighth_order_pade
export tenth_order_pade

export get_suffix
export add_suffix
export remove_suffix
export modify_integrator_suffix

using NamedTrajectories
using TrajectoryIndexingUtils
using PiccoloQuantumObjects
using LinearAlgebra
using SparseArrays
using ForwardDiff
using TestItems

import PiccoloQuantumObjects

const ⊗ = kron

abstract type AbstractIntegrator end

function comps(P::AbstractIntegrator, traj::NamedTrajectory)
    state_comps = traj.components[state(P)]
    u = controls(P)
    if u isa Tuple
        control_comps = Tuple(traj.components[uᵢ] for uᵢ ∈ u)
    else
        control_comps = traj.components[u]
    end
    if traj.timestep isa Float64
        return state_comps, control_comps
    else
        timestep_comp = traj.components[traj.timestep]
        return state_comps, control_comps, timestep_comp
    end
end



abstract type QuantumIntegrator <: AbstractIntegrator end

abstract type UnitaryIntegrator <: QuantumIntegrator end
abstract type QuantumStateIntegrator <: QuantumIntegrator end
abstract type DensityOperatorIntegrator <: QuantumIntegrator end

include("_integrator_utils.jl")
include("derivative_integrator.jl")
include("pade_integrators.jl")
include("exponential_integrators.jl")


# ----------------------------------------------------------------------------
# Integrator direct sum methods
# ----------------------------------------------------------------------------

function modify_integrator_suffix(
    integrator::AbstractIntegrator,
    modifier::Function,
    suffix::String,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory
)
    mod_integrator = deepcopy(integrator)

    if integrator isa QuantumIntegrator
        state_name = get_component_names(traj, integrator.state_components)
        drive_name = get_component_names(traj, integrator.drive_components)
        mod_integrator.state_components = mod_traj.components[modifier(state_name, suffix)]
        mod_integrator.drive_components = mod_traj.components[modifier(drive_name, suffix)]
        return mod_integrator
    elseif integrator isa DerivativeIntegrator
        var_name = get_component_names(traj, integrator.variable_components)
        der_name = get_component_names(traj, integrator.derivative_components)
        mod_integrator.variable_components = mod_traj.components[modifier(var_name, suffix)]
        mod_integrator.derivative_components = mod_traj.components[modifier(der_name, suffix)]
        return mod_integrator
    else
        error("Integrator type not recognized")
    end
end

function NamedTrajectories.add_suffix(
    integrator::AbstractIntegrator,
    suffix::String,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory
)
    return modify_integrator_suffix(integrator, add_suffix, suffix, traj, mod_traj)
end

function NamedTrajectories.add_suffix(
    integrators::AbstractVector{<:AbstractIntegrator},
    suffix::String,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory
)
    return [
        add_suffix(integrator, suffix, traj, mod_traj)
            for integrator ∈ integrators
    ]
end

function NamedTrajectories.remove_suffix(
    integrator::AbstractIntegrator,
    suffix::String,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory
)
    return modify_integrator_suffix(integrator, remove_suffix, suffix, traj, mod_traj)
end

function NamedTrajectories.remove_suffix(
    integrators::AbstractVector{<:AbstractIntegrator},
    suffix::String,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory
)
    return [remove_suffix(intg, suffix, traj, mod_traj) for intg in integrators]
end


# Get suffix utilities
# --------------------

Base.endswith(integrator::UnitaryPadeIntegrator, suffix::String) = endswith(integrator.unitary_symb, suffix)
Base.endswith(integrator::DerivativeIntegrator, suffix::String) = endswith(integrator.variable, suffix)

function Base.endswith(integrator::AbstractIntegrator, traj::NamedTrajectory, suffix::String)
    if integrator isa UnitaryExponentialIntegrator
        name = get_component_names(traj, integrator.state_components)
    elseif integrator isa QuantumStateExponentialIntegrator
        name = get_component_names(traj, integrator.state_components)
    elseif integrator isa UnitaryPadeIntegrator
        name = get_component_names(traj, integrator.state_components)
    elseif integrator isa QuantumStatePadeIntegrator
        name = get_component_names(traj, integrator.state_components)
    elseif integrator isa DerivativeIntegrator
        name = get_component_names(traj, integrator.variable_components)
    else
        error("Integrator type not recognized")
    end
    return endswith(name, suffix)
end

function NamedTrajectories.get_suffix(
    integrators::AbstractVector{<:AbstractIntegrator},
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory,
    suffix::String
)
    found = AbstractIntegrator[]
    for integrator ∈ integrators
        if endswith(integrator, traj, suffix)
            push!(found, remove_suffix(integrator, sys, traj, mod_traj, suffix))
        end
    end
    return found
end


end
