module DenseJacobian

"""Finite Differences are far from the best solution to find jacobians but are the easiest to
implement. Hence this solution.

    getJacobian!(jacobian::Array{Float64,2}, f::Function, x::Array{Float64,1})

Here "jacobian" must have row size equal to length of vector returned by Function "f" and column size equal to
length of vector "x"
"""
function getJacobian!(jacobian::Array{Float64,2}, f::Function, x::Array{Float64,1})
    f_col = f(x)
    rows::Int64 = length(f_col)
    cols::Int64 = length(x)
    @assert size(jacobian) == (rows, cols) "The rows must be the length of the function output and the cols the length of the input function"
    h::Float64 = 0.0
    for col ∈ 1:cols
        h = x[col] == 0.0 ? sqrt(eps(1.0)) : sqrt(eps(x[col]))*x[col]
        jacobian[:, col] .= (f(x+[zeros(col-1); h; zeros(cols-col)]) - f_col)/h
        #println((x+[zeros(col-1); h; zeros(cols-col)]))

    end
    return nothing
end

"""getJacobian is an easier to use version of denseJacobian!. This is less efficient than denseJacobian!.
Hence whenever possible denseJacobian! must be used."""
function getJacobian(f::Function, x::Array{Float64,1})
    f_col = f(x)
    rows::Int64 = length(f_col)
    cols::Int64 = length(x)
    jacobian::Array{Float64,2} = zeros(rows, cols)
    getJacobian!(jacobian, f, x)
    return jacobian
end

export getJacobian!, getJacobian

end # module
