"""
    transfer_function(A, B, C, D, s; simplify = false)

Return the transfer function matrix `G(s) = C (sI - A)⁻¹ B + D` of the state-space system
`x' = Ax + Bu`, `y = Cx + Du` as a `Matrix{Num}` of size `(ny, nu)`.

`A` is an `(nx, nx)` matrix, `B` an `(nx, nu)` matrix, or a vector of length `nx` for a
single input, and `C` an `(ny, nx)` matrix, or a vector of length `nx` for a single
output. `D` is an `(ny, nu)` matrix, or a number if `ny == nu == 1`. Other sizes throw a
`DimensionMismatch`. A system without states (`A` of size `(0, 0)`) returns `D`.

`s` may appear in `B` and `D`, as in `B = B1 + s*B2` and `D = D1 + s*D2` for a system
driven by `u` and its derivative `u'`. The formula is applied as written, also if `A` or
`C` contain `s`; an `ArgumentError` is thrown if `sI - A` is then identically singular.
For a linearized nonlinear model, the entries contain states and inputs, which must be
substituted by their values at the operating point before calling
[`frequency_response`](@ref).

Each entry is a fraction of two expanded polynomials, after clearing the denominators of
the entries of `A`, `B`, `C` and `D`, also those nested in sums, products and integer
powers. Common factors of numerator and denominator, such as
uncontrollable or unobservable modes, are only cancelled with `simplify = true`, which
calls `simplify_fractions` on each entry and can be slow. The determinants are expanded
without division, so the cost and the size of the result can grow exponentially with `nx`
for dense symbolic `A`.

Arithmetic on constants is exact, with floating-point constants taken as the rationals they
represent, so `0.1` becomes `3602879701896397//36028797018963968`. Coefficients are never
rounded, because rounding the expanded polynomials can move or create poles and zeros.
They are returned as `Int` or `Rational{Int}` if they fit, and as `BigInt` or
`Rational{BigInt}` otherwise. An `ArgumentError` is thrown for `Inf` and `NaN` constants.
Constants inside other functions, such as `sin`, are left as given.

Only the explicit form above is supported, not the descriptor form `Ex' = Ax + Bu`. If the
matrices come from symbolic differentiation (as with ModelingToolkit's
`linearize_symbolic`), the model must consist of operations that can be differentiated
symbolically.

# Examples

```jldoctest
julia> using Symbolics

julia> @variables s m c k;

julia> transfer_function([0 1; -k/m -c/m], [0, 1/m], [1, 0], 0, s)
1×1 Matrix{Num}:
 1 / (k + c*s + m*(s^2))
```

See also [`frequency_response`](@ref).
"""
function transfer_function(
        A::AbstractMatrix{<:Real}, B::AbstractVecOrMat{<:Real},
        C::AbstractVecOrMat{<:Real}, D::Union{Real, AbstractVecOrMat{<:Real}}, s::Num;
        simplify::Bool = false
    )
    nx = size(A, 1)
    Cm = C isa AbstractVector ? permutedims(collect(C)) : collect(C)
    Dm = D isa Real ? fill(D, 1, 1) : collect(D)
    ny, nu = size(Cm, 1), size(B, 2)
    size(A, 2) == nx || throw(DimensionMismatch("`A` must be square, got size $(size(A))"))
    size(B, 1) == nx ||
        throw(DimensionMismatch("`B` must have $nx rows, got size $(size(B))"))
    size(Cm, 2) == nx ||
        throw(DimensionMismatch("`C` must have $nx columns, got size $(size(C))"))
    size(Dm) == (ny, nu) ||
        throw(DimensionMismatch("`D` must have size $((ny, nu)), got $(size(D))"))

    # All arithmetic is done on polynomials with `BigInt`-based coefficients, because
    # `expand` computes in machine integers, which wrap around silently.
    atoms = Bijections.Bijection{SymbolicUtils.PolyVarT, SymbolicT}()
    fraction(x) = _fraction!(atoms, unwrap(x))

    # Row `i` of `[A B]` and of `[C D]` is multiplied by its common denominator. Then
    # `M = s*diagm(E) - A` and `G[i, j] = det([M B[:, j]; -C[i, :]' D[i, j]]) / (F[i] det(M))`.
    # converted before concatenation, which would promote rationals to floats
    E, AB = _clear_denominators(fraction, Matrix{Num}([Num.(collect(A)) Num.(collect(B))]))
    F, CD = _clear_denominators(fraction, Matrix{Num}([Num.(Cm) Num.(Dm)]))
    sp, o = first(fraction(s)), _expand(_Factors())
    M = DP.Polynomial[(i == j ? E[i] * sp : zero(o)) - AB[i, j] for i in 1:nx, j in 1:nx]
    detM = _determinant(M, o)
    iszero(detM) && throw(ArgumentError("`sI - A` is singular for every `$s`"))
    narrow(p) = Num(
        SymbolicUtils.from_poly(
            atoms,
            DP.Polynomial(
                SymbolicUtils.PolyCoeffT[_narrow(c) for c in MP.coefficients(p)],
                MP.monomials(p)
            )
        )
    )
    G = Matrix{Num}(undef, ny, nu)
    for i in 1:ny, j in 1:nu
        bordered = DP.Polynomial[M AB[:, nx + j]; -permutedims(CD[i, 1:nx]) CD[i, nx + j]]
        num = _determinant(bordered, o)
        den = F[i] * detM
        # dividing by a constant after narrowing could overflow `Int`, or give a float
        G[i, j] = iszero(num) ? Num(0) :
            MP.isconstant(den) ? narrow(num * (1 // MP.leading_coefficient(den))) :
            narrow(num) / narrow(den)
    end
    return simplify ? map(simplify_fractions, G) : G
end

# The factors of a denominator, as `polynomial => multiplicity`
const _Factors = Dict{DP.Polynomial, Int}

# `x` as `(numerator, factors of the denominator)`, with exact coefficients. Anything but
# `+`, `*`, `/` and integer powers becomes a variable of its own, recorded in `atoms`.
function _fraction!(atoms, x::SymbolicT)
    if SymbolicUtils.isconst(x)
        v = unwrap_const(x)
        v isa AbstractFloat && !isfinite(v) &&
            throw(ArgumentError("the constant $v is not finite"))
        c = v isa Integer ? BigInt(v) :
            v isa Union{Rational, AbstractFloat} ? Rational{BigInt}(v) : nothing
        c === nothing || return MP.polynomial(c * one(SymbolicUtils.PolyVarT)), _Factors()
    elseif iscall(x)
        op, args = operation(x), arguments(x)
        if op === (+)
            terms = map(a -> _fraction!(atoms, a), args)
            common = mergewith(max, _Factors(), map(last, terms)...)
            return sum(((num, den),) -> num * _expand(common, den), terms), common
        end
        op === (*) && return _product(map(a -> _fraction!(atoms, a), args))
        op === (/) && return _product([_fraction!(atoms, args[1]), _inverse!(atoms, args[2])])
        if op === (^)
            n = _integer_exponent(args[2])
            n isa Integer && n >= 0 && return _power(_fraction!(atoms, args[1]), n)
            n isa Integer && return _power(_inverse!(atoms, args[1]), -n)
        end
    end
    var = SymbolicUtils.basicsymbolic_to_polyvar(Bijections.active_inv(atoms), x)
    return MP.polynomial(big(1) * var), _Factors()
end

# `1 / x` as in `_fraction!`, keeping the factors of a product apart so that they can be
# shared with other denominators
function _inverse!(atoms, x::SymbolicT)
    if iscall(x)
        op, args = operation(x), arguments(x)
        op === (*) && return _product(map(a -> _inverse!(atoms, a), args))
        op === (/) && return _product([_fraction!(atoms, args[2]), _inverse!(atoms, args[1])])
        if op === (^)
            n = _integer_exponent(args[2])
            n isa Integer && n >= 0 && return _power(_inverse!(atoms, args[1]), n)
            n isa Integer && return _power(_fraction!(atoms, args[1]), -n)
        end
    end
    num, den = _fraction!(atoms, x)
    MP.isconstant(num) && !iszero(num) &&
        return _expand(den) * (1 // MP.leading_coefficient(num)), _Factors()
    return _expand(den), _Factors(num => 1)
end

# An integer-valued constant exponent, such as `2//1`, as an `Integer`
function _integer_exponent(x)
    n = unwrap_const(x)
    return n isa Real && isfinite(n) && isinteger(n) ? Integer(n) : nothing
end

_product(fractions) = prod(first, fractions), mergewith(+, _Factors(), map(last, fractions)...)
_power((num, den), n) = num^n, _Factors(p => n * k for (p, k) in den)

# The product of the factors in `common` which are not in `den`
function _expand(common, den = _Factors())
    return prod(
        (p^(k - get(den, p, 0)) for (p, k) in common);
        init = MP.polynomial(big(1) * one(SymbolicUtils.PolyVarT))
    )
end

# Coefficients are converted back to machine numbers if they fit
function _narrow(v)
    q = isone(denominator(v)) ? numerator(v) : v
    fits = abs(numerator(q)) <= typemax(Int) && denominator(q) <= typemax(Int)
    return fits ? (q isa BigInt ? Int(q) : Rational{Int}(q)) : q
end

# Returns `(d, P)` with `P == d .* X` row-wise and no fractions left in `P`, so that like
# terms cancel in the determinants.
function _clear_denominators(fraction, X::Matrix{Num})
    d = Vector{DP.Polynomial}(undef, size(X, 1))
    P = Matrix{DP.Polynomial}(undef, size(X))
    for i in axes(X, 1)
        fractions = map(fraction, X[i, :])
        common = mergewith(max, _Factors(), map(last, fractions)...)
        d[i] = _expand(common)
        for j in axes(X, 2)
            num, den = fractions[j]
            P[i, j] = num * _expand(common, den)
        end
    end
    return d, P
end

# Division-free Laplace expansion, memoised on the set of remaining columns. Zero entries
# are skipped, so sparse matrices stay cheap.
function _determinant(M::Matrix{DP.Polynomial}, unit)
    n = size(M, 1)
    memo = Dict{BitSet, DP.Polynomial}()
    function minor_det(row, cols)
        row > n && return unit
        return get!(memo, cols) do
            total = zero(unit)
            for (k, j) in enumerate(cols)
                iszero(M[row, j]) && continue
                term = M[row, j] * minor_det(row + 1, setdiff(cols, j))
                total += isodd(k) ? term : -term
            end
            total
        end
    end
    return minor_det(1, BitSet(1:n))
end

"""
    frequency_response(G, s, ωs; exact = true)

Evaluate the transfer function matrix `G` (see [`transfer_function`](@ref)) at `s = im*ω`
for every angular frequency `ω` in `ωs`. Returns an `Array{ComplexF64, 3}` of size
`(ny, nu, length(ωs))`.

If every entry of `G` is a fraction of polynomials in `s`, as the result of
`transfer_function` is, it is evaluated in exact rational arithmetic, with `ω` and the
constants in `G` taken as the rationals they represent, and rounded to `ComplexF64` once.
The expanded polynomials can cancel to far fewer digits than `Float64` holds, for example
near a lightly damped resonance. This takes tens of microseconds per entry and frequency
for a system with six states. With `exact = false`, or if `G` has other entries, for
example a delay `exp(-s)`, `G` is evaluated in `ComplexF64` arithmetic, with a function
generated once by [`build_function`](@ref). This is tens of times faster, but the
cancellation can leave no correct digits, or give `NaN`, close to a resonance.

`s` must be the only variable in `G`: substitute numeric values for all parameters first,
otherwise an `ArgumentError` is thrown. `substitute` folds the constants in the arithmetic
of the values, so machine integers can wrap around and floating-point numbers round. To
keep the exact evaluation, substitute `Rational{BigInt}` values, such as
`Rational{BigInt}(0.1)`. With `exact = true`, an `ArgumentError` is thrown for an infinite
frequency. At a pole on the grid the entry is not finite (it contains `Inf`
or `NaN`). A common factor of numerator and denominator which vanishes on the grid also
gives `NaN`.

# Examples

```jldoctest
julia> using Symbolics

julia> @variables s T;

julia> G = transfer_function(fill(-1/T, 1, 1), [1/T], [1], 0, s);

julia> frequency_response(substitute(G, Dict(T => 2)), s, [0.5, 1.0])
1×1×2 Array{ComplexF64, 3}:
[:, :, 1] =
 0.5 - 0.5im

[:, :, 2] =
 0.2 - 0.4im
```
"""
function frequency_response(
        G::AbstractMatrix{Num}, s::Num, ωs::AbstractVector{<:Real}; exact::Bool = true
    )
    free = Set{SymbolicT}()
    for g in G
        union!(free, get_variables(g))
    end
    delete!(free, unwrap(s))
    isempty(free) || throw(
        ArgumentError(
            "`G` contains variables other than `$s`: $(join(free, ", ")). \
            `substitute` their values first."
        )
    )
    response = Array{ComplexF64}(undef, size(G)..., length(ωs))
    atoms = Bijections.Bijection{SymbolicUtils.PolyVarT, SymbolicT}()
    x = SymbolicUtils.basicsymbolic_to_polyvar(Bijections.active_inv(atoms), unwrap(s))
    fractions = map(G) do g
        num, den = _fraction!(atoms, unwrap(g))
        num, _expand(den)
    end
    polynomial = length(atoms) == 1
    if exact && polynomial
        all(isfinite, ωs) ||
            throw(ArgumentError("`exact = true` needs finite frequencies, got $ωs"))
        # Fractions of polynomials in `s` are evaluated exactly. Cancellation in the
        # expanded polynomials can leave far fewer correct digits than `ComplexF64` has.
        coefficients = map(f -> map(p -> _coefficients(p, x), f), fractions)
        for (k, ω) in enumerate(ωs), I in CartesianIndices(G)
            num, den = map(c -> _horner(c, Rational{BigInt}(ω)), coefficients[I])
            response[I, k] = iszero(den) ? ComplexF64(num) / 0 : ComplexF64(num / den)
        end
        return response
    end
    # `BigInt` and rational constants would make the generated function compute in
    # `BigFloat`. In other functions, converting them could change the method called.
    Gf = polynomial ? map(g -> Num(SymbolicUtils.Postwalk(_float_constant)(unwrap(g))), G) : G
    # NaNMath functions do not accept complex arguments
    _, evaluate! = build_function(Gf, s; expression = Val{false}, nanmath = false)
    for (k, ω) in enumerate(ωs)
        evaluate!(view(response, :, :, k), complex(0.0, ω))
    end
    return response
end

function _float_constant(x)
    SymbolicUtils.isconst(x) && unwrap_const(x) isa Union{BigInt, Rational} || return x
    return SymbolicUtils.Const{SymbolicUtils.SymReal}(Float64(unwrap_const(x)))
end

# The coefficients of `p`, a polynomial in `x` alone, in increasing degree
function _coefficients(p, x)
    c = zeros(Rational{BigInt}, MP.maxdegree(p) + 1)
    for term in MP.terms(p)
        c[MP.degree(MP.monomial(term), x) + 1] = MP.coefficient(term)
    end
    return c
end

# The polynomial with coefficients `c` at `s = im*ω`, by Horner's rule
function _horner(c, ω)
    re, im = zero(ω), zero(ω)
    for k in reverse(eachindex(c))
        re, im = c[k] - im * ω, re * ω
    end
    return complex(re, im)
end
