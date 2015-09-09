import GSL.deriv_central
using Base.Test


# verify derivatives of f by finite differences for functions of
# numeric vectors that return a tuple (value, derivient).
function test_deriv(f::Function, x::FloatingPoint, claimed_deriv::FloatingPoint)
    numeric_deriv, abs_err = deriv_central(f, x, 1e-3)
    info("deriv: $numeric_deriv vs $(claimed_deriv) [tol: $abs_err]")
    obs_err = abs(numeric_deriv - claimed_deriv)
    @test obs_err < 1e-11 || abs_err < 1e-4 || abs_err / abs(numeric_deriv) < 1e-4
    @test_approx_eq_eps numeric_deriv claimed_deriv 10abs_err
end

# test the test
test_deriv( (x)->x^2, 4., 8. )

