"""Convenience type aliases for jaxtyping scalar annotations."""

from jaxtyping import Array, Complex, Float, Int

type FloatScalar = Float[Array, ""]
type IntScalar = Int[Array, ""]
type ComplexScalar = Complex[Array, ""]
type FloatLike = float | Float[Array, ""]
type ComplexLike = complex | ComplexScalar
