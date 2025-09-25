//! pervasive array operations.
pub mod scalar_and_array {
    //! traits for scalar $op array
    macro_rules! op {
        ($op:ident, $n:ident, $f:ident) => {
            #[doc = concat!("see [`", stringify!($f), "`](core::ops::", stringify!($op), "::", stringify!($f), ")")]
            pub trait $n<T, const N: usize> {
                #[doc = concat!("apply the [`", stringify!($f), "`](core::ops::", stringify!($op), "::", stringify!($f), ") function to each element of this array.")]
                fn $f(self, rhs: [T; N]) -> [T; N];
            }

            impl<T: core::ops::$op<Output = T> + Copy, const N: usize> $n<T, N> for T {
                fn $f(self, rhs: [T; N]) -> [T; N] {
                    rhs.map(|x| core::ops::$op::$f(self, x))
                }
            }
        };
    }
    op!(Add, SAAAdd, add);
    op!(BitAnd, SAAAnd, bitand);
    op!(BitOr, SAAOr, bitor);
    op!(BitXor, SAAXor, bitxor);
    op!(Div, SAADiv, div);
    op!(Mul, SAAMul, mul);
    op!(Rem, SAARem, rem);
    op!(Shl, SAAShl, shl);
    op!(Shr, SAAShr, shr);
    op!(Sub, SAASub, sub);
}

pub mod array_and_scalar {
    //! traits for array $op scalar
    macro_rules! op {
        ($op:ident, $n:ident, $f:ident) => {
            #[doc = concat!("see [`", stringify!($f), "`](core::ops::", stringify!($op), "::", stringify!($f), ")")]
            pub trait $n<T, const N: usize> {
                #[doc = concat!("apply the [`", stringify!($f), "`](core::ops::", stringify!($op), "::", stringify!($f), ") function to each element of this array.")]
                fn $f(self, rhs: T) -> Self;
            }

            impl<T: core::ops::$op<Output = T> + Copy, const N: usize> $n<T, N> for [T; N] {
                fn $f(self, rhs: T) -> Self {
                    self.map(|x| core::ops::$op::$f(x, rhs))
                }
            }
        };
    }
    op!(Add, AASAdd, add);
    op!(BitAnd, AASAnd, bitand);
    op!(BitOr, AASOr, bitor);
    op!(BitXor, AASXor, bitxor);
    op!(Div, AASDiv, div);
    op!(Mul, AASMul, mul);
    op!(Rem, AASRem, rem);
    op!(Shl, AASShl, shl);
    op!(Shr, AASShr, shr);
    op!(Sub, AASSub, sub);
}

mod array_and_array {
    //! traits for array $op scalar
    macro_rules! op {
            ($op:ident, $n:ident, $f:ident, $name:ident) => {
                #[doc = concat!("see [`", stringify!($f), "`](core::ops::", stringify!($op), "::", stringify!($f), ")")]
                pub trait $n<T, const N: usize> {
                    #[doc = concat!("apply the [`", stringify!($f), "`](core::ops::", stringify!($op), "::", stringify!($f), ") function to the elements of both of these arrays.")]
                    fn $name(self, rhs: [T; N]) -> Self;
                }

                impl<T: core::ops::$op<Output = T> + Copy, const N: usize> $n<T, N> for [T; N] {
                    fn $name(self, rhs: [T; N]) -> Self {
                        use crate::Zip;
                        self.zip(rhs).map(|(a, b)| core::ops::$op::$f(a, b))
                    }
                }
            };
        }
    op!(Add, AAAdd, add, aadd);
    op!(BitAnd, AAAnd, bitand, aand);
    op!(BitOr, AAOr, bitor, aor);
    op!(BitXor, AAXor, bitxor, axor);
    op!(Div, AADiv, div, adiv);
    op!(Mul, AAMul, mul, amul);
    op!(Rem, AARem, rem, arem);
    op!(Shl, AAShl, shl, ashl);
    op!(Shr, AAShr, shr, ashr);
    op!(Sub, AASub, sub, asub);
}

/// see [`not`](core::ops::Not::not)
pub trait ANot<T, const N: usize> {
    /// apply the [`not`](core::ops::Not::not) function to each element of this array.
    fn not(self) -> Self;
}
impl<T: core::ops::Not<Output = T>, const N: usize> ANot<T, N> for [T; N] {
    fn not(self) -> Self {
        self.map(core::ops::Not::not)
    }
}

/// see [`neg`](core::ops::Not::not)
pub trait ANeg<T, const N: usize> {
    /// apply the [`not`](core::ops::Not::not) function to each element of this array.
    fn neg(self) -> Self;
}
impl<T: core::ops::Neg<Output = T>, const N: usize> ANeg<T, N> for [T; N] {
    fn neg(self) -> Self {
        self.map(core::ops::Neg::neg)
    }
}

/// Prelude for pervasive operations.
pub mod prelude {
    #[doc(inline)]
    pub use super::{array_and_array::*, array_and_scalar::*, scalar_and_array::*, ANeg, ANot};
}
#[test]
fn x() {
    use prelude::*;
    assert_eq!(2.mul([5, 2].add(5)), [20, 14]);
    assert_eq!(5.0.sub([2., 6.]), [3., -1.]);
}
