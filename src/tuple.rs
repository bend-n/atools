mod seal {
    pub trait Seal {}
    impl<T, const N: usize> Seal for [T; N] {}
}
use seal::Seal;
/// Turn a tuple into a array. Implemented for N≤32
pub trait Array<const N: usize, T>: Sized + Seal {
    /// Turn a tuple into a array
    fn array(self) -> [T; N];
    /// Turn a array into a tuple.
    /// Prefer postfix [`Tuple::tuple`].
    fn tuple(x: [T; N]) -> Self;
}

/// Turn a array into a tuple. Implemented for N≤32
pub trait Tuple<T, const N: usize, O: Array<N, T>> {
    /// Turn a array into a tuple.
    fn tuple(self) -> O;
}

impl<T, const N: usize, O: Array<N, T>> Tuple<T, N, O> for [T; N] {
    fn tuple(self) -> O {
        Array::tuple(self)
    }
}
// thanks yandros
macro_rules! with_vars {
    (
        [acc: $($acc:tt)*]
        [to_munch: T $($rest:tt)*]
        $($cb:tt)*
    ) => (with_vars! {
        [acc: $($acc)* x]
        [to_munch: $($rest)*]
        $($cb)*
    });

    (
        [acc: $($var:ident)*]
        [to_munch: /* nothing */]
        |$_:tt $metavar:ident| $body:tt
    ) => ({
        macro_rules! __emit__ {(
            $_( $_ $metavar:ident)*
        ) =>
            $body
        }
        __emit__! { $($var)* }
    });
}

macro_rules! generate {(
    $($Hd:tt $($T:tt)*)?
) => (
    $(generate! { $($T)* })?
    do_impl! { [$] $($Hd $($T)*)? }
)}
macro_rules! do_impl {
([$_:tt]) => {};
(
    [$_:tt] // `$` sigil
    $($i:tt)+
) => (
    impl<T> Seal for ($($i, )*) {}
    impl<T> Array<{ 0 $(+ { stringify!($i); 1 } )* }, T> for ($($i, )*) {
        fn array(self) -> [T; 0 $(+ { stringify!($i); 1 } )*] {
            with_vars! {
                [acc: ]
                [to_munch: $($i)*]
                |$_ x| {
                    let ($_($_ x, )*) = self;
                    [$_($_ x, )*]
                }
            }
        }
        fn tuple(x: [T;  0 $(+ { stringify!($i); 1 } )*]) -> ($($i, )*) {
            with_vars! {
                [acc: ]
                [to_munch: $($i)*]
                |$_ x| {
                    let [$_($x, )*] = x;
                    ($_($x, )*)
                }
            }
        }
    }
);
}

generate! {
    T T T T  T T T T
    T T T T  T T T T
    T T T T  T T T T
    T T T T  T T T T
    T T T T  T T T T
}
