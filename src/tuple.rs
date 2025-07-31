/// Turn a tuple into a array. Implemented for N≤32
pub trait Array<const N: usize, T> {
    /// Turn a tuple into a array
    fn array(self) -> [T; N];
}

/// Turn a array into a tuple. Implemented for N≤32
pub trait Tuple<O> {
    /// Turn a array into a tuple.
    fn tuple(self) -> O;
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
macro_rules! do_impl {(
    [$_:tt] // `$` sigil
    $($i:tt)*
) => (
    impl<T> Tuple<($($i, )*)> for [T; 0 $(+ { stringify!($i); 1 } )*] {
        fn tuple(self) -> ($($i, )*) {
            with_vars! {
                [acc: ]
                [to_munch: $($i)*]
                |$_ x| {
                    let [$_($x, )*] = self;
                    ($_($x, )*)
                }
            }
        }
    }

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
    }
)}

generate! {
    T T T T  T T T T
    T T T T  T T T T
    T T T T  T T T T
    T T T T  T T T T
    T T T T  T T T T
}
