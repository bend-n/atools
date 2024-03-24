//! a collection of useful features for working with arrays
#![cfg_attr(not(test), no_std)]
#![allow(incomplete_features, internal_features)]
#![feature(
    generic_const_exprs,
    core_intrinsics,
    iter_intersperse,
    maybe_uninit_array_assume_init,
    inline_const,
    array_windows,
    iter_map_windows
)]
#![warn(
    clippy::undocumented_unsafe_blocks,
    clippy::missing_const_for_fn,
    clippy::missing_safety_doc,
    clippy::suboptimal_flops,
    unsafe_op_in_unsafe_fn,
    clippy::dbg_macro,
    clippy::use_self,
    missing_docs
)]

/// The prelude. You should
/// ```
/// use atools::prelude::*;
/// ```
pub mod prelude {
    #[doc(inline)]
    pub use super::{
        pervasive::prelude::*, range, splat, Array, ArrayTools, Chunked, CollectArray, Couple,
        DropFront, Flatten, Join, Pop, Trunc, Tuple,
    };
    #[doc(inline)]
    pub use core::array::from_fn;
}

use core::{array::from_fn, mem::ManuallyDrop as MD};
pub mod pervasive;
mod tuple;
pub use tuple::*;

/// Convenience function for when clonage is required; prefer `[T; N]` if possible. Also useful if `N` should be inferred.
pub fn splat<T: Clone, const N: usize>(a: T) -> [T; N] {
    from_fn(|_| a.clone())
}

const fn id<T>(x: T) -> T {
    x
}

/// Creates a array of indices.
/// ```
/// # use atools::prelude::*;
/// assert_eq!(range::<5>(), [0, 1, 2, 3, 4]);
/// ```
pub fn range<const N: usize>() -> [usize; N] {
    from_fn(id)
}

/// Collect an iterator into a array.
pub trait CollectArray<T> {
    /// Collect an iterator into a array.
    ///
    /// # Panics
    ///
    /// if the array isn't big enough.
    fn carr<const N: usize>(&mut self) -> [T; N];
}

impl<T, I: Iterator<Item = T>> CollectArray<T> for I {
    fn carr<const N: usize>(&mut self) -> [T; N] {
        from_fn(|_| self.next().unwrap())
    }
}

/// Pop parts of a array.
/// Use
/// ```
/// let [t, arr @ ..] = [1, 2];
/// ```
/// when possible. If the length of the array is a const generic, use
/// ```
/// # use atools::prelude::*;
/// let (t, arr) = [1, 2].pop_front();
/// ```
pub trait Pop<T, const N: usize> {
    /// Pop the front of a array.
    /// ```
    /// # use atools::prelude::*;
    /// let (t, arr) = b"abc".pop_front();
    /// # assert_eq!(t, b'a');
    /// # assert_eq!(arr, *b"bc");
    /// ```
    fn pop_front(self) -> (T, [T; N - 1]);
    /// Pop the back (end) of a array.
    /// ```
    /// # use atools::prelude::*;
    /// let (arr, t) = [0.1f32, 0.2, 0.3].pop();
    /// # assert_eq!(arr, [0.1, 0.2]);
    /// assert_eq!(t, 0.3);
    /// ```
    fn pop(self) -> ([T; N - 1], T);
}

impl<T, const N: usize> Pop<T, N> for [T; N] {
    fn pop_front(self) -> (T, [T; N - 1]) {
        // SAFETY: hi crater
        unsafe { core::intrinsics::transmute_unchecked(self) }
    }

    fn pop(self) -> ([T; N - 1], T) {
        // SAFETY: i am evil
        unsafe { core::intrinsics::transmute_unchecked(self) }
    }
}

/// Removes the last element of a array. The opposite of [`DropFront`].
pub trait Trunc<T, const N: usize> {
    /// Remove the last element of a array.
    /// You can think of this like <code>a.[pop()](Pop::pop).0</code>
    /// ```
    /// # use atools::prelude::*;
    /// let a = [1u64, 2].trunc();
    /// assert_eq!(a, [1]);
    /// ```
    fn trunc(self) -> [T; N - 1];
}

impl<const N: usize, T> Trunc<T, N> for [T; N] {
    fn trunc(self) -> [T; N - 1] {
        self.pop().0
    }
}

/// Remove the first element of a array. The opposite of [`Trunc`].
pub trait DropFront<T, const N: usize> {
    /// Removes the first element.
    fn drop_front(self) -> [T; N - 1];
}

impl<const N: usize, T> DropFront<T, N> for [T; N] {
    fn drop_front(self) -> [T; N - 1] {
        self.pop_front().1
    }
}

/// Join scalars together.
pub trait Join<T, const N: usize, const O: usize, U> {
    /// Join a array and an scalar together. For joining two arrays together, see [`Couple`].
    /// ```
    /// # use atools::prelude::*;
    /// let a = [1, 2].join(3);
    /// let b = 1.join([2, 3]);
    /// let c = 1.join(2).join(3);
    /// ```
    fn join(self, with: U) -> [T; N + O];
}

/// Couple two arrays together.
pub trait Couple<T, const N: usize, const O: usize> {
    /// Couple two arrays together. This could have been [`Join`], but the methods would require disambiguation.
    /// ```
    /// # use atools::prelude::*;
    /// let a = 1.join(2).couple([3, 4]);
    /// ```
    fn couple(self, with: [T; O]) -> [T; N + O];
}

impl<T, const N: usize, const O: usize> Couple<T, N, O> for [T; N] {
    fn couple(self, with: [T; O]) -> [T; N + O] {
        self.into_iter().chain(with).carr()
    }
}

impl<T, const N: usize> Join<T, N, 1, T> for [T; N] {
    fn join(self, with: T) -> [T; N + 1] {
        self.couple([with])
    }
}

impl<T> Join<T, 1, 1, T> for T {
    fn join(self, with: T) -> [T; 2] {
        [self, with]
    }
}

impl<T, const O: usize> Join<T, 1, O, [T; O]> for T {
    fn join(self, with: [T; O]) -> [T; 1 + O] {
        [self].couple(with)
    }
}

pub(crate) const fn assert_zero(x: usize) -> usize {
    if x != 0 {
        panic!("expected zero");
    } else {
        0
    }
}

/// 🍪
#[allow(private_bounds)]
pub trait Chunked<T, const N: usize> {
    /// Chunks.
    /// This will compile fail if `N ∤ (does not divide) C`
    /// ```
    /// # use atools::prelude::*;
    /// assert_eq!(range::<6>().chunked::<3>(), [[0, 1, 2], [3, 4, 5]]);
    /// ```
    #[allow(private_bounds)]
    fn chunked<const C: usize>(self) -> [[T; C]; N / C]
    where
        // N % C == 0
        [(); assert_zero(N % C)]:;
}

impl<const N: usize, T> Chunked<T, N> for [T; N] {
    #[allow(private_bounds)]
    fn chunked<const C: usize>(self) -> [[T; C]; N / C]
    where
        [(); assert_zero(N % C)]:,
    {
        // SAFETY: N != 0 && wont leak as N % C == 0.
        unsafe { MD::new(self).as_ptr().cast::<[[T; C]; N / C]>().read() }
    }
}

/// Flatten arrays.
pub trait Flatten<T, const N: usize, const N2: usize> {
    /// Takes a `[[T; N]; N2]`, and flattens it to a `[T; N * N2]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(generic_const_exprs)]
    /// # use atools::prelude::*;
    /// assert_eq!([[1, 2, 3], [4, 5, 6]].flatten(), [1, 2, 3, 4, 5, 6]);
    ///
    /// assert_eq!(
    ///     [[1, 2, 3], [4, 5, 6]].flatten(),
    ///     [[1, 2], [3, 4], [5, 6]].flatten(),
    /// );
    ///
    /// let array_of_empty_arrays: [[i32; 0]; 5] = [[], [], [], [], []];
    /// assert!(array_of_empty_arrays.flatten().is_empty());
    ///
    /// let empty_array_of_arrays: [[u32; 10]; 0] = [];
    /// assert!(empty_array_of_arrays.flatten().is_empty());
    /// ```
    fn flatten(self) -> [T; N * N2];
}

impl<T, const N: usize, const N2: usize> Flatten<T, N, N2> for [[T; N]; N2] {
    fn flatten(self) -> [T; N * N2] {
        // SAFETY: layout is the same.
        unsafe { core::intrinsics::transmute_unchecked(self) }
    }
}

/// Array tools.
pub trait ArrayTools<T, const N: usize> {
    /// Skip `BY` elements.
    fn skip<const BY: usize>(self) -> [T; N - BY];
    /// Skip every `BY` elements.
    ///
    /// ```
    /// # use atools::prelude::*;
    /// let x = range::<5>().step::<2>();
    /// assert_eq!(x, [0, 2, 4]);
    /// let x = range::<20>().step::<5>();
    /// assert_eq!(x, [0, 5, 10, 15]);
    /// assert_eq!(range::<50>().step::<3>(), [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]);
    /// ```
    fn step<const STEP: usize>(self) -> [T; 1 + (N - 1) / (STEP)];
    /// Zip arrays together.
    fn zip<U>(self, with: [U; N]) -> [(T, U); N];
    /// Intersperse a element in between items.
    /// ```
    /// # use atools::prelude::*;
    /// let x = range::<3>().intersperse(5);
    /// assert_eq!(x, [0, 5, 1, 5, 2]);
    /// ```
    fn intersperse(self, with: T) -> [T; (N * 2) - 1]
    where
        T: Clone;
    /// Run a function on every element.
    fn each(self, apply: impl FnMut(T));
    /// Embed the index.
    fn enumerate(self) -> [(T, usize); N];
    /// Take `M` elements, discarding the rest.
    /// ```
    /// # use atools::prelude::*;
    /// assert_eq!(range::<50>().take::<5>(), range::<5>());
    /// ```
    fn take<const M: usize>(self) -> [T; M];
    /// Get the sliding windows of this array.
    /// ```
    /// # use atools::prelude::*;
    /// assert_eq!(range::<5>().windowed::<2>(), [&[0, 1], &[1, 2], &[2, 3], &[3, 4]]);
    /// ```
    fn windowed<const W: usize>(&self) -> [&[T; W]; N - W + 1];
    /// Inspect every element of this array.
    fn inspect(self, f: impl FnMut(&T)) -> Self;
    /// Reverse this array.
    fn rev(self) -> Self;
    /// Interleave items from two arrays.
    /// ```
    /// # use atools::prelude::*;
    /// assert_eq!([0u8, 2, 4].interleave([1, 3, 5]), [0, 1, 2, 3, 4, 5]);
    /// ```
    fn interleave(self, with: [T; N]) -> [T; N * 2];
    /// [Cartesian product](https://en.wikipedia.org/wiki/Cartesian_product) (`A  ×  B`) of two arrays.
    /// ```
    /// # use atools::prelude::*;
    /// assert_eq!([1u64, 2].cartesian_product(&["Π", "Σ"]), [(1, "Π"), (1, "Σ"), (2, "Π"), (2, "Σ")]);
    /// ```
    fn cartesian_product<U: Clone, const M: usize>(&self, with: &[U; M]) -> [(T, U); N + M]
    where
        T: Clone;
    /// Sorts it. This uses <code>[[T](slice)]::[sort_unstable](slice::sort_unstable)</code>.
    fn sort(self) -> Self
    where
        T: Ord;
    /// Sum of the array.
    fn sum(self) -> T
    where
        T: core::iter::Sum<T>;
    /// Product of the array.
    fn product(self) -> T
    where
        T: core::iter::Product<T>;
}

impl<T, const N: usize> ArrayTools<T, N> for [T; N] {
    fn skip<const BY: usize>(self) -> [T; N - BY] {
        self.into_iter().skip(BY).carr()
    }
    fn step<const STEP: usize>(self) -> [T; 1 + (N - 1) / (STEP)] {
        self.into_iter().step_by(STEP).carr()
    }
    fn zip<U>(self, with: [U; N]) -> [(T, U); N] {
        self.into_iter().zip(with).carr()
    }

    fn intersperse(self, with: T) -> [T; (N * 2) - 1]
    where
        T: Clone,
    {
        self.into_iter().intersperse(with).carr()
    }

    fn each(self, apply: impl FnMut(T)) {
        self.into_iter().for_each(apply);
    }

    fn enumerate(self) -> [(T, usize); N] {
        let mut n = 0;
        self.map(|x| {
            let o = n;
            n += 1;
            (x, o)
        })
    }

    fn take<const M: usize>(self) -> [T; M] {
        self.into_iter().take(M).carr()
    }

    fn windowed<const W: usize>(&self) -> [&[T; W]; N - W + 1] {
        self.array_windows().carr()
    }

    fn inspect(self, f: impl FnMut(&T)) -> Self {
        self.iter().for_each(f);
        self
    }

    fn rev(self) -> Self {
        self.into_iter().rev().carr()
    }

    fn interleave(self, with: [T; N]) -> [T; N * 2] {
        let mut which = true;
        let mut a = self.into_iter();
        let mut b = with.into_iter();
        from_fn(|_| {
            which = !which;
            match which {
                false => a.next().unwrap(),
                true => b.next().unwrap(),
            }
        })
    }

    fn cartesian_product<U: Clone, const M: usize>(&self, with: &[U; M]) -> [(T, U); N + M]
    where
        T: Clone,
    {
        self.iter()
            .flat_map(|a| with.iter().map(move |b| (a.clone(), b.clone())))
            .carr()
    }

    fn sort(mut self) -> Self
    where
        T: Ord,
    {
        <[T]>::sort_unstable(&mut self);
        self
    }

    fn sum(self) -> T
    where
        T: core::iter::Sum<T>,
    {
        self.into_iter().sum()
    }

    fn product(self) -> T
    where
        T: core::iter::Product<T>,
    {
        self.into_iter().product()
    }
}
