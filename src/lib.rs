//! a collection of useful features for working with arrays
#![cfg_attr(not(test), no_std)]
#![allow(incomplete_features, internal_features)]
#![feature(
    const_destruct,
    adt_const_params,
    generic_const_exprs,
    core_intrinsics,
    iter_intersperse,
    const_trait_impl,
    maybe_uninit_array_assume_init,
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
use core::{
    array::from_fn,
    intrinsics::transmute_unchecked,
    marker::Destruct,
    mem::{offset_of, MaybeUninit as MU},
};
pub mod pervasive;
mod slice;
mod tuple;
#[doc(inline)]
pub use slice::Slice;
pub use tuple::*;

/// The prelude. You should
/// ```
/// use atools::prelude::*;
/// ```
pub mod prelude {
    #[doc(inline)]
    pub use super::{
        pervasive::prelude::*, range, slice::r, slice::Slice, splat, Array, ArrayTools, Chunked,
        CollectArray, Couple, Deconstruct, Flatten, Join, Split, Tuple, Zip,
    };
    #[doc(inline)]
    pub use core::array::from_fn;
}

#[repr(C)]
struct Pair<X, Y>(X, Y);
impl<X, Y> Pair<X, Y> {
    const fn tuple() -> bool {
        (size_of::<(X, Y)>() == size_of::<Self>())
            & (offset_of!(Self, 0) == offset_of!((X, Y), 0))
            & (offset_of!(Self, 1) == offset_of!((X, Y), 1))
    }

    const fn into(self) -> (X, Y) {
        if Self::tuple() {
            // SAFETY: we are a tuple!!!
            unsafe { transmute_unchecked::<Self, (X, Y)>(self) }
        } else {
            // SAFETY: this is safe.
            let out = unsafe { (core::ptr::read(&self.0), core::ptr::read(&self.1)) };
            core::mem::forget(self);
            out
        }
    }

    const unsafe fn splat<T>(x: T) -> (X, Y) {
        assert!(core::mem::size_of::<T>() == core::mem::size_of::<Pair<X, Y>>());
        // SAFETY: well.
        unsafe { transmute_unchecked::<_, Self>(x) }.into()
    }
}

/// Convenience function for when clonage is required; prefer `[T; N]` if possible. Also useful if `N` should be inferred.
pub fn splat<T: Clone, const N: usize>(a: T) -> [T; N] {
    from_fn(|_| a.clone())
}

/// Creates a array of indices.
/// ```
/// # #![feature(generic_const_exprs)]
/// # use atools::prelude::*;
/// assert_eq!(range::<5>(), [0, 1, 2, 3, 4]);
/// ```
pub const fn range<const N: usize>() -> [usize; N] {
    let mut out = unsafe { MU::<[MU<usize>; N]>::uninit().assume_init() };
    let mut i = 0usize;
    while i < out.len() {
        out[i] = MU::new(i);
        i += 1;
    }
    unsafe { transmute_unchecked(out) }
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

/// Deconstruct some array.
/// Use
/// ```
/// let [t, arr @ ..] = [1, 2];
/// ```
/// when possible. If the length of the array is a const generic, use
/// ```
/// # #![feature(generic_const_exprs)]
/// # use atools::prelude::*;
/// let (t, arr) = [1, 2].uncons();
/// ```
#[const_trait]
pub trait Deconstruct<T, const N: usize> {
    /// Gives you the <code>[[head](Deconstruct_::head), [tail](Deconstruct_::tail) @ ..]</code>
    /// ```
    /// # #![feature(generic_const_exprs)]
    /// # use atools::prelude::*;
    /// let (t, arr) = b"abc".uncons();
    /// # assert_eq!(t, b'a');
    /// # assert_eq!(arr, *b"bc");
    /// ```
    fn uncons(self) -> (T, [T; N - 1]);
    /// Gives you the <code>[[init](Deconstruct_::init) @ .., [last](Deconstruct_::last)]</code>
    /// ```
    /// # #![feature(generic_const_exprs)]
    /// # use atools::prelude::*;
    /// let (arr, t) = [0.1f32, 0.2, 0.3].unsnoc();
    /// # assert_eq!(arr, [0.1, 0.2]);
    /// assert_eq!(t, 0.3);
    /// ```
    fn unsnoc(self) -> ([T; N - 1], T);
}

/// Deconstruct some array. (dropping edition).
///
/// <img src="https://media.discordapp.net/attachments/273541645579059201/1404772577259294770/listmonster.png?ex=689c67e9&is=689b1669&hm=00525f7bb8ffb2eb096a46d10509ebf8def669ca3175a713df686ff4be7a4e67">
pub trait Deconstruct_<T, const N: usize> {
    /// Gives you a <code>[[_](Deconstruct_::init) @ .., last]</code>.
    /// See also [`unsnoc`](Deconstruct::unsnoc).
    fn last(self) -> T;
    /// Gives you a <code>[init @ .., [_](Deconstruct_::last)]</code>
    /// See also [`unsnoc`](Deconstruct::unsnoc).
    /// ```
    /// # #![feature(generic_const_exprs)]
    /// # use atools::prelude::*;
    /// let a = *[1u64, 2, 3].init();
    /// assert_eq!(a, [1, 2]);
    /// ```
    fn init(self) -> [T; N - 1];
    /// Gives you a <code>[head, [_](Deconstruct_::tail) @ ..]</code>.
    /// See also [`uncons`](Deconstruct::uncons).
    fn head(self) -> T;
    /// Gives you a <code>[[_](Deconstruct_::head), tail @ ..]</code>.
    /// See also [`uncons`](Deconstruct::uncons).
    fn tail(self) -> [T; N - 1];
}

impl<T, const N: usize> const Deconstruct<T, N> for [T; N] {
    #[doc(alias = "pop_front")]
    fn uncons(self) -> (T, [T; N - 1]) {
        // SAFETY: the layout is alright.
        unsafe { Pair::splat(self) }
    }

    #[doc(alias = "pop")]
    fn unsnoc(self) -> ([T; N - 1], T) {
        // SAFETY: the layout is still alright.
        unsafe { Pair::splat(self) }
    }
}

impl<T, const N: usize> Deconstruct_<T, N> for [T; N]
where
    [(); N - 1]:,
{
    fn last(self) -> T {
        self.unsnoc().1
    }
    #[doc(alias = "trunc")]
    fn init(self) -> [T; N - 1] {
        self.unsnoc().0
    }
    fn head(self) -> T {
        self.uncons().0
    }
    fn tail(self) -> [T; N - 1] {
        self.uncons().1
    }
}

/// Join scalars together.
#[const_trait]
pub trait Join<T, const N: usize, const O: usize, U> {
    /// Join a array and an scalar together. For joining two arrays together, see [`Couple`].
    /// ```
    /// # #![feature(generic_const_exprs)]
    /// # use atools::prelude::*;
    /// let a = [1, 2].join(3);
    /// let b = 1.join([2, 3]);
    /// let c = 1.join(2).join(3);
    /// ```
    fn join(self, with: U) -> [T; N + O];
}

/// Couple two arrays together.
#[const_trait]
pub trait Couple<T, const N: usize, const O: usize> {
    /// Couple two arrays together. This could have been [`Join`], but the methods would require disambiguation.
    /// ```
    /// # #![feature(generic_const_exprs)]
    /// # use atools::prelude::*;
    /// let a = 1.join(2).couple([3, 4]);
    /// ```
    fn couple(self, with: [T; O]) -> [T; N + O];
}

impl<T, const N: usize, const O: usize> const Couple<T, N, O> for [T; N] {
    fn couple(self, with: [T; O]) -> [T; N + O] {
        // SAFETY: adjacent
        unsafe { transmute_unchecked(Pair(self, with)) }
    }
}

impl<T, const N: usize> const Join<T, N, 1, T> for [T; N] {
    fn join(self, with: T) -> [T; N + 1] {
        self.couple([with])
    }
}

impl<T> const Join<T, 1, 1, T> for T {
    fn join(self, with: T) -> [T; 2] {
        [self, with]
    }
}

impl<T, const O: usize> const Join<T, 1, O, [T; O]> for T {
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
#[const_trait]
pub trait Chunked<T, const N: usize> {
    /// Chunks.
    /// This will compile fail if `N ∤ (does not divide) C`
    /// ```
    /// # #![feature(generic_const_exprs)]
    /// # use atools::prelude::*;
    /// assert_eq!(range::<6>().chunked::<3>(), [[0, 1, 2], [3, 4, 5]]);
    /// ```
    #[allow(private_bounds)]
    fn chunked<const C: usize>(self) -> [[T; C]; N / C]
    where
        // N % C == 0
        [(); assert_zero(N % C)]:;
}

impl<const N: usize, T> const Chunked<T, N> for [T; N] {
    #[allow(private_bounds)]
    fn chunked<const C: usize>(self) -> [[T; C]; N / C]
    where
        [(); assert_zero(N % C)]:,
    {
        // SAFETY: N != 0 && wont leak as N % C == 0.
        let retval = unsafe { self.as_ptr().cast::<[[T; C]; N / C]>().read() };
        core::mem::forget(self);
        retval
    }
}

/// Flatten arrays.
#[const_trait]
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

impl<T, const N: usize, const N2: usize> const Flatten<T, N, N2> for [[T; N]; N2] {
    fn flatten(self) -> [T; N * N2] {
        // SAFETY: layout is the same.
        unsafe { core::intrinsics::transmute_unchecked(self) }
    }
}

#[const_trait]
/// Splitting arrays up.
pub trait Split<T, const N: usize> {
    /// Splits the array into twain.
    /// ```
    /// # #![feature(generic_const_exprs)]
    /// # use atools::prelude::*;
    /// let x = [1u8, 2, 3];
    /// let ([x], [y, z]) = x.split::<1>();
    /// ```
    fn split<const AT: usize>(self) -> ([T; AT], [T; N - AT]);
    /// Take `AT` elements, discarding the rest.
    /// ```
    /// # #![feature(generic_const_exprs)]
    /// # use atools::prelude::*;
    /// assert_eq!(range::<50>().take::<5>(), range::<5>());
    /// ```
    fn take<const AT: usize>(self) -> [T; AT]
    where
        [(); N - AT]:,
        T: [const] Destruct;
    /// Discard `AT` elements, returning the rest.
    fn drop<const AT: usize>(self) -> [T; N - AT]
    where
        T: [const] Destruct;
}

impl<T, const N: usize> const Split<T, N> for [T; N] {
    fn split<const AT: usize>(self) -> ([T; AT], [T; N - AT]) {
        // SAFETY: N - AT overflows when AT > N so the size of the returned "array" is the same.
        unsafe { Pair::splat(self) }
    }
    fn take<const M: usize>(self) -> [T; M]
    where
        // M <= N
        [(); N - M]:,
        T: [const] Destruct,
    {
        self.split::<M>().0
    }
    fn drop<const M: usize>(self) -> [T; N - M]
    where
        T: [const] Destruct,
    {
        self.split::<M>().1
    }
}
#[const_trait]
/// Zip arrays together.
pub trait Zip<T, const N: usize> {
    /// Zip arrays together.
    fn zip<U>(self, with: [U; N]) -> [(T, U); N];
}

impl<T, const N: usize> const Zip<T, N> for [T; N] {
    fn zip<U>(self, with: [U; N]) -> [(T, U); N] {
        let mut out = unsafe { MU::<[MU<_>; N]>::uninit().assume_init() };
        let mut i = 0usize;
        while i < out.len() {
            out[i] = MU::new(unsafe { (self.as_ptr().add(i).read(), with.as_ptr().add(i).read()) });
            i += 1;
        }
        core::mem::forget((self, with));
        unsafe { transmute_unchecked(out) }
    }
}

/// Array tools.
pub trait ArrayTools<T, const N: usize> {
    /// Skip `BY` elements.
    fn skip<const BY: usize>(self) -> [T; N - BY];
    /// Skip every `BY` elements.
    ///
    /// ```
    /// # #![feature(generic_const_exprs)]
    /// # use atools::prelude::*;
    /// let x = range::<5>().step::<2>();
    /// assert_eq!(x, [0, 2, 4]);
    /// let x = range::<20>().step::<5>();
    /// assert_eq!(x, [0, 5, 10, 15]);
    /// assert_eq!(range::<50>().step::<3>(), [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]);
    /// ```
    fn step<const STEP: usize>(self) -> [T; 1 + (N - 1) / (STEP)];
    /// Intersperse a element in between items.
    /// ```
    /// # #![feature(generic_const_exprs)]
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
    /// Get the sliding windows of this array.
    /// ```
    /// # #![feature(generic_const_exprs)]
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
    /// # #![feature(generic_const_exprs)]
    /// # use atools::prelude::*;
    /// assert_eq!([0u8, 2, 4].interleave([1, 3, 5]), [0, 1, 2, 3, 4, 5]);
    /// ```
    fn interleave(self, with: [T; N]) -> [T; N * 2];
    /// [Cartesian product](https://en.wikipedia.org/wiki/Cartesian_product) (`A  ×  B`) of two arrays.
    /// ```
    /// # #![feature(generic_const_exprs)]
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

    fn windowed<const W: usize>(&self) -> [&[T; W]; N - W + 1] {
        self.array_windows().carr()
    }

    fn inspect(self, f: impl FnMut(&T)) -> Self {
        self.iter().for_each(f);
        self
    }

    fn rev(mut self) -> Self {
        self.reverse();
        self
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
