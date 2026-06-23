use core::ops::{RangeFrom, RangeInclusive, RangeTo, RangeToInclusive};
#[derive(Eq, PartialEq, Copy, Clone, core::marker::ConstParamTy)]
pub enum Range {
    Range(usize, usize),
    RangeFrom(usize),
    RangeTo(usize),
}

impl Range {
    pub const fn range<const N: usize>(self) -> (usize, usize) {
        match self {
            Range::Range(begin, end) => (begin, end),
            Range::RangeFrom(begin) => (begin, N),
            Range::RangeTo(end) => (0, end),
        }
    }
    pub const fn length<const N: usize>(self) -> usize {
        let (begin, end) = self.range::<N>();
        end - begin
    }
    pub const fn valid<const N: usize>(self) -> usize {
        (self.range::<N>().1 <= N) as usize
    }
}

#[diagnostic::on_unimplemented(
    message = "{Self} is not a valid range type",
    label = "use a correct range type, such as {{Range(x..y), RangeInclusive(x..=y), RangeTo(..x), RangeToInclusive(..=x)}}"
)]
pub const trait Ranged {
    fn range(self) -> Range;
}
const impl Ranged for core::ops::Range<usize> {
    fn range(self) -> Range {
        Range::Range(self.start, self.end)
    }
}
const impl Ranged for RangeInclusive<usize> {
    fn range(self) -> Range {
        Range::Range(*self.start(), *self.end() + 1)
    }
}

const impl Ranged for RangeFrom<usize> {
    fn range(self) -> Range {
        Range::RangeFrom(self.start)
    }
}
const impl Ranged for RangeTo<usize> {
    fn range(self) -> Range {
        Range::RangeTo(self.end)
    }
}
const impl Ranged for RangeToInclusive<usize> {
    fn range(self) -> Range {
        Range::RangeTo(self.end + 1)
    }
}
/// Constifies a range. For use with [`slice`](Slice::slice).
///
/// Takes a type in the form {[`Range`], [`RangeInclusive`], [`RangeTo`], [`RangeToInclusive`]}.
#[allow(private_bounds)]
pub const fn r<T: [const] Ranged>(x: T) -> Range {
    Ranged::range(x)
}

/// Slicing arrays up.
pub const trait Slice<T, const N: usize> {
    /// Slices the array.
    /// Compile time checked.
    /// ```
    /// # #![feature(generic_const_exprs, const_trait_impl)]
    /// # use atools::prelude::*;
    /// let x = atools::range::<5>();
    /// assert_eq!(*x.slice::<{ r(2..=4) }>(), [2, 3, 4]);
    /// // x.slice::<{ r(..10) }>(); // ERROR
    /// ```
    fn slice<const RANGE: Range>(&self) -> &[T; RANGE.length::<N>()]
    where
        // comptime length check
        [(); RANGE.valid::<N>() - 1]:;
}

const unsafe fn slice<T, const N: usize, const TO: usize>(x: &[T; N], begin: usize) -> &[T; TO] {
    // SAFETY: up to caller
    unsafe { &*x.as_ptr().add(begin).cast::<[T; TO]>() }
}

const impl<T, const N: usize> Slice<T, N> for [T; N] {
    fn slice<const RANGE: Range>(&self) -> &[T; RANGE.length::<N>()]
    where
        [(); RANGE.valid::<N>() - 1]:,
    {
        // SAFETY: the validity check ensures that the array will be in bounds.
        unsafe { slice::<T, N, { RANGE.length::<N>() }>(self, RANGE.range::<N>().0) }
    }
}

#[test]
fn slicing() {
    let x = [1, 2, 3];
    let &[y] = x.slice::<{ r(2..) }>();
    assert_eq!(y, 3);
    let &[y, z] = x.slice::<{ r(1..=2) }>();
    assert_eq!([y, z], [2, 3]);
    let &y = x.slice::<{ r(..=2) }>();
    assert_eq!(x, y);
    let &y = x.slice::<{ r(..2) }>();
    assert_eq!(y, [1, 2]);
}
