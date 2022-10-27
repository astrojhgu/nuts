use itertools::izip;
use num::traits::Float;

pub(crate) fn logaddexp<T>(a: T, b: T) -> T
where
    T: Float,
{
    if a == b {
        return a + T::from(2).unwrap().ln();
    }
    let diff = a - b;
    if diff > T::zero() {
        a + (-diff).exp().ln_1p()
    } else if diff < T::zero() {
        b + diff.exp().ln_1p()
    } else {
        // diff is NAN
        diff
    }
}

pub fn multiply<T>(x: &[T], y: &[T], out: &mut [T])
where
    T: Float,
{
    let n = x.len();
    assert!(y.len() == n);
    assert!(out.len() == n);

    izip!(out.iter_mut(), x.iter(), y.iter()).for_each(|(out, &x, &y)| {
        *out = x * y;
    });
}

pub fn scalar_prods2<T>(positive1: &[T], positive2: &[T], x: &[T], y: &[T]) -> (T, T)
where
    T: Float,
{
    let n = positive1.len();

    assert!(positive1.len() == n);
    assert!(positive2.len() == n);
    assert!(x.len() == n);
    assert!(y.len() == n);

    izip!(positive1, positive2, x, y).fold((T::zero(), T::zero()), |(s1, s2), (&a, &b, &c, &d)| {
        (s1 + c * (a + b), s2 + d * (a + b))
    })
}

pub fn scalar_prods3<T>(
    positive1: &[T],
    negative1: &[T],
    positive2: &[T],
    x: &[T],
    y: &[T],
) -> (T, T)
where
    T: Float,
{
    let n = positive1.len();

    assert!(positive1.len() == n);
    assert!(positive2.len() == n);
    assert!(negative1.len() == n);
    assert!(x.len() == n);
    assert!(y.len() == n);

    izip!(positive1, negative1, positive2, x, y)
        .fold((T::zero(), T::zero()), |(s1, s2), (&a, &b, &c, &x, &y)| {
            (s1 + x * (a - b + c), s2 + y * (a - b + c))
        })
}

pub fn vector_dot<T>(a: &[T], b: &[T]) -> T
where
    T: Float,
{
    assert!(a.len() == b.len());

    let mut result = T::zero();
    for (&val1, &val2) in a.iter().zip(b) {
        result = result + val1 * val2;
    }
    result
}

pub fn axpy<T>(x: &[T], y: &mut [T], a: T)
where
    T: Float,
{
    let n = x.len();
    assert!(y.len() == n);

    izip!(x, y).for_each(|(x, y)| {
        *y = x.mul_add(a, *y);
    });
}
pub fn axpy_out<T>(x: &[T], y: &[T], a: T, out: &mut [T])
where
    T: Float,
{
    let n = x.len();
    assert!(y.len() == n);
    assert!(out.len() == n);

    izip!(x, y, out).for_each(|(&x, &y, out)| {
        *out = a.mul_add(x, y);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use ndarray::prelude::*;
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    fn assert_approx_eq(a: f64, b: f64) {
        if a.is_nan() {
            if b.is_nan() | b.is_infinite() {
                return;
            }
        }
        if b.is_nan() {
            if a.is_nan() | a.is_infinite() {
                return;
            }
        }
        assert_ulps_eq!(a, b);
    }

    prop_compose! {
        fn array2(maxsize: usize) (size in 0..maxsize) (
            vec1 in prop::collection::vec(prop::num::f64::ANY, size),
            vec2 in prop::collection::vec(prop::num::f64::ANY, size)
        )
        -> (Vec<f64>, Vec<f64>) {
            (vec1, vec2)
        }
    }

    prop_compose! {
        fn array3(maxsize: usize) (size in 0..maxsize) (
            vec1 in prop::collection::vec(prop::num::f64::ANY, size),
            vec2 in prop::collection::vec(prop::num::f64::ANY, size),
            vec3 in prop::collection::vec(prop::num::f64::ANY, size)
        )
        -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            (vec1, vec2, vec3)
        }
    }

    prop_compose! {
        fn array4(maxsize: usize) (size in 0..maxsize) (
            vec1 in prop::collection::vec(prop::num::f64::ANY, size),
            vec2 in prop::collection::vec(prop::num::f64::ANY, size),
            vec3 in prop::collection::vec(prop::num::f64::ANY, size),
            vec4 in prop::collection::vec(prop::num::f64::ANY, size)
        )
        -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
            (vec1, vec2, vec3, vec4)
        }
    }

    prop_compose! {
        fn array5(maxsize: usize) (size in 0..maxsize) (
            vec1 in prop::collection::vec(prop::num::f64::ANY, size),
            vec2 in prop::collection::vec(prop::num::f64::ANY, size),
            vec3 in prop::collection::vec(prop::num::f64::ANY, size),
            vec4 in prop::collection::vec(prop::num::f64::ANY, size),
            vec5 in prop::collection::vec(prop::num::f64::ANY, size)
        )
        -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
            (vec1, vec2, vec3, vec4, vec5)
        }
    }

    proptest! {
        #[test]
        fn check_logaddexp(x in -10f64..10f64, y in -10f64..10f64) {
            let a = (x.exp() + y.exp()).ln();
            let b = logaddexp(x, y);
            let neginf = std::f64::NEG_INFINITY;
            let nan = std::f64::NAN;
            prop_assert!((a - b).abs() < 1e-10);
            prop_assert_eq!(b, logaddexp(y, x));
            prop_assert_eq!(x, logaddexp(x, neginf));
            prop_assert_eq!(logaddexp(neginf, neginf), neginf);
            prop_assert!(logaddexp(nan, x).is_nan());
        }

        #[test]
        fn test_axpy((x, y) in array2(10), a in prop::num::f64::ANY) {
            let orig = y.clone();
            let mut y = y.clone();
            axpy(&x[..], &mut y[..], a);
            for ((&x, y), out) in x.iter().zip(orig).zip(y) {
                assert_approx_eq(out, a * x + y);
            }
        }

        #[test]
        fn test_scalar_prods2((x1, x2, y1, y2) in array4(10)) {
            let (p1, p2) = scalar_prods2(&x1[..], &x2[..], &y1[..], &y2[..]);
            let x1 = ndarray::Array1::from_vec(x1);
            let x2 = ndarray::Array1::from_vec(x2);
            let y1 = ndarray::Array1::from_vec(y1);
            let y2 = ndarray::Array1::from_vec(y2);
            assert_approx_eq(p1, (&x1 + &x2).dot(&y1));
            assert_approx_eq(p2, (&x1 + &x2).dot(&y2));
        }

        #[test]
        fn test_scalar_prods3((x1, x2, x3, y1, y2) in array5(10)) {
            let (p1, p2) = scalar_prods3(&x1[..], &x2[..], &x3[..], &y1[..], &y2[..]);
            let x1 = ndarray::Array1::from_vec(x1);
            let x2 = ndarray::Array1::from_vec(x2);
            let x3 = ndarray::Array1::from_vec(x3);
            let y1 = ndarray::Array1::from_vec(y1);
            let y2 = ndarray::Array1::from_vec(y2);
            assert_approx_eq(p1, (&x1 - &x2 + &x3).dot(&y1));
            assert_approx_eq(p2, (&x1 - &x2 + &x3).dot(&y2));
        }

        #[test]
        fn test_axpy_out(a in prop::num::f64::ANY, (x, y, out) in array3(10)) {
            let mut out = out.clone();
            axpy_out(&x[..], &y[..], a, &mut out[..]);
            let x = ndarray::Array1::from_vec(x);
            let mut y = ndarray::Array1::from_vec(y);
            y.scaled_add(a, &x);
            for (&out1, out2) in out.iter().zip(y) {
                assert_approx_eq(out1, out2);
            }
        }

        #[test]
        fn test_multiplty((x, y, out) in array3(10)) {
            let mut out = out.clone();
            multiply(&x[..], &y[..], &mut out[..]);
            let x = ndarray::Array1::from_vec(x);
            let y = ndarray::Array1::from_vec(y);
            for (&out1, out2) in out.iter().zip(&x * &y) {
                assert_approx_eq(out1, out2);
            }
        }
    }

    #[test]
    fn check_neginf() {
        assert_eq!(logaddexp(std::f64::NEG_INFINITY, 2.), 2.);
        assert_eq!(logaddexp(2., std::f64::NEG_INFINITY), 2.);
    }
}
