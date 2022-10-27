use itertools::izip;
use num::Float;
use rand_distr::{Distribution, StandardNormal};

use std::fmt::Debug;

use crate::{
    cpu_state::{InnerState, State},
    math::{multiply, vector_dot},
    nuts::Collector,
};

pub trait MassMatrix<T>
where
    T: Clone,
{
    fn update_velocity(&self, state: &mut InnerState<T>);
    fn update_kinetic_energy(&self, state: &mut InnerState<T>);
    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut InnerState<T>, rng: &mut R);
}

pub struct NullCollector {}

impl<T> Collector<T> for NullCollector
where
    T: Debug + Clone + Float,
{
    type State = State<T>;
}

#[derive(Debug)]
pub struct DiagMassMatrix<T> {
    inv_stds: Box<[T]>,
    pub variance: Box<[T]>,
}

impl<T> DiagMassMatrix<T>
where
    T: Float + Debug,
{
    pub fn new(ndim: usize) -> Self {
        Self {
            inv_stds: vec![T::zero(); ndim].into(),
            variance: vec![T::zero(); ndim].into(),
        }
    }

    pub fn update_diag(&mut self, new_variance: impl Iterator<Item = T>) {
        update_diag(&mut self.variance, &mut self.inv_stds, new_variance);
    }
}

fn update_diag<T>(
    variance_out: &mut [T],
    inv_std_out: &mut [T],
    new_variance: impl Iterator<Item = T>,
) where
    T: Float + Debug,
{
    izip!(variance_out, inv_std_out, new_variance,).for_each(|(var, inv_std, x)| {
        assert!(x.is_finite(), "Illegal value on mass matrix: {:?}", x);
        assert!(x > T::zero(), "Illegal value on mass matrix: {:?}", x);
        *var = x;
        *inv_std = (T::one() / x).sqrt();
    });
}

impl<T> MassMatrix<T> for DiagMassMatrix<T>
where
    T: Clone + Float,
    StandardNormal: Distribution<T>,
{
    fn update_velocity(&self, state: &mut InnerState<T>) {
        multiply(&self.variance, &state.p, &mut state.v);
    }

    fn update_kinetic_energy(&self, state: &mut InnerState<T>) {
        state.kinetic_energy = T::from(0.5).unwrap() * vector_dot(&state.p, &state.v);
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut InnerState<T>, rng: &mut R) {
        let dist = rand_distr::StandardNormal;
        state
            .p
            .iter_mut()
            .zip(self.inv_stds.iter())
            .for_each(|(p, &s)| {
                let norm: T = rng.sample(dist);
                *p = s * norm;
            });
    }
}

#[derive(Debug)]
pub struct ExpWeightedVariance<T> {
    mean: Box<[T]>,
    variance: Box<[T]>,
    count: u64,
    pub alpha: T, // TODO
    pub use_mean: bool,
}

impl<T> ExpWeightedVariance<T>
where
    T: Float,
{
    pub fn new(dim: usize, alpha: T, use_mean: bool) -> Self {
        ExpWeightedVariance {
            mean: vec![T::zero(); dim].into(),
            variance: vec![T::zero(); dim].into(),
            count: 0,
            alpha,
            use_mean,
        }
    }

    pub fn set_mean(&mut self, values: impl Iterator<Item = T>) {
        self.mean
            .iter_mut()
            .zip(values)
            .for_each(|(out, val)| *out = val);
    }

    pub fn set_variance(&mut self, values: impl Iterator<Item = T>) {
        self.variance
            .iter_mut()
            .zip(values)
            .for_each(|(out, val)| *out = val);
    }

    pub fn add_sample(&mut self, value: impl Iterator<Item = T>) {
        add_sample(self, value);
        self.count += 1;
    }

    pub fn current(&self) -> &[T] {
        &self.variance
    }

    pub fn count(&self) -> u64 {
        self.count
    }
}

fn add_sample<T>(self_: &mut ExpWeightedVariance<T>, value: impl Iterator<Item = T>)
where
    T: Float,
{
    if self_.use_mean {
        izip!(value, self_.mean.iter_mut(), self_.variance.iter_mut()).for_each(
            |(x, mean, var)| {
                //if self_.count > 1 {
                //    assert!(x - *mean != T::zero(), "var = {}, mean = {}, x = {}, delta = {}, count = {}", var, mean, x, x - *mean, self_.count);
                //}
                let delta = x - *mean;
                //*mean += self_.alpha * delta;
                *mean = self_.alpha.mul_add(delta, *mean);
                *var = (T::one() - self_.alpha) * (*var + self_.alpha * delta * delta);
            },
        );
    } else {
        izip!(value, self_.mean.iter_mut(), self_.variance.iter_mut()).for_each(
            |(x, _mean, var)| {
                let delta = x;
                *var = (T::one() - self_.alpha) * (*var + self_.alpha * delta * delta);
                //assert!(*var > T::zero(), "var = {}, x = {}, delta = {}", var, x, delta);
            },
        );
    }
}

/// Settings for mass matrix adaptation
#[derive(Clone, Copy)]
pub struct DiagAdaptExpSettings<T> {
    /// An exponenital decay parameter for the variance estimator
    pub variance_decay: T,
    /// Exponenital decay parameter for the variance estimator in the first adaptation window
    pub early_variance_decay: T,
    /// Stop adaptation `final_window` draws before tuning ends.
    pub final_window: u64,
    /// Save the current adapted mass matrix as sampler stat
    pub store_mass_matrix: bool,
    /// Switch to a new variance estimator every `window_switch_freq` draws.
    pub window_switch_freq: u64,
    pub grad_init: bool,
}

impl<T> Default for DiagAdaptExpSettings<T>
where
    T: Float,
{
    fn default() -> Self {
        Self {
            variance_decay: T::from(0.02).unwrap(),
            final_window: 50,
            store_mass_matrix: false,
            window_switch_freq: 50,
            early_variance_decay: T::from(0.1).unwrap(),
            grad_init: false,
        }
    }
}

pub struct DrawGradCollector<T> {
    pub draw: Box<[T]>,
    pub grad: Box<[T]>,
    pub is_good: bool,
}

impl<T> DrawGradCollector<T>
where
    T: Float,
{
    pub fn new(dim: usize) -> Self {
        DrawGradCollector {
            draw: vec![T::zero(); dim].into(),
            grad: vec![T::zero(); dim].into(),
            is_good: true,
        }
    }
}

impl<T> Collector<T> for DrawGradCollector<T>
where
    T: Clone + Debug + Float,
{
    type State = State<T>;

    fn register_draw(&mut self, state: &Self::State, info: &crate::nuts::SampleInfo<T>) {
        self.draw.copy_from_slice(&state.q);
        self.grad.copy_from_slice(&state.grad);
        let idx = state.index_in_trajectory();
        if info.divergence_info.is_some() {
            self.is_good = (idx <= -4) | (idx >= 4);
        } else {
            self.is_good = idx != 0;
        }
    }
}
