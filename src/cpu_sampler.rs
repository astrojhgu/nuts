use num::Float;
use rand::Rng;
use std::fmt::Debug;
use thiserror::Error;

use crate::{
    adapt_strategy::{
        CombinedStrategy, DualAverageSettings, DualAverageStrategy, ExpWindowDiagAdapt, Limits,
    },
    cpu_potential::EuclideanPotential,
    mass_matrix::{DiagAdaptExpSettings, DiagMassMatrix},
    nuts::{NutsChain, NutsError, NutsOptions},
    CpuLogpFunc,
};

/// Settings for the NUTS sampler
#[derive(Clone, Copy)]
pub struct SamplerArgs<T>
where
    T: Copy,
{
    /// The number of tuning steps, where we fit the step size and mass matrix.
    pub num_tune: usize,
    /// The maximum tree depth during sampling. The number of leapfrog steps
    /// is smaller than 2 ^ maxdepth.
    pub maxdepth: usize,
    /// Store the gradient in the SampleStats
    pub store_gradient: bool,
    /// If the energy error is larger than this threshold we treat the leapfrog
    /// step as a divergence.
    pub max_energy_error: T,
    /// Settings for step size adaptation.
    pub step_size_adapt: DualAverageSettings<T>,
    /// Settings for mass matrix adaptation.
    pub mass_matrix_adapt: DiagAdaptExpSettings<T>,
}

impl<T> Default for SamplerArgs<T>
where
    T: Float,
{
    fn default() -> Self {
        Self {
            num_tune: 1000,
            maxdepth: 10,
            max_energy_error: T::from(1000).unwrap(),
            store_gradient: false,
            step_size_adapt: DualAverageSettings::default(),
            mass_matrix_adapt: DiagAdaptExpSettings::default(),
        }
    }
}

/// Propose new initial points for a sampler
///
/// This trait can be implemented by users to control how the different
/// chains should be initialized when using [`sample_parallel`].
pub trait InitPointFunc<T>
where
    T: Float,
{
    fn new_init_point<R: Rng + ?Sized>(&mut self, rng: &mut R, out: &mut [T]);
}

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum ParallelSamplingError {
    #[error("Could not send sample to controller thread")]
    ChannelClosed(),
    #[error("Nuts failed because of unrecoverable logp function error: {source}")]
    NutsError {
        #[from]
        source: NutsError,
    },
    #[error("Initialization of first point failed")]
    InitError { source: NutsError },
    #[error("Timeout occured while waiting for next sample")]
    Timeout,
    #[error("Drawing sample paniced")]
    Panic,
    #[error("Creating a logp function failed")]
    LogpFuncCreation {
        #[from]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

pub type ParallelChainResult = Result<(), ParallelSamplingError>;


pub type NutsChainT<T, F>=NutsChain<T, EuclideanPotential<T, F, DiagMassMatrix<T>>, CombinedStrategy<DualAverageStrategy<T, F, DiagMassMatrix<T>>, ExpWindowDiagAdapt<T, F>>>;

/// Create a new sampler
pub fn new_sampler<T: Copy + Float + Debug + Send + Limits<T> + 'static, F: CpuLogpFunc<T>>(
    logp: F,
    settings: SamplerArgs<T>,
) -> NutsChainT<T, F>
//-> impl Chain<T>
{
    use crate::nuts::AdaptStrategy;
    let num_tune = settings.num_tune;
    let step_size_adapt =
        DualAverageStrategy::<T, _, _>::new(settings.step_size_adapt, num_tune, logp.dim());
    let mass_matrix_adapt =
        ExpWindowDiagAdapt::<T, F>::new(settings.mass_matrix_adapt, num_tune, logp.dim());

    let strategy = CombinedStrategy::new(step_size_adapt, mass_matrix_adapt);

    let mass_matrix = DiagMassMatrix::new(logp.dim());
    let max_energy_error = settings.max_energy_error;
    let potential = EuclideanPotential::new(logp, mass_matrix, max_energy_error, T::one());

    let options = NutsOptions {
        maxdepth: settings.maxdepth,
        store_gradient: settings.store_gradient,
    };

    //let rng = { rand::rngs::StdRng::seed_from_usize(seed) };

    NutsChain::new(potential, strategy, options)
}

/// Initialize chains using uniform jitter around zero or some other provided value
#[derive(Default)]
pub struct JitterInitFunc<T> {
    mu: Option<Vec<T>>,
}

impl<T> JitterInitFunc<T> {
    /// Initialize new chains with jitter in [-1, 1] around zero
    pub fn new() -> JitterInitFunc<T> {
        JitterInitFunc { mu: None }
    }

    /// Initialize new chains with jitter in [mu - 1, mu + 1].
    pub fn new_with_mean(mu: Vec<T>) -> Self {
        Self { mu: Some(mu) }
    }
}

impl<T> InitPointFunc<T> for JitterInitFunc<T>
where
    T: Float,
{
    fn new_init_point<R: Rng + ?Sized>(&mut self, rng: &mut R, out: &mut [T]) {
        //rng.fill(out);
        let mut out1 = vec![0.0_f64; out.len()];
        rng.fill(&mut out1[..]);
        out.iter_mut()
            .zip(out1.iter())
            .for_each(|(a, &b)| *a = T::from(b).unwrap());
        let one = T::one();
        let two = one + one;
        if self.mu.is_none() {
            out.iter_mut().for_each(|val| *val = two * *val - one);
        } else {
            let mu = self.mu.as_ref().unwrap();
            out.iter_mut()
                .zip(mu.iter().copied())
                .for_each(|(val, mu)| *val = two * *val - one + mu);
        }
    }
}

