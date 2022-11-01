use crate::{
    adapt_strategy::{
        CombinedStrategy, DualAverageSettings, DualAverageStrategy, ExpWindowDiagAdapt, Limits,
    },
    cpu_potential::{CpuLogpFunc, EuclideanPotential},
    mass_matrix::{DiagAdaptExpSettings, DiagMassMatrix},
    nuts::{Chain, NutsChain, NutsError, NutsOptions, SampleStats},
};
use num::Float;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fmt::Debug;

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

pub type NutsChainT<T, F> = NutsChain<
    T,
    EuclideanPotential<T, F, DiagMassMatrix<T>>,
    CombinedStrategy<DualAverageStrategy<T, F, DiagMassMatrix<T>>, ExpWindowDiagAdapt<T, F>>,
>;

/// Create a new sampler
pub fn new_sampler<T: Copy + Float + Debug + Send + Limits + 'static, F: CpuLogpFunc<T>>(
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

pub fn sample_sequentially<T, F>(
    logp: F,
    settings: SamplerArgs<T>,
    start: &[T],
    draws: usize,
    seed: u64,
) -> Result<impl Iterator<Item = Result<(Vec<T>, impl SampleStats<T>), NutsError>>, NutsError>
where
    T: Float + Debug + Send + Limits + 'static,
    F: CpuLogpFunc<T>,
{
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut sampler = new_sampler(logp, settings);
    sampler.set_position(start)?;
    Ok((0..draws).into_iter().map(move |_| sampler.draw(&mut rng)))
}
