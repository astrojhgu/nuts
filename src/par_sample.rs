use std::{thread::JoinHandle
    , fmt::Debug
};

use crate::{
    cpu_potential::CpuLogpFunc,
    cpu_sampler::{SamplerArgs, new_sampler},
    nuts::{NutsError, SampleStats, Chain}, adapt_strategy::Limits,

};
use num::traits::Float;
use rand::{Rng, rngs::{StdRng, SmallRng}, SeedableRng};
use rayon::prelude::*;
use thiserror::Error;


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

pub trait CpuLogpFuncMaker<T>: Send + Sync {
    type Func: CpuLogpFunc<T>;

    fn make_logp_func(&self) -> Result<Self::Func, Box<dyn std::error::Error + Send + Sync>>;
    fn dim(&self) -> usize;
}

/// Propose new initial points for a sampler
///
/// This trait can be implemented by users to control how the different
/// chains should be initialized when using [`sample_parallel`].
pub trait InitPointFunc<T> {
    fn new_init_point<R: Rng + ?Sized>(&mut self, rng: &mut R, out: &mut [T]);
}


/// Initialize chains using uniform jitter around zero or some other provided value
#[derive(Default)]
pub struct JitterInitFunc<T> {
    mu: Option<Vec<T>>,
}

impl<T> JitterInitFunc<T>
where T: Float+Debug
{
    /// Initialize new chains with jitter in [-1, 1] around zero
    pub fn new() -> Self {
        JitterInitFunc { mu: None }
    }

    /// Initialize new chains with jitter in [mu - 1, mu + 1].
    pub fn new_with_mean(mu: Vec<T>) -> Self {
        Self { mu: Some(mu) }
    }
}

impl<T> InitPointFunc<T> for JitterInitFunc<T>
where T: Float+Debug
{
    fn new_init_point<R: Rng + ?Sized>(&mut self, rng: &mut R, out: &mut [T]) {
        let mut out1=vec![0.0_f64; out.len()];
        rng.fill(out1.as_mut_slice());
        out1.iter().zip(out.iter_mut()).for_each(|(&a,b)| *b=T::from(a).unwrap());
        if self.mu.is_none() {
            out.iter_mut().for_each(|val| *val = T::from(2.0).unwrap() * *val - T::one());
        } else {
            let mu = self.mu.as_ref().unwrap();
            out.iter_mut()
                .zip(mu.iter().copied())
                .for_each(|(val, mu)| *val = T::from(2.0).unwrap() * *val - T::one() + mu);
        }
    }
}


/// Sample several chains in parallel and return all of the samples live in a channel
#[allow(clippy::type_complexity)]
pub fn sample_parallel<T, F, I>(
    logp_func_maker: F,
    init_point_func: &mut I,
    settings: SamplerArgs<T>,
    n_chains: usize,
    n_draws: usize,
    seed: u64,
    n_try_init: usize,
) -> Result<
    (
        JoinHandle<Vec<ParallelChainResult>>,
        crossbeam::channel::Receiver<(Vec<T>, Box<dyn SampleStats<T>>)>,
    ),
    ParallelSamplingError,
>
where
    T: Float + Debug+Send+Sync+Limits+'static,
    F: CpuLogpFuncMaker<T> + 'static,
    I: InitPointFunc<T>,
{
    let ndim = logp_func_maker.dim();
    let mut func = logp_func_maker.make_logp_func()?;
    assert!(ndim == func.dim());
    let draws = settings.num_tune + n_draws;
    let mut rng = StdRng::seed_from_u64(seed.wrapping_sub(1));
    let mut points: Vec<Result<(Vec<T>, Vec<T>), <F::Func as CpuLogpFunc<T>>::Err>> = (0
        ..n_chains)
        .map(|_| {
            let mut position = vec![T::zero(); ndim];
            let mut grad = vec![T::zero(); ndim];
            init_point_func.new_init_point(&mut rng, &mut position);

            let mut error = None;
            for _ in 0..n_try_init {
                match func.logp(&position, &mut grad) {
                    Err(e) => error = Some(e),
                    Ok(_) => {
                        error = None;
                        break;
                    }
                }
            }
            match error {
                Some(e) => Err(e),
                None => Ok((position, grad)),
            }
        })
        .collect();

    let points: Result<Vec<(Vec<T>, Vec<T>)>, _> = points.drain(..).collect();
    let points = points.map_err(|e| NutsError::LogpFailure(Box::new(e)))?;

    let (sender, receiver) = crossbeam::channel::bounded(128);

    let handle = std::thread::spawn(move || {
        let results: Vec<Result<(), ParallelSamplingError>> = points
            .into_par_iter()
            .with_max_len(1)
            .enumerate()
            .map_with(sender, |sender, (chain, point)| {
                let func = logp_func_maker.make_logp_func()?;
                let mut sampler = new_sampler(
                    func,
                    settings
                );
                let mut rng=SmallRng::seed_from_u64(seed.wrapping_add(chain as u64));
                sampler.set_position(&point.0)?;
                for _ in 0..draws {
                    let (point2, info) = sampler.draw(&mut rng)?;
                    sender
                        .send((point2, Box::new(info) as Box<dyn SampleStats<T>>))
                        .map_err(|_| ParallelSamplingError::ChannelClosed())?;
                }
                Ok(())
            })
            .collect();
        results
    });

    Ok((handle, receiver))
}
