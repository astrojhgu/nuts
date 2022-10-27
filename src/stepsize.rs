use std::marker::PhantomData;

use num::Float;

use crate::nuts::{Collector, NutsOptions, State};

/// Settings for step size adaptation
#[derive(Debug, Clone, Copy)]
pub struct DualAverageOptions<T> {
    pub k: T,
    pub t0: T,
    pub gamma: T,
    pub initial_step: T,
}

impl<T> Default for DualAverageOptions<T>
where
    T: Float,
{
    fn default() -> DualAverageOptions<T> {
        DualAverageOptions {
            k: T::from(0.75).unwrap(),
            t0: T::from(10.).unwrap(),
            gamma: T::from(0.05).unwrap(),
            initial_step: T::from(0.1).unwrap(),
        }
    }
}

#[derive(Clone)]
pub struct DualAverage<T> {
    log_step: T,
    log_step_adapted: T,
    hbar: T,
    mu: T,
    count: usize,
    settings: DualAverageOptions<T>,
}

impl<T> DualAverage<T>
where
    T: Float,
{
    pub fn new(settings: DualAverageOptions<T>) -> DualAverage<T> {
        let initial_step = settings.initial_step;
        DualAverage {
            log_step: initial_step.ln(),
            log_step_adapted: initial_step.ln(),
            hbar: T::zero(),
            //mu: (10. * initial_step).ln(),
            mu: (T::from(2).unwrap() * initial_step).ln(),
            count: 1,
            settings,
        }
    }

    pub fn advance(&mut self, accept_stat: T, target: T) {
        let w = T::one() / (T::from(self.count).unwrap() + self.settings.t0);
        self.hbar = (T::one() - w) * self.hbar + w * (target - accept_stat);
        self.log_step =
            self.mu - self.hbar * T::from(self.count).unwrap().sqrt() / self.settings.gamma;
        let mk = T::from(self.count).unwrap().powf(-self.settings.k);
        self.log_step_adapted = mk * self.log_step + (T::one() - mk) * self.log_step_adapted;
        self.count += 1;
    }

    pub fn current_step_size(&self) -> T {
        self.log_step.exp()
    }

    pub fn current_step_size_adapted(&self) -> T {
        self.log_step_adapted.exp()
    }

    #[allow(dead_code)]
    pub fn reset(&mut self, initial_step: T) {
        self.log_step = initial_step.ln();
        self.log_step_adapted = initial_step.ln();
        self.hbar = T::zero();
        self.mu = (T::from(10).unwrap() * initial_step).ln();
        self.count = 1;
    }
}

#[derive(Default)]
pub struct RunningMean<T> {
    sum: T,
    count: usize,
}

impl<T> RunningMean<T>
where
    T: Float,
{
    fn new() -> RunningMean<T> {
        RunningMean {
            sum: T::zero(),
            count: 0,
        }
    }

    fn add(&mut self, value: T) {
        self.sum = self.sum + value;
        self.count += 1;
    }

    pub fn current(&self) -> T {
        self.sum / T::from(self.count).unwrap()
    }

    pub fn reset(&mut self) {
        self.sum = T::zero();
        self.count = 0;
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

#[derive(Default)]
pub struct AcceptanceRateCollector<T: Float, S: State<T>> {
    initial_energy: T,
    pub mean: RunningMean<T>,
    phantom: PhantomData<S>,
}

impl<T: Float, S: State<T>> AcceptanceRateCollector<T, S> {
    pub fn new() -> AcceptanceRateCollector<T, S> {
        AcceptanceRateCollector {
            initial_energy: T::zero(),
            mean: RunningMean::new(),
            phantom: PhantomData::default(),
        }
    }
}

impl<T: Float, S: State<T>> Collector<T> for AcceptanceRateCollector<T, S> {
    type State = S;

    fn register_leapfrog(
        &mut self,
        _start: &Self::State,
        end: &Self::State,
        divergence_info: Option<&dyn crate::nuts::DivergenceInfo<T>>,
    ) {
        match divergence_info {
            Some(_) => self.mean.add(T::zero()),
            None => self
                .mean
                .add(end.log_acceptance_probability(self.initial_energy).exp()),
        }
    }

    fn register_init(&mut self, state: &Self::State, _options: &NutsOptions) {
        self.initial_energy = state.energy();
        self.mean.reset();
    }
}
