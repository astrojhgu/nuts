use rand::Rng;
use thiserror::Error;

use std::{fmt::Debug, marker::PhantomData};

use crate::math::logaddexp;
use num::traits::Float;

#[derive(Error, Debug)]
pub enum NutsError {
    #[error("Logp function returned error: {0}")]
    LogpFailure(Box<dyn std::error::Error + Send>),
}

pub type Result<T> = std::result::Result<T, NutsError>;

// TODO This should be an Box<enum> instead of a trait object?
/// Details about a divergence that might have occured during sampling
///
/// There are two reasons why we might observe a divergence:
/// - The integration error of the Hamiltonian is larger than
///   a cutoff value or nan.
/// - The logp function caused a recoverable error (eg if an ODE solver
///   failed)
pub trait DivergenceInfo<T>: AsSampleStatVec<T> + std::fmt::Debug + Send
where
    T: Send + Debug + Clone,
{
    /// The position in parameter space where the diverging leapfrog started
    fn start_location(&self) -> Option<&[T]>;

    /// The position in parameter space where the diverging leapfrog ended
    fn end_location(&self) -> Option<&[T]>;

    /// The difference between the energy at the initial location of the trajectory and
    /// the energy at the end of the diverging leapfrog step.
    ///
    /// This is not available if the divergence was caused by a logp function error
    fn energy_error(&self) -> Option<T>;

    /// The index of the end location of the diverging leapfrog.
    fn end_idx_in_trajectory(&self) -> Option<isize>;

    /// The index of the start location of the diverging leapfrog.
    fn start_idx_in_trajectory(&self) -> Option<isize>;

    /// Return the logp function error that caused the divergence if there was any
    ///
    /// This is not available if the divergence was cause because of a large energy
    /// difference.
    fn logp_function_error(&self) -> Option<&dyn std::error::Error>;
}

#[derive(Debug, Copy, Clone)]
pub enum Direction {
    Forward,
    Backward,
}

impl rand::distributions::Distribution<Direction> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Direction {
        if rng.gen::<bool>() {
            Direction::Forward
        } else {
            Direction::Backward
        }
    }
}

/// Callbacks for various events during a Nuts sampling step.
///
/// Collectors can compute statistics like the mean acceptance rate
/// or collect data for mass matrix adaptation.
pub trait Collector<T>
where
    T: Float,
{
    type State: State<T>;

    fn register_leapfrog(
        &mut self,
        _start: &Self::State,
        _end: &Self::State,
        _divergence_info: Option<&dyn DivergenceInfo<T>>,
    ) {
    }
    fn register_draw(&mut self, _state: &Self::State, _info: &SampleInfo<T>) {}
    fn register_init(&mut self, _state: &Self::State, _options: &NutsOptions) {}
}

/// Errors that happen when we evaluate the logp and gradient function
pub trait LogpError: std::error::Error {
    /// Unrecoverable errors during logp computation stop sampling,
    /// recoverable errors are seen as divergences.
    fn is_recoverable(&self) -> bool;
}

/// The hamiltonian defined by the potential energy and the kinetic energy
pub trait Hamiltonian<T>
where
    T: Send + Debug + Clone + Float,
{
    /// The type that stores a point in phase space
    type State: State<T>;
    /// More detailed information about divergences
    type DivergenceInfo: DivergenceInfo<T> + 'static;
    /// Errors that happen during logp evaluation
    type LogpError: LogpError + Send;
    /// Statistics that should be exported to the trace as part of the sampler stats
    type Stats: Send + AsSampleStatVec<T>;

    /// Perform one leapfrog step.
    ///
    /// Return either an unrecoverable error, a new state or a divergence.
    fn leapfrog<C: Collector<T, State = Self::State>>(
        &mut self,
        pool: &mut <Self::State as State<T>>::Pool,
        start: &Self::State,
        dir: Direction,
        initial_energy: T,
        collector: &mut C,
    ) -> Result<std::result::Result<Self::State, Self::DivergenceInfo>>;

    /// Initialize a state at a new location.
    ///
    /// The momentum should be initialized to some arbitrary invalid number,
    /// it will later be set using Self::randomize_momentum.
    fn init_state(
        &mut self,
        pool: &mut <Self::State as State<T>>::Pool,
        init: &[T],
    ) -> Result<Self::State>;

    /// Randomize the momentum part of a state
    fn randomize_momentum<R: rand::Rng + ?Sized>(&self, state: &mut Self::State, rng: &mut R);

    /// Return sampler statistics defined in Self::Stats
    fn current_stats(&self) -> Self::Stats;

    fn new_empty_state(&mut self, pool: &mut <Self::State as State<T>>::Pool) -> Self::State;

    /// Crate a new state pool that can be used to crate new states.
    fn new_pool(&mut self, capacity: usize) -> <Self::State as State<T>>::Pool;

    /// The dimension of the hamiltonian (position only).
    fn dim(&self) -> usize;
}

/// A point in phase space
///
/// This also needs to store the sum of momentum terms
/// from the initial point of the trajectory to this point,
/// so that it can compute the termination criterion in
/// `is_turming`.
pub trait State<T>: Clone + Debug
where
    T: Float,
{
    /// The state pool can be used to crate new states
    type Pool;

    /// Write the position stored in the state to a different location
    fn write_position(&self, out: &mut [T]);

    /// Write the gradient stored in the state to a different location
    fn write_gradient(&self, out: &mut [T]);

    /// Compute the termination criterion for NUTS
    fn is_turning(&self, other: &Self) -> bool;

    /// The total energy (potential + kinetic)
    fn energy(&self) -> T;
    fn potential_energy(&self) -> T;
    fn index_in_trajectory(&self) -> isize;

    /// Initialize the point to be the first in the trajectory.
    ///
    /// Set index_in_trajectory to 0 and reinitialize the sum of
    /// the momentum terms.
    fn make_init_point(&mut self);

    fn log_acceptance_probability(&self, initial_energy: T) -> T {
        (initial_energy - self.energy()).min(T::zero())
    }
}

/// Information about a draw, exported as part of the sampler stats
#[derive(Debug)]
pub struct SampleInfo<T> {
    /// The depth of the trajectory that this point was sampled from
    pub depth: usize,

    /// More detailed information about a divergence that might have
    /// occured in the trajectory.
    pub divergence_info: Option<Box<dyn DivergenceInfo<T>>>,

    /// Whether the trajectory was terminated because it reached
    /// the maximum tree depth.
    pub reached_maxdepth: bool,
}

/// A part of the trajectory tree during NUTS sampling.
struct NutsTree<
    T: Send + Clone + Debug + Float,
    P: Hamiltonian<T>,
    C: Collector<T, State = P::State>,
> {
    /// The left position of the tree.
    ///
    /// The left side always has the smaller index_in_trajectory.
    /// Leapfrogs in backward direction will replace the left.
    left: P::State,
    right: P::State,

    /// A draw from the trajectory between left and right using
    /// multinomial sampling.
    draw: P::State,
    log_size: T,
    depth: usize,
    initial_energy: T,

    /// A tree is the main tree if it contains the initial point
    /// of the trajectory.
    is_main: bool,
    collector: PhantomData<C>,
}

enum ExtendResult<
    T: Send + Debug + Clone + Float,
    P: Hamiltonian<T>,
    C: Collector<T, State = P::State>,
> {
    /// The tree extension succeeded properly, and the termination
    /// criterion was not reached.
    Ok(NutsTree<T, P, C>),
    /// An unrecoverable error happend during a leapfrog step
    Err(NutsError),
    /// Tree extension succeeded and the termination criterion
    /// was reached.
    Turning(NutsTree<T, P, C>),
    /// A divergence happend during tree extension.
    Diverging(NutsTree<T, P, C>, P::DivergenceInfo),
}

impl<T: Send + Debug + Clone + Float, P: Hamiltonian<T>, C: Collector<T, State = P::State>>
    NutsTree<T, P, C>
{
    fn new(state: P::State) -> NutsTree<T, P, C> {
        let initial_energy = state.energy();
        NutsTree {
            right: state.clone(),
            left: state.clone(),
            draw: state,
            depth: 0,
            log_size: T::zero(),
            initial_energy,
            is_main: true,
            collector: PhantomData,
        }
    }

    #[inline]
    fn extend<R>(
        mut self,
        pool: &mut <<P as Hamiltonian<T>>::State as State<T>>::Pool,
        rng: &mut R,
        potential: &mut P,
        direction: Direction,
        options: &NutsOptions,
        collector: &mut C,
    ) -> ExtendResult<T, P, C>
    where
        P: Hamiltonian<T>,
        R: rand::Rng + ?Sized,
    {
        let mut other = match self.single_step(pool, potential, direction, collector) {
            Ok(Ok(tree)) => tree,
            Ok(Err(info)) => return ExtendResult::Diverging(self, info),
            Err(err) => return ExtendResult::Err(err),
        };

        while other.depth < self.depth {
            use ExtendResult::*;
            other = match other.extend(pool, rng, potential, direction, options, collector) {
                Ok(tree) => tree,
                Turning(_) => {
                    return Turning(self);
                }
                Diverging(_, info) => {
                    return Diverging(self, info);
                }
                Err(error) => {
                    return Err(error);
                }
            };
        }

        let (first, last) = match direction {
            Direction::Forward => (&self.left, &other.right),
            Direction::Backward => (&other.left, &self.right),
        };

        let mut turning = first.is_turning(last);
        if self.depth > 0 {
            if !turning {
                turning = self.right.is_turning(&other.right);
            }
            if !turning {
                turning = self.left.is_turning(&other.left);
            }
        }

        self.merge_into(other, rng, direction);

        if turning {
            ExtendResult::Turning(self)
        } else {
            ExtendResult::Ok(self)
        }
    }

    fn merge_into<R: rand::Rng + ?Sized>(
        &mut self,
        other: NutsTree<T, P, C>,
        rng: &mut R,
        direction: Direction,
    ) {
        assert!(self.depth == other.depth);
        assert!(self.left.index_in_trajectory() <= self.right.index_in_trajectory());
        match direction {
            Direction::Forward => {
                self.right = other.right;
            }
            Direction::Backward => {
                self.left = other.left;
            }
        }
        let log_size = logaddexp(self.log_size, other.log_size);

        let self_log_size = if self.is_main {
            assert!(self.left.index_in_trajectory() <= 0);
            assert!(self.right.index_in_trajectory() >= 0);
            self.log_size
        } else {
            log_size
        };

        if other.log_size >= self_log_size
            || rng.gen_bool((other.log_size - self_log_size).exp().to_f64().unwrap())
        {
            self.draw = other.draw;
        }
        self.depth += 1;
        self.log_size = log_size;
    }

    fn single_step(
        &self,
        pool: &mut <<P as Hamiltonian<T>>::State as State<T>>::Pool,
        potential: &mut P,
        direction: Direction,
        collector: &mut C,
    ) -> Result<std::result::Result<NutsTree<T, P, C>, P::DivergenceInfo>> {
        let start = match direction {
            Direction::Forward => &self.right,
            Direction::Backward => &self.left,
        };
        let end = match potential.leapfrog(pool, start, direction, self.initial_energy, collector) {
            Ok(Ok(end)) => end,
            Ok(Err(info)) => return Ok(Err(info)),
            Err(error) => return Err(error),
        };

        let log_size = self.initial_energy - end.energy();
        Ok(Ok(NutsTree {
            right: end.clone(),
            left: end.clone(),
            draw: end,
            depth: 0,
            log_size,
            initial_energy: self.initial_energy,
            is_main: false,
            collector: PhantomData,
        }))
    }

    fn info(&self, maxdepth: bool, divergence_info: Option<P::DivergenceInfo>) -> SampleInfo<T> {
        let info: Option<Box<dyn DivergenceInfo<T>>> = match divergence_info {
            Some(info) => Some(Box::new(info)),
            None => None,
        };
        SampleInfo {
            depth: self.depth,
            divergence_info: info,
            reached_maxdepth: maxdepth,
        }
    }
}

pub struct NutsOptions {
    pub maxdepth: usize,
    pub store_gradient: bool,
}

pub fn draw<T, P, R, C>(
    pool: &mut <P::State as State<T>>::Pool,
    init: &mut P::State,
    rng: &mut R,
    potential: &mut P,
    options: &NutsOptions,
    collector: &mut C,
) -> Result<(P::State, SampleInfo<T>)>
where
    T: Debug + Send + Clone + Float,
    P: Hamiltonian<T>,
    R: rand::Rng + ?Sized,
    C: Collector<T, State = P::State>,
{
    potential.randomize_momentum(init, rng);
    init.make_init_point();
    collector.register_init(init, options);

    let mut tree = NutsTree::new(init.clone());
    while tree.depth < options.maxdepth {
        let direction: Direction = rng.gen();
        tree = match tree.extend(pool, rng, potential, direction, options, collector) {
            ExtendResult::Ok(tree) => tree,
            ExtendResult::Turning(tree) => {
                let info = tree.info(false, None);
                collector.register_draw(&tree.draw, &info);
                return Ok((tree.draw, info));
            }
            ExtendResult::Diverging(tree, info) => {
                let info = tree.info(false, Some(info));
                collector.register_draw(&tree.draw, &info);
                return Ok((tree.draw, info));
            }
            ExtendResult::Err(error) => {
                return Err(error);
            }
        };
    }
    let info = tree.info(true, None);
    Ok((tree.draw, info))
}

#[derive(Debug)]
pub struct NutsSampleStats<T: Send + Debug, HStats: Send + Debug, AdaptStats: Send + Debug> {
    pub depth: usize,
    pub maxdepth_reached: bool,
    pub idx_in_trajectory: isize,
    pub logp: T,
    pub energy: T,
    pub divergence_info: Option<Box<dyn DivergenceInfo<T>>>,
    pub draw: usize,
    pub gradient: Option<Vec<T>>,
    pub potential_stats: HStats,
    pub strategy_stats: AdaptStats,
}

#[derive(Debug, Clone)]
pub enum SampleStatValue<T>
where
    T: Debug + Clone,
{
    Array(Vec<T>),
    OptionArray(Option<Vec<T>>),
    USize(usize),
    ISize(isize),
    OptionISize(Option<isize>),
    FT(T),
    OptionFT(Option<T>),
    Bool(bool),
    String(String),
}

impl<T> From<Vec<T>> for SampleStatValue<T>
where
    T: Debug + Clone,
{
    fn from(val: Vec<T>) -> Self {
        SampleStatValue::Array(val)
    }
}

impl<T> From<Option<Vec<T>>> for SampleStatValue<T>
where
    T: Debug + Clone,
{
    fn from(val: Option<Vec<T>>) -> Self {
        SampleStatValue::OptionArray(val)
    }
}

impl<T> From<usize> for SampleStatValue<T>
where
    T: Debug + Clone,
{
    fn from(val: usize) -> Self {
        SampleStatValue::USize(val)
    }
}

impl<T> From<isize> for SampleStatValue<T>
where
    T: Debug + Clone,
{
    fn from(val: isize) -> Self {
        SampleStatValue::ISize(val)
    }
}

impl<T> From<Option<isize>> for SampleStatValue<T>
where
    T: Debug + Clone,
{
    fn from(val: Option<isize>) -> Self {
        SampleStatValue::OptionISize(val)
    }
}

impl<T> From<bool> for SampleStatValue<T>
where
    T: Debug + Clone,
{
    fn from(val: bool) -> Self {
        SampleStatValue::Bool(val)
    }
}

impl<T> From<String> for SampleStatValue<T>
where
    T: Debug + Clone,
{
    fn from(val: String) -> Self {
        SampleStatValue::String(val)
    }
}

pub trait AsSampleStatVec<T>: Debug
where
    T: Debug + Clone,
{
    fn add_to_vec(&self, vec: &mut Vec<SampleStatItem<T>>);
}

pub type SampleStatItem<T> = (&'static str, SampleStatValue<T>);

/// Diagnostic information about draws and the state of the sampler for each draw
pub trait SampleStats<T>: Send + Debug
where
    T: Send + Debug + Clone,
{
    /// The depth of the NUTS tree that the draw was sampled from
    fn depth(&self) -> usize;
    /// Whether the trajectory was stopped because the maximum size
    /// was reached.
    fn maxdepth_reached(&self) -> bool;
    /// The index of the accepted sample in the trajectory
    fn index_in_trajectory(&self) -> isize;
    /// The unnormalized posterior density at the draw
    fn logp(&self) -> T;
    /// The value of the hamiltonian of the draw
    fn energy(&self) -> T;
    /// More detailed information if the draw came from a diverging trajectory.
    fn divergence_info(&self) -> Option<&dyn DivergenceInfo<T>>;
    /// The draw number
    fn draw(&self) -> usize;
    /// The logp gradient at the location of the draw. This is only stored
    /// if NutsOptions.store_gradient is `true`.
    fn gradient(&self) -> Option<&[T]>;
    /// Export the sample statisitcs to a vector. This might include some additional
    /// diagnostics coming from the step size and matrix adaptation strategies.
    fn to_vec(&self) -> Vec<SampleStatItem<T>>;
}

impl<T, H, A> SampleStats<T> for NutsSampleStats<T, H, A>
where
    T: Send + Debug + Clone,
    H: Send + Debug + AsSampleStatVec<T>,
    A: Send + Debug + AsSampleStatVec<T>,
{
    fn depth(&self) -> usize {
        self.depth
    }
    fn maxdepth_reached(&self) -> bool {
        self.maxdepth_reached
    }
    fn index_in_trajectory(&self) -> isize {
        self.idx_in_trajectory
    }
    fn logp(&self) -> T {
        self.logp.clone()
    }
    fn energy(&self) -> T {
        self.energy.clone()
    }
    fn divergence_info(&self) -> Option<&dyn DivergenceInfo<T>> {
        self.divergence_info.as_ref().map(|x| x.as_ref())
    }
    fn draw(&self) -> usize {
        self.draw
    }
    fn gradient(&self) -> Option<&[T]> {
        self.gradient.as_ref().map(|x| &x[..])
    }
    fn to_vec(&self) -> Vec<SampleStatItem<T>> {
        let mut vec = Vec::with_capacity(20);
        vec.push(("depth", self.depth.into()));
        vec.push(("maxdepth_reached", self.maxdepth_reached.into()));
        vec.push(("index_in_trajectory", self.idx_in_trajectory.into()));
        vec.push(("logp", SampleStatValue::FT(self.logp.clone())));
        vec.push(("energy", SampleStatValue::FT(self.energy.clone())));
        vec.push(("diverging", self.divergence_info.is_some().into()));
        self.potential_stats.add_to_vec(&mut vec);
        self.strategy_stats.add_to_vec(&mut vec);
        if let Some(info) = self.divergence_info() {
            info.add_to_vec(&mut vec);
        }
        if let Some(grad) = self.gradient() {
            vec.push(("gradient", grad.to_vec().into()));
        } else {
            vec.push(("gradient", SampleStatValue::OptionArray(None)));
        }
        vec
    }
}

/// Draw samples from the posterior distribution using Hamiltonian MCMC.
pub trait Chain<T>
where
    T: Send + Debug + Clone + Float,
{
    type Hamiltonian: Hamiltonian<T>;
    type AdaptStrategy: AdaptStrategy<T>;
    type Stats: SampleStats<T>;

    /// Initialize the sampler to a position. This should be called
    /// before calling draw.
    ///
    /// This fails if the logp function returns an error.
    fn set_position(&mut self, position: &[T]) -> Result<()>;

    /// Draw a new sample and return the position and some diagnosic information.
    fn draw<R>(&mut self, rng: &mut R) -> Result<(Vec<T>, Self::Stats)>
    where R: Rng;

    /// The dimensionality of the posterior.
    fn dim(&self) -> usize;
}

pub struct NutsChain<T, P, S>
where
    T: Debug + Clone + Send + Float,
    P: Hamiltonian<T>,
    S: AdaptStrategy<T, Potential = P>,
{
    pool: <P::State as State<T>>::Pool,
    potential: P,
    collector: S::Collector,
    options: NutsOptions,
    init: P::State,
    draw_count: usize,
    strategy: S,
}

impl<T, P, S> NutsChain<T, P, S>
where
    T: Debug + Clone + Send + Float,
    P: Hamiltonian<T>,
    S: AdaptStrategy<T, Potential = P>,
{
    pub fn new(mut potential: P, strategy: S, options: NutsOptions) -> Self {
        let pool_size: usize = options.maxdepth.checked_mul(2).unwrap().try_into().unwrap();
        let mut pool = potential.new_pool(pool_size);
        let init = potential.new_empty_state(&mut pool);
        let collector = strategy.new_collector();
        NutsChain {
            pool,
            potential,
            collector,
            options,
            init,
            draw_count: 0,
            strategy,
        }
    }
}

pub trait AdaptStrategy<T>
where
    T: Debug + Clone + Send + Float,
{
    type Potential: Hamiltonian<T>;
    type Collector: Collector<T, State = <Self::Potential as Hamiltonian<T>>::State>;
    type Stats: Send + AsSampleStatVec<T>;
    type Options: Copy + Send + Default;

    fn new(options: Self::Options, num_tune: usize, dim: usize) -> Self;

    fn init(
        &mut self,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &<Self::Potential as Hamiltonian<T>>::State,
    );

    fn adapt(
        &mut self,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: usize,
        collector: &Self::Collector,
    );

    fn new_collector(&self) -> Self::Collector;

    fn current_stats(
        &self,
        options: &NutsOptions,
        potential: &Self::Potential,
        collector: &Self::Collector,
    ) -> Self::Stats;
}

impl<T, H, S> Chain<T> for NutsChain<T, H, S>
where
    T: Float + Debug + Send,
    H: Hamiltonian<T>,
    S: AdaptStrategy<T, Potential = H>,
{
    type Hamiltonian = H;
    type AdaptStrategy = S;
    type Stats = NutsSampleStats<T, H::Stats, S::Stats>;

    fn set_position(&mut self, position: &[T]) -> Result<()> {
        let state = self.potential.init_state(&mut self.pool, position)?;
        self.init = state;
        self.strategy
            .init(&mut self.options, &mut self.potential, &self.init);
        Ok(())
    }

    fn draw<R>(&mut self, rng: &mut R) -> Result<(Vec<T>, Self::Stats)>
    where R: Rng
    {
        let (state, info) = draw(
            &mut self.pool,
            &mut self.init,
            rng,
            &mut self.potential,
            &self.options,
            &mut self.collector,
        )?;
        let mut position: Vec<T> = vec![T::zero(); self.potential.dim()];
        state.write_position(&mut position);
        let stats = NutsSampleStats {
            depth: info.depth,
            maxdepth_reached: info.reached_maxdepth,
            idx_in_trajectory: state.index_in_trajectory(),
            logp: -state.potential_energy(),
            energy: state.energy(),
            divergence_info: info.divergence_info,
            draw: self.draw_count,
            potential_stats: self.potential.current_stats(),
            strategy_stats: self.strategy.current_stats(
                &self.options,
                &self.potential,
                &self.collector,
            ),
            gradient: if self.options.store_gradient {
                let mut gradient: Vec<T> = vec![T::zero(); self.potential.dim()];
                state.write_gradient(&mut gradient);
                Some(gradient)
            } else {
                None
            },
        };
        self.strategy.adapt(
            &mut self.options,
            &mut self.potential,
            self.draw_count,
            &self.collector,
        );
        self.init = state;
        self.draw_count += 1;
        Ok((position, stats))
    }

    fn dim(&self) -> usize {
        self.potential.dim()
    }
}
