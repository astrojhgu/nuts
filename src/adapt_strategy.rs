use std::{fmt::Debug, iter, marker::PhantomData};

use autodiff::FT;
use num::traits::Float;

use itertools::izip;

use crate::{
    cpu_potential::{CpuLogpFunc, EuclideanPotential},
    mass_matrix::{
        DiagAdaptExpSettings, DiagMassMatrix, DrawGradCollector, ExpWeightedVariance, MassMatrix,
    },
    nuts::{
        AdaptStrategy, AsSampleStatVec, Collector, Hamiltonian, NutsOptions, SampleStatItem,
        SampleStatValue,
    },
    stepsize::{AcceptanceRateCollector, DualAverage, DualAverageOptions},
};

pub trait Limits:Float
{
    fn lower_limit() -> Self;
    fn upper_limit() -> Self;
    fn clamp(&self) -> Self {
        self.max(Self::lower_limit()).min(Self::upper_limit())
    }
}

impl Limits for f64 {
    fn lower_limit() -> f64 {
        1e-10
    }

    fn upper_limit() -> f64 {
        1e10
    }
}

impl<T> Limits for FT<T>
where
    T: Limits + Float + Debug,
{
    fn lower_limit() -> FT<T> {
        T::lower_limit().into()
    }

    fn upper_limit() -> FT<T> {
        T::upper_limit().into()
    }
}

pub struct DualAverageStrategy<T, F, M>
where
    T: Clone + Copy + Float,
{
    step_size_adapt: DualAverage<T>,
    options: DualAverageSettings<T>,
    num_tune: usize,
    num_early: usize,
    _phantom1: PhantomData<F>,
    _phantom2: PhantomData<M>,
}

#[derive(Debug, Clone, Copy)]
pub struct DualAverageStats<T>
where
    T: Clone + Copy,
{
    step_size_bar: T,
    mean_tree_accept: T,
    n_steps: usize,
}

impl<T> AsSampleStatVec<T> for DualAverageStats<T>
where
    T: Float + Copy + Clone + Debug,
{
    fn add_to_vec(&self, vec: &mut Vec<SampleStatItem<T>>) {
        vec.push(("step_size_bar", SampleStatValue::FT(self.step_size_bar)));
        vec.push((
            "mean_tree_accept",
            SampleStatValue::FT(self.mean_tree_accept),
        ));
        vec.push(("n_steps", SampleStatValue::USize(self.n_steps)));
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DualAverageSettings<T>
where
    T: Clone + Copy,
{
    pub early_target_accept: T,
    pub target_accept: T,
    pub final_window_ratio: T,
    pub params: DualAverageOptions<T>,
}

impl<T> Default for DualAverageSettings<T>
where
    T: Clone + Copy + Float,
{
    fn default() -> Self {
        Self {
            early_target_accept: T::from(0.5).unwrap(),
            target_accept: T::from(0.8).unwrap(),
            final_window_ratio: T::from(0.4).unwrap(),
            params: DualAverageOptions::<T>::default(),
        }
    }
}

impl<T, F, M> AdaptStrategy<T> for DualAverageStrategy<T, F, M>
where
    T: Float + Clone + Copy + Send + Debug + 'static,
    F: CpuLogpFunc<T>,
    M: MassMatrix<T>,
{
    type Potential = EuclideanPotential<T, F, M>;
    type Collector = AcceptanceRateCollector<T, crate::cpu_state::State<T>>;
    type Stats = DualAverageStats<T>;
    type Options = DualAverageSettings<T>;

    fn new(options: Self::Options, num_tune: usize, _dim: usize) -> Self {
        Self {
            num_tune,
            num_early: (T::from(num_tune).unwrap() * options.final_window_ratio)
                .ceil()
                .to_usize()
                .unwrap(),
            options,
            step_size_adapt: DualAverage::new(options.params),
            _phantom1: PhantomData::default(),
            _phantom2: PhantomData::default(),
        }
    }

    fn init(
        &mut self,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        _state: &<Self::Potential as Hamiltonian<T>>::State,
    ) {
        potential.step_size = self.options.params.initial_step;
    }

    fn adapt(
        &mut self,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: usize,
        collector: &Self::Collector,
    ) {
        let target = if draw >= self.num_early {
            self.options.target_accept
        } else {
            let start = self.options.early_target_accept;
            let end = self.options.target_accept;
            let time = T::from(draw).unwrap() / T::from(self.num_early).unwrap();
            start
                + (end - start)
                    * (T::one() + (T::from(6).unwrap() * (time - T::from(0.6).unwrap())).tanh())
                    / T::from(2).unwrap()
        };
        if draw < self.num_tune {
            self.step_size_adapt
                .advance(collector.mean.current(), target);
            potential.step_size = self.step_size_adapt.current_step_size()
        } else {
            potential.step_size = self.step_size_adapt.current_step_size_adapted()
        }
    }

    fn new_collector(&self) -> Self::Collector {
        AcceptanceRateCollector::new()
    }

    fn current_stats(
        &self,
        _options: &NutsOptions,
        _potential: &Self::Potential,
        collector: &Self::Collector,
    ) -> Self::Stats {
        DualAverageStats {
            step_size_bar: self.step_size_adapt.current_step_size_adapted(),
            mean_tree_accept: collector.mean.current(),
            n_steps: collector.mean.count(),
        }
    }
}

pub struct ExpWindowDiagAdapt<T, F> {
    dim: usize,
    num_tune: usize,
    exp_variance_draw: ExpWeightedVariance<T>,
    exp_variance_grad: ExpWeightedVariance<T>,
    exp_variance_draw_bg: ExpWeightedVariance<T>,
    exp_variance_grad_bg: ExpWeightedVariance<T>,
    settings: DiagAdaptExpSettings<T>,
    _phantom: PhantomData<F>,
}

#[derive(Clone, Debug)]
pub struct ExpWindowDiagAdaptStats<T> {
    mass_matrix_inv: Option<Vec<T>>,
}

impl<T> AsSampleStatVec<T> for ExpWindowDiagAdaptStats<T>
where
    T: Debug + Clone,
{
    fn add_to_vec(&self, vec: &mut Vec<SampleStatItem<T>>) {
        vec.push((
            "mass_matrix_inv",
            SampleStatValue::OptionArray(self.mass_matrix_inv.clone()),
        ));
    }
}

impl<T: Float + Limits + Send + Clone + Debug + Float + 'static, F: CpuLogpFunc<T>>
    AdaptStrategy<T> for ExpWindowDiagAdapt<T, F>
//where
//StandardNormal: Distribution<T>,
{
    type Potential = EuclideanPotential<T, F, DiagMassMatrix<T>>;
    type Collector = DrawGradCollector<T>;
    type Stats = ExpWindowDiagAdaptStats<T>;
    type Options = DiagAdaptExpSettings<T>;

    fn new(options: Self::Options, num_tune: usize, dim: usize) -> Self {
        Self {
            dim,
            num_tune: num_tune.saturating_sub(options.final_window),
            exp_variance_draw: ExpWeightedVariance::new(dim, options.early_variance_decay, true),
            exp_variance_grad: ExpWeightedVariance::new(dim, options.early_variance_decay, true),
            exp_variance_draw_bg: ExpWeightedVariance::new(dim, options.early_variance_decay, true),
            exp_variance_grad_bg: ExpWeightedVariance::new(dim, options.early_variance_decay, true),
            settings: options,
            _phantom: PhantomData::default(),
        }
    }

    fn init(
        &mut self,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &<Self::Potential as Hamiltonian<T>>::State,
    ) {
        self.exp_variance_draw.set_variance(iter::repeat(T::one()));
        self.exp_variance_draw.set_mean(state.q.iter().copied());
        self.exp_variance_grad
            .set_variance(state.grad.iter().map(|&val| {
                let diag = if !self.settings.grad_init {
                    T::one()
                } else {
                    assert!(val != T::zero(), "Gradient at initial position is zero");
                    val * val
                };
                assert!(diag.is_finite());
                diag
            }));
        self.exp_variance_grad.set_mean(iter::repeat(T::zero()));

        potential.mass_matrix.update_diag(
            izip!(
                self.exp_variance_draw.current(),
                self.exp_variance_grad.current(),
            )
            .map(|(&draw, &grad)| {
                let val = <T as Limits>::clamp(&(draw / grad).sqrt());
                assert!(val.is_finite());
                val
            }),
        );
    }

    fn adapt(
        &mut self,
        _options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: usize,
        collector: &Self::Collector,
    ) {
        if draw >= self.num_tune {
            return;
        }

        if (draw % self.settings.window_switch_freq == 0) & (self.exp_variance_draw_bg.count() > 5)
        {
            self.exp_variance_draw = std::mem::replace(
                &mut self.exp_variance_draw_bg,
                ExpWeightedVariance::new(self.dim, self.settings.variance_decay, true),
            );
            self.exp_variance_grad = std::mem::replace(
                &mut self.exp_variance_grad_bg,
                ExpWeightedVariance::new(self.dim, self.settings.variance_decay, true),
            );

            self.exp_variance_draw_bg
                .set_mean(collector.draw.iter().copied());
            self.exp_variance_grad_bg
                .set_mean(collector.grad.iter().copied());
        } else if collector.is_good {
            self.exp_variance_draw
                .add_sample(collector.draw.iter().copied());
            self.exp_variance_grad
                .add_sample(collector.grad.iter().copied());
            self.exp_variance_draw_bg
                .add_sample(collector.draw.iter().copied());
            self.exp_variance_grad_bg
                .add_sample(collector.grad.iter().copied());
        }

        if self.exp_variance_draw.count() > 2 {
            assert!(self.exp_variance_draw.count() == self.exp_variance_grad.count());
            if (self.settings.grad_init) | (draw > self.settings.window_switch_freq) {
                potential.mass_matrix.update_diag(
                    izip!(
                        self.exp_variance_draw.current(),
                        self.exp_variance_grad.current(),
                    )
                    .map(|(&draw, &grad)| {
                        let val = <T as Limits>::clamp(&(draw / grad).sqrt());
                        assert!(val.is_finite());
                        val
                    }),
                );
            }
        }
    }

    fn new_collector(&self) -> Self::Collector {
        DrawGradCollector::new(self.dim)
    }

    fn current_stats(
        &self,
        _options: &NutsOptions,
        potential: &Self::Potential,
        _collector: &Self::Collector,
    ) -> Self::Stats {
        let diag = if self.settings.store_mass_matrix {
            Some(potential.mass_matrix.variance.clone())
        } else {
            None
        };
        ExpWindowDiagAdaptStats {
            mass_matrix_inv: diag,
        }
    }
}

pub struct CombinedStrategy<S1, S2> {
    data1: S1,
    data2: S2,
}

impl<S1, S2> CombinedStrategy<S1, S2> {
    pub fn new(s1: S1, s2: S2) -> Self {
        Self {
            data1: s1,
            data2: s2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CombinedStats<D1: Debug, D2: Debug> {
    stats1: D1,
    stats2: D2,
}

impl<T: Debug + Send + Clone, D1: AsSampleStatVec<T>, D2: AsSampleStatVec<T>> AsSampleStatVec<T>
    for CombinedStats<D1, D2>
{
    fn add_to_vec(&self, vec: &mut Vec<SampleStatItem<T>>) {
        self.stats1.add_to_vec(vec);
        self.stats2.add_to_vec(vec);
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct CombinedOptions<O1: Copy + Send + Default, O2: Copy + Send + Default> {
    options1: O1,
    options2: O2,
}

impl<T, S1, S2> AdaptStrategy<T> for CombinedStrategy<S1, S2>
where
    T: Send + Clone + Debug + Float,
    S1: AdaptStrategy<T>,
    S2: AdaptStrategy<T, Potential = S1::Potential>,
{
    type Potential = S1::Potential;
    type Collector = CombinedCollector<T, S1::Collector, S2::Collector>;
    type Stats = CombinedStats<S1::Stats, S2::Stats>;
    type Options = CombinedOptions<S1::Options, S2::Options>;

    fn new(options: Self::Options, num_tune: usize, dim: usize) -> Self {
        Self {
            data1: S1::new(options.options1, num_tune, dim),
            data2: S2::new(options.options2, num_tune, dim),
        }
    }

    fn init(
        &mut self,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        state: &<Self::Potential as Hamiltonian<T>>::State,
    ) {
        self.data1.init(options, potential, state);
        self.data2.init(options, potential, state);
    }

    fn adapt(
        &mut self,
        options: &mut NutsOptions,
        potential: &mut Self::Potential,
        draw: usize,
        collector: &Self::Collector,
    ) {
        self.data1
            .adapt(options, potential, draw, &collector.collector1);
        self.data2
            .adapt(options, potential, draw, &collector.collector2);
    }

    fn new_collector(&self) -> Self::Collector {
        CombinedCollector {
            collector1: self.data1.new_collector(),
            collector2: self.data2.new_collector(),
            _dummy: std::marker::PhantomData::default(),
        }
    }

    fn current_stats(
        &self,
        options: &NutsOptions,
        potential: &Self::Potential,
        collector: &Self::Collector,
    ) -> Self::Stats {
        CombinedStats {
            stats1: self
                .data1
                .current_stats(options, potential, &collector.collector1),
            stats2: self
                .data2
                .current_stats(options, potential, &collector.collector2),
        }
    }
}

pub struct CombinedCollector<T: Float, C1: Collector<T>, C2: Collector<T>> {
    collector1: C1,
    collector2: C2,
    _dummy: std::marker::PhantomData<T>,
}

impl<T, C1, C2> Collector<T> for CombinedCollector<T, C1, C2>
where
    T: Float,
    C1: Collector<T>,
    C2: Collector<T, State = C1::State>,
{
    type State = C1::State;

    fn register_leapfrog(
        &mut self,
        start: &Self::State,
        end: &Self::State,
        divergence_info: Option<&dyn crate::nuts::DivergenceInfo<T>>,
    ) {
        self.collector1
            .register_leapfrog(start, end, divergence_info);
        self.collector2
            .register_leapfrog(start, end, divergence_info);
    }

    fn register_draw(&mut self, state: &Self::State, info: &crate::nuts::SampleInfo<T>) {
        self.collector1.register_draw(state, info);
        self.collector2.register_draw(state, info);
    }

    fn register_init(&mut self, state: &Self::State, options: &crate::nuts::NutsOptions) {
        self.collector1.register_init(state, options);
        self.collector2.register_init(state, options);
    }
}
