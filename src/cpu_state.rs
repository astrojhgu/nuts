use std::{
    fmt::Debug,
    ops::Deref,
    rc::{Rc}, sync::Arc,
};

pub type StatePool<T> = lockfree_object_pool::LinearObjectPool<T>;
pub type InnerStateReusable<T> = lockfree_object_pool::LinearOwnedReusable<T>;

use num::traits::Float;

use crate::math::{axpy, axpy_out, scalar_prods2, scalar_prods3};

#[derive(Debug, Clone)]
pub struct InnerState<T>
where
    T: Clone,
{
    pub p: Vec<T>,
    pub q: Vec<T>,
    pub v: Vec<T>,
    pub p_sum: Vec<T>,
    pub grad: Vec<T>,
    pub idx_in_trajectory: isize,
    pub kinetic_energy: T,
    pub potential_energy: T,
}

impl<T> InnerState<T>
where T: Clone+Float{
    pub fn new(dim: usize)->Self{
        InnerState { p: vec![T::zero(); dim], q: vec![T::zero(); dim], v: vec![T::zero(); dim], p_sum: vec![T::zero(); dim], grad: vec![T::zero(); dim], idx_in_trajectory: 0, kinetic_energy: T::zero(), potential_energy: T::zero() }
    }
}

pub struct State<T>
where
    T: Clone + Debug,
{
    //inner: std::mem::ManuallyDrop<Rc<InnerStateReusable<T>>>,
    pub inner: Rc<InnerStateReusable<InnerState<T>>>,
}

impl<T> Debug for State<T>
where
    T: Clone + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "state")
    }
}

impl<T> Deref for State<T>
where
    T: Clone + Debug,
{
    type Target = InnerState<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug)]
pub struct StateInUse {}

type Result<T> = std::result::Result<T, StateInUse>;

impl<T> State<T>
where
    T: Clone + Debug,
{
    pub fn try_mut_inner(&mut self) -> Result<&mut InnerState<T>> {
        match Rc::get_mut(&mut self.inner) {
            Some(val) => Ok( val),
            None => Err(StateInUse {}),
        }
    }
    pub fn clone_inner(&self) -> InnerState<T> {
        (*self.inner.as_ref()).clone()
    }
}

impl<T> Clone for State<T>
where
    T: Clone + Debug,
{
    fn clone(&self) -> Self {
        State {
            inner: self.inner.clone(),
        }
    }
}

impl<T> crate::nuts::State<T> for State<T>
where
    T: Clone + Debug + Float,
{
    type Pool = Arc<StatePool<InnerState<T>>>;

    fn is_turning(&self, other: &Self) -> bool {
        let (start, end) = if self.idx_in_trajectory < other.idx_in_trajectory {
            (self, other)
        } else {
            (other, self)
        };

        let a = start.idx_in_trajectory;
        let b = end.idx_in_trajectory;

        assert!(a < b);
        let (turn1, turn2) = if (a >= 0) & (b >= 0) {
            scalar_prods3(&end.p_sum, &start.p_sum, &start.p, &end.v, &start.v)
        } else if (b >= 0) & (a < 0) {
            scalar_prods2(&end.p_sum, &start.p_sum, &end.v, &start.v)
        } else {
            assert!((a < 0) & (b < 0));
            scalar_prods3(&start.p_sum, &end.p_sum, &end.p, &end.v, &start.v)
        };

        (turn1 < T::zero()) | (turn2 < T::zero())
    }

    fn write_position(&self, out: &mut [T]) {
        out.copy_from_slice(&self.q)
    }

    fn write_gradient(&self, out: &mut [T]) {
        out.copy_from_slice(&self.grad);
    }

    fn energy(&self) -> T {
        self.kinetic_energy + self.potential_energy
    }

    fn index_in_trajectory(&self) -> isize {
        self.idx_in_trajectory
    }

    fn make_init_point(&mut self) {
        let inner = self.try_mut_inner().unwrap();
        inner.idx_in_trajectory = 0;
        inner.p_sum.copy_from_slice(&inner.p);
    }

    fn potential_energy(&self) -> T {
        self.potential_energy
    }
}

impl<T> State<T>
where
    T: Clone + Debug + Float,
{
    pub fn first_momentum_halfstep(&self, out: &mut Self, epsilon: T) {
        axpy_out(
            &self.grad,
            &self.p,
            epsilon / T::from(2.0).unwrap(),
            &mut out.try_mut_inner().expect("State already in use").p,
        );
    }

    pub fn position_step(&self, out: &mut Self, epsilon: T) {
        let out = out.try_mut_inner().expect("State already in use");
        axpy_out(&out.v, &self.q, epsilon, &mut out.q);
    }

    pub fn second_momentum_halfstep(&mut self, epsilon: T) {
        let inner = self.try_mut_inner().expect("State already in use");
        axpy(&inner.grad, &mut inner.p, epsilon / T::from(2).unwrap());
    }

    pub fn set_psum(&self, target: &mut Self, _dir: crate::nuts::Direction) {
        let out = target.try_mut_inner().expect("State already in use");

        assert!(out.idx_in_trajectory != 0);

        if out.idx_in_trajectory == -1 {
            out.p_sum.copy_from_slice(&out.p);
        } else {
            axpy_out(&out.p, &self.p_sum, T::one(), &mut out.p_sum);
        }
    }

    pub fn index_in_trajectory(&self) -> isize {
        self.idx_in_trajectory
    }

    pub fn index_in_trajectory_mut(&mut self) -> &mut isize {
        &mut self
            .try_mut_inner()
            .expect("State already in use")
            .idx_in_trajectory
    }
}

