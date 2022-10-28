use std::{
    cell::RefCell,
    fmt::Debug,
    ops::{Deref},
    rc::{Rc, Weak},
};

use num::traits::Float;

use crate::math::{axpy, axpy_out, scalar_prods2, scalar_prods3};

#[derive(Debug)]
struct StateStorage<T>
where
    T: Clone + Debug,
{
    free_states: RefCell<Vec<Rc<InnerStateReusable<T>>>>,
}

impl<T> StateStorage<T>
where
    T: Clone + Debug,
{
    fn new() -> StateStorage<T> {
        StateStorage {
            free_states: RefCell::new(Vec::with_capacity(20)),
        }
    }
}

impl<T> ReuseState<T> for StateStorage<T>
where
    T: Clone + Debug,
{
    fn reuse_state(&self, state: Rc<InnerStateReusable<T>>) {
        self.free_states.borrow_mut().push(state)
    }
}

pub struct StatePool<T>
where
    T: Clone + Debug,
{
    storage: Rc<StateStorage<T>>,
    dim: usize,
}

impl<T> StatePool<T>
where
    T: Clone + Debug + Float + 'static,
{
    pub fn new(dim: usize) -> StatePool<T> {
        StatePool {
            storage: Rc::new(StateStorage::new()),
            dim,
        }
    }

    pub fn new_state(&mut self) -> State<T> {
        let inner = match self.storage.free_states.borrow_mut().pop() {
            Some(inner) => {
                if self.dim != inner.inner.q.len() {
                    panic!("dim mismatch");
                }
                inner
            }
            None => {
                let owner: Rc<dyn ReuseState<T>> = self.storage.clone();
                Rc::new(InnerStateReusable::<T>::new(self.dim, &owner))
            }
        };
        State {
            inner: std::mem::ManuallyDrop::new(inner),
        }
    }
}

trait ReuseState<T>: Debug
where
    T: Clone + Debug,
{
    fn reuse_state(&self, state: Rc<InnerStateReusable<T>>);
}

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

#[derive(Debug)]
pub struct InnerStateReusable<T>
where
    T: Clone,
{
    inner: InnerState<T>,
    reuser: Weak<dyn ReuseState<T>>,
}


impl<T> InnerStateReusable<T>
where
    T: Clone + Float,
{
    fn new(size: usize, owner: &Rc<dyn ReuseState<T>>) -> InnerStateReusable<T> {
        InnerStateReusable {
            inner: InnerState {
                p: vec![T::zero(); size].into(),
                //p: AlignedArray::new(size),
                q: vec![T::zero(); size].into(),
                //q: AlignedArray::new(size),
                v: vec![T::zero(); size].into(),
                //v: AlignedArray::new(size),
                p_sum: vec![T::zero(); size].into(),
                //p_sum: AlignedArray::new(size),
                grad: vec![T::zero(); size].into(),
                //grad: AlignedArray::new(size),
                idx_in_trajectory: 0,
                kinetic_energy: T::zero(),
                potential_energy: T::zero(),
            },
            reuser: Rc::downgrade(owner),
        }
    }
}

#[derive(Debug)]
pub struct State<T>
where
    T: Clone + Debug,
{
    inner: std::mem::ManuallyDrop<Rc<InnerStateReusable<T>>>,
}

impl<T> Deref for State<T>
where
    T: Clone + Debug,
{
    type Target = InnerState<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner.inner
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
            Some(val) => Ok(&mut val.inner),
            None => Err(StateInUse {}),
        }
    }

    pub fn clone_inner(&self) -> InnerState<T> {
        self.inner.inner.clone()
    }
}

impl<T> Drop for State<T>
where
    T: Clone + Debug,
{
    fn drop(&mut self) {
        let mut rc = unsafe { std::mem::ManuallyDrop::take(&mut self.inner) };
        if let Some(state_ref) = Rc::get_mut(&mut rc) {
            if let Some(reuser) = &mut state_ref.reuser.upgrade() {
                reuser.reuse_state(rc);
            }
        }
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
    type Pool = StatePool<T>;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_pool() {
        let mut pool = StatePool::<f64>::new(10);
        let mut state = pool.new_state();
        assert!(state.p.len() == 10);
        state.try_mut_inner().unwrap();

        let mut state2 = state.clone();
        assert!(state.try_mut_inner().is_err());
        assert!(state2.try_mut_inner().is_err());
    }

    #[test]
    fn make_state() {
        let dim = 10;
        let mut pool = StatePool::<f64>::new(dim);
        let a = pool.new_state();

        assert_eq!(a.idx_in_trajectory, 0);
        assert!(a.p_sum.iter().all(|&x| x == 0.0));
        assert_eq!(a.p_sum.len(), dim);
        assert_eq!(a.grad.len(), dim);
        assert_eq!(a.q.len(), dim);
        assert_eq!(a.p.len(), dim);
    }
}
