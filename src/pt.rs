use num::traits::Float;
use rand::{seq::SliceRandom, Rng};
use rand_distr::Uniform;

use std::fmt::Debug;

use crate::nuts::{AdaptStrategy, Chain, Hamiltonian, NutsChain, State};

fn exchange_prob<T>(lp1: T, lp2: T, beta1: T, beta2: T) -> T
where
    T: Float,
{
    let x = ((beta2 - beta1) * (-lp2 + lp1)).exp();

    if x > T::one() {
        T::one()
    } else {
        x
    }
}

pub fn swap_chains<T, P, S, R>(chains: &mut [NutsChain<T, P, S>], rng: &mut R, beta_list: &[T])
where
    T: Debug + Clone + Send + Float,
    P: Hamiltonian<T>,
    S: AdaptStrategy<T, Potential = P>,
    R: Rng,
{
    //let mut new_ensemble = ensemble_logprob.0.clone();
    //let mut new_logprob = ensemble_logprob.1.clone();
    let nbeta = beta_list.len();
    let nchains_per_beta = chains.len() / nbeta;
    assert!(nbeta * nchains_per_beta == chains.len());
    let mut jvec: Vec<usize> = (0..nchains_per_beta).collect();
    for i in (1..nbeta).rev() {
        //println!("ibeta={}", i);
        let beta1 = beta_list[i];
        let beta2 = beta_list[i - 1];
        if beta1 > beta2 {
            panic!("beta list must be in decreasing order, with no duplicatation");
        }
        //rng.shuffle(&mut jvec);
        jvec.shuffle(rng);
        //let jvec=shuffle(&jvec, &mut rng);
        for (j2, &j1) in jvec.iter().enumerate() {
            let lp1 = -chains[i * nchains_per_beta + j1].current.potential_energy();
            let lp2 = -chains[(i - 1) * nchains_per_beta + j2]
                .current
                .potential_energy();
            let ep = exchange_prob(lp1, lp2, beta1, beta2).to_f64().unwrap();
            //println!("{}",ep);
            let r: f64 = rng.sample(Uniform::new(0.0, 1.0));
            if r < ep {
                let mut p1 = vec![T::zero(); chains[(i - 1) * nchains_per_beta + j2].dim()];
                let mut p2 = vec![T::zero(); chains[i * nchains_per_beta + j1].dim()];
                chains[(i - 1) * nchains_per_beta + j2]
                    .current
                    .write_position(&mut p1);
                chains[i * nchains_per_beta + j1]
                    .current
                    .write_position(&mut p2);

                chains[(i - 1) * nchains_per_beta + j2]
                    .set_position(&p2)
                    .unwrap();
                chains[i * nchains_per_beta + j1].set_position(&p1).unwrap();

                //let (a,b)=chains.split_at_mut(i * nchains_per_beta + j1);
                //a[(i-1)*nchains_per_beta+j2].swap(&mut b[0]);
            }
        }
    }
}
