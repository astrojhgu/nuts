use autodiff::Float;
use nuts::{
    forward_autodiff::{eval_grad, F},
    new_sampler,
    nuts::State,
    pt::swap_chains,
    Chain, CpuLogpFunc, LogpError, LogpFromFn, SamplerArgs,
};
use rand::SeedableRng;
use std::io::Write;
use thiserror::Error;

// Define a function that computes the unnormalized posterior density
// and its gradient.
#[derive(Clone)]
struct PosteriorDensity {}

// The density might fail in a recoverable or non-recoverable manner...
#[derive(Debug, Error)]
enum PosteriorLogpError {}
impl LogpError for PosteriorLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

type F64 = F<f64>;

fn posterior(input: &[F64]) -> F64 {
    let x = input[0];
    let y = input[1];
    -((x.powi(2) + y - F64::from(11.0)).powi(2) + (x + y.powi(2) - F64::from(7.0)).powi(2))
}

impl CpuLogpFunc<f64> for PosteriorDensity {
    type Err = PosteriorLogpError;

    // We define a 10 dimensional normal distribution
    fn dim(&self) -> usize {
        2
    }

    // The normal likelihood with mean 3 and its gradient.
    fn logp(&mut self, position: &[f64], grad1: &mut [f64]) -> Result<f64, Self::Err> {
        let logp = eval_grad(&posterior, position, grad1);
        Ok(logp)
    }
}

fn main() {
    // We get the default sampler arguments
    let mut sampler_args = SamplerArgs::<f64>::default();

    // and modify as we like
    sampler_args.step_size_adapt.target_accept = 0.8;
    sampler_args.num_tune = 1000;
    sampler_args.maxdepth = 20; // small value just for testing...
    sampler_args.mass_matrix_adapt.store_mass_matrix = true;

    // We instanciate our posterior density function
    //let logp_func = PosteriorDensity {};
    let logp_func = LogpFromFn::new(posterior, 2);

    let seed = 42;
    //let x0 = vec![0f64.into(); logp_func.dim()];
    let x0 = vec![3.5, -1.8];

    let beta_list = vec![1.0, 0.5];
    let mut samplers = beta_list
        .iter()
        .map(|&b| {
            let mut s = new_sampler(logp_func.with_beta(b), sampler_args);
            s.set_position(&x0).unwrap();
            s
        })
        .collect::<Vec<_>>();

    //let mut trace = vec![]; // Collection of all draws
    //let mut stats = vec![]; // Collection of statistics like the acceptance rate for each draw
    let mut outfile = std::fs::File::create("a.txt").unwrap();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    for i in 0..100000000 {
        samplers.iter_mut().for_each(|s| {
            let _ = s.draw(&mut rng).unwrap();
        });

        if i % 100 == 0 {
            if i > 2000 {
                let mut position = vec![0.0; samplers[0].dim()];
                samplers[0].current.write_position(&mut position);
                for x in position {
                    write!(&mut outfile, " {}", x).unwrap();
                }
                writeln!(&mut outfile).unwrap();
            }

            swap_chains(&mut samplers, &mut rng, &beta_list);
        }
        //dbg!(&info);
        //stats.push(info);
    }
}
