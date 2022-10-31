use autodiff::Float;
use nuts::{
    forward_autodiff::{eval_grad, F},
    new_sampler,
    nuts::State,
    Chain, CpuLogpFunc, LogpError, LogpFromFn, SampleStats, SamplerArgs,
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
    sampler_args.step_size_adapt.target_accept = 0.8.into();
    sampler_args.num_tune = 1000;
    sampler_args.maxdepth = 20; // small value just for testing...
    sampler_args.mass_matrix_adapt.store_mass_matrix = true;

    // We instanciate our posterior density function
    //let logp_func = PosteriorDensity {};
    let logp_func = LogpFromFn::new(posterior, 2);

    let seed = 42;
    //let x0 = vec![0f64.into(); logp_func.dim()];
    let x0 = vec![3.5, -1.8];
    let mut sampler = new_sampler(logp_func.clone(), sampler_args.clone());

    let mut sampler2 = new_sampler(logp_func, sampler_args);

    // Set to some initial position and start drawing samples.
    sampler
        .set_position(&x0)
        .expect("Unrecoverable error during init");

    sampler2
        .set_position(&x0)
        .expect("Unrecoverable error during init");
    //let mut trace = vec![]; // Collection of all draws
    //let mut stats = vec![]; // Collection of statistics like the acceptance rate for each draw
    let mut outfile = std::fs::File::create("a.txt").unwrap();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    for i in 0..200000000 {
        let (draw, info): (Vec<_>, _) = sampler
            .draw(&mut rng)
            .expect("Unrecoverable error during sampling");

        let (_draw, _info): (Vec<_>, _) = sampler2
            .draw(&mut rng)
            .expect("Unrecoverable error during sampling");

        let draw1: Vec<_> = draw.iter().map(|&x| F64::from(x)).collect();
        assert_eq!(info.logp(), posterior(&draw1).x);

        assert_eq!(info.logp(), -sampler.current.potential_energy());

        if i % 10000 == 0 {
            for j in 0..sampler.dim() {
                write!(&mut outfile, " {}", draw[j]).unwrap();
            }
            writeln!(&mut outfile).unwrap();
            sampler.swap(&mut sampler2);
        }
        //trace.push(draw);
        let _info_vec = info.to_vec(); // We can collect the stats in a Vec
                                       // Or get more detailed information about divergences

        if let Some(div_info) = info.divergence_info() {
            println!("Divergence at position {:?}", div_info.start_location());
        }

        //dbg!(&info);
        //stats.push(info);
    }
}
