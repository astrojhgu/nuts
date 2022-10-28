use autodiff::Float;
use nuts::{new_sampler, Chain, CpuLogpFunc, LogpError, SampleStats, SamplerArgs, forward_autodiff::{F, eval_grad}};
use rand::SeedableRng;
use thiserror::Error;
use std::io::Write;

// Define a function that computes the unnormalized posterior density
// and its gradient.
struct PosteriorDensity {}

// The density might fail in a recoverable or non-recoverable manner...
#[derive(Debug, Error)]
enum PosteriorLogpError {}
impl LogpError for PosteriorLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

type F64=F<f64>;

fn posterior(input: &[F64])->F64{
    let x=input[0];
    let y=input[1];
    -((F64::from(1.5)-x+x*y).powi(2)+(F64::from(2.25)-x+x*y.powi(2)).powi(2)+(F64::from(2.625)-x+x*y.powi(3)).powi(2))
}



impl CpuLogpFunc<f64> for PosteriorDensity {
    type Err = PosteriorLogpError;

    // We define a 10 dimensional normal distribution
    fn dim(&self) -> usize {
        2
    }

    // The normal likelihood with mean 3 and its gradient.
    fn logp(&mut self, position: &[f64], grad1: &mut [f64]) -> Result<f64, Self::Err> {
        let logp=eval_grad(&posterior, position, grad1);
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
    let logp_func = PosteriorDensity {};

    let seed = 42;
    let x0=&vec![0f64.into(); logp_func.dim()];
    let mut sampler = new_sampler(logp_func, sampler_args);
    
    // Set to some initial position and start drawing samples.
    sampler
        .set_position(&x0)
        .expect("Unrecoverable error during init");
    let mut trace = vec![]; // Collection of all draws
    let mut stats = vec![]; // Collection of statistics like the acceptance rate for each draw
    let mut outfile=std::fs::File::create("a.txt").unwrap();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    for _ in 0..2000 {
        let (draw, info):(Vec<_>, _) = sampler.draw(&mut rng).expect("Unrecoverable error during sampling");
        for i in 0..sampler.dim(){
            write!(&mut outfile, " {}", draw[i]).unwrap();
        }
        writeln!(&mut outfile).unwrap();
        trace.push(draw);
        let _info_vec = info.to_vec(); // We can collect the stats in a Vec
                                       // Or get more detailed information about divergences
        if let Some(div_info) = info.divergence_info() {
            println!("Divergence at position {:?}", div_info.start_location());
        }
        
        dbg!(&info);
        stats.push(info);
    }
}
