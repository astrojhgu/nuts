use autodiff::Float;
use nuts::{
    forward_autodiff::F, new_sampler, nuts::State, pt::swap_chains, Chain, LogpFromFn, SamplerArgs,
};
use rand::SeedableRng;
use std::io::Write;

type F64 = F<f64>;

fn normal(input: &[F64], m: F64, s: F64) -> F64 {
    (-input.iter().map(|&x| ((x - m) / s).powi(2)).sum::<F64>()).exp() / s.powi(input.len() as i32)
}

fn posterior(input: &[F64]) -> F64 {
    let s1 = F64::from(5e-2);
    let s2 = F64::from(1e-1);
    let m1 = F64::from(-1.0);
    let m2 = F64::from(1.0);

    (normal(input, m1, s1) + normal(input, m2, s2)).ln()
}

fn main() {
    // We get the default sampler arguments
    let mut sampler_args = SamplerArgs::<f64>::default();

    const DIM: usize = 2;
    // and modify as we like
    sampler_args.step_size_adapt.target_accept = 0.8;
    sampler_args.num_tune = 1000;
    sampler_args.maxdepth = 200; // small value just for testing...
    sampler_args.mass_matrix_adapt.store_mass_matrix = true;

    // We instanciate our posterior density function
    //let logp_func = PosteriorDensity {};
    let logp_func = LogpFromFn::new(posterior, DIM);

    let seed = 42;
    //let x0 = vec![0f64.into(); logp_func.dim()];
    let x0 = vec![0.0; DIM];

    let beta_list = vec![1.0, 0.5, 0.25, 0.125];
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

        if i % 1000 == 0 {
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
