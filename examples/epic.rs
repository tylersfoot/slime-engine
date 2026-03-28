#![allow(unused)]
use slime_engine::*;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    println!("starting gpu test");
    run();
}
