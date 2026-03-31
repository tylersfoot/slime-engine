#![allow(unused)]
use slime_engine::*;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut window = Window::new(
        "awesome window",
        640,
        480,
        WindowOptions {
            resize: true,
            ..Default::default()
        },
    ).unwrap_or_else(|e| panic!("{}", e));
    window.set_target_fps(0); // uncapped framerate

    run(window);
}
