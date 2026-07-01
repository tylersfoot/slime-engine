#![allow(unused)]

use cgmath::{Quaternion, Rotation3, Rad, Euler, Deg};
use slime_engine::{
    App,
    Engine,
    transform::Transform3D, 
    window::Window,
    WindowOptions,
    node::Node3D,
    primitives::Primitive,
    scene::{NodeId, CameraId, ModelId},
    input::{Key},
    pollster::block_on,
    env_logger,
};
use std::time::Duration;

fn rand_color() -> [f32; 4] {
    [rand::random(), rand::random(), rand::random(), 1.0]
}

enum BlockType {
    Grass,
    Dirt,
    Stone,
}

struct Block {
    is_active: bool,
    block_type: BlockType,
}

struct ExampleScene {
    time_passed: f32,
    camera: Option<CameraId>,
    cube_model: Option<ModelId>,
    // cube_id: Option<NodeId>,
}

impl App for ExampleScene {
    fn start(&mut self, engine: &mut Engine) {
        let camera = engine.scene.spawn_camera(
            [-10.0, 20.0, -10.0],
            45.0,
            -20.0
        );
        self.camera = Some(camera);

        let cube_model = engine.scene.load_primitive(Primitive::Cube, &engine.gfx, &engine.renderer);
        self.cube_model = Some(cube_model);


        const CHUNK_SIZE: usize = 64;

        // for i in 0..(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) {
        //     let x = i % CHUNK_SIZE;
        //     let y = (i / CHUNK_SIZE) % CHUNK_SIZE;
        //     let z = i / (CHUNK_SIZE * CHUNK_SIZE);

        //     engine.scene.spawn_node(
        //         Node3D::new(Some(cube_model)).with_transform(
        //             Transform3D::new()
        //                 .with_position([x as f32, y as f32, z as f32])
        //                 .with_scale([1.0, 1.0, 1.0])
        //             ).with_color(rand_color())
        //     );
        // }

        // temporary ai generated code to test bruteforce chunk example
        for i in 0..(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) {
            let x = i % CHUNK_SIZE;
            let y = (i / CHUNK_SIZE) % CHUNK_SIZE;
            let z = i / (CHUNK_SIZE * CHUNK_SIZE);

            let fx = x as f32;
            let fy = y as f32;
            let fz = z as f32;

            // 1. Calculate Terrain Height (Rolling hills using sine/cosine interference)
            let base_height = 8.0;
            let height_offset = (fx * 0.3).sin() * 2.0 + (fz * 0.4).cos() * 2.0;
            let surface_y = (base_height + height_offset).round() as usize;

            // 2. Calculate Cave Density (3D interference pattern)
            // If the density is above a certain threshold, it becomes "air"
            let cave_density = (fx * 0.6).sin() * (fy * 0.8).cos() * (fz * 0.6).sin();
            let is_cave = cave_density > 0.4;

            // 3. Spawning Logic
            // Only spawn a block if it is below the surface AND not a cave
            if y <= surface_y && !is_cave {
                
                // Contextual colors so you can test your new flat lighting!
                // Adjust these arrays to whatever your color type requires (vec4, rgba, etc.)
                let block_color = if y == surface_y {
                    [0.3, 0.7, 0.3, 1.0] // Grass (Green)
                } else if y >= surface_y.saturating_sub(2) {
                    [0.5, 0.3, 0.1, 1.0] // Dirt (Brown)
                } else {
                    [0.4, 0.4, 0.4, 1.0] // Stone (Gray)
                };

                engine.scene.spawn_node(
                    Node3D::new(Some(cube_model)).with_transform(
                        Transform3D::new()
                            .with_position([fx, fy, fz])
                            .with_scale([1.0, 1.0, 1.0])
                    ).with_color(block_color) // <-- Swap with your color setter
                );
            }
        }

    }

    fn update(&mut self, engine: &mut Engine, dt: Duration) {
        self.time_passed += dt.as_secs_f32();
        let delta = (dt.as_secs_f64() as f32).max(1e-6);

    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut window = Window::new(
        "voxel-engine",
        1000,
        800,
        WindowOptions {
            resize: true,
            ..Default::default()
        },
    ).unwrap_or_else(|e| panic!("{}", e));
    window.set_target_fps(200);
    let engine = block_on(Engine::new(window));

    let program = ExampleScene {
        time_passed: 0.0,
        camera: None,
        cube_model: None,
        // cube_id: None,
    };

    engine.run(program);
}
