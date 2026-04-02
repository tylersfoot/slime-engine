#![allow(unused)]
use cgmath::prelude::*;
use slime_engine::{
    App,
    Engine,
    Transform, 
    Window,
    WindowOptions,
};
use std::time::Duration;
use pollster::block_on;

struct EpicGame {
    time_passed: f32,
    cube_id: Option<usize>,
}

impl App for EpicGame {
    fn start(&mut self, engine: &mut Engine) {
        let cannon_model_id = block_on(engine.scene.load_model("cannon/cannon.obj", &engine.gfx, &engine.renderer));
        let cube_model_id = block_on(engine.scene.load_model("cube2/cube.obj", &engine.gfx, &engine.renderer));
        let unit_cube_model_id = block_on(engine.scene.load_model("unit_cube.obj", &engine.gfx, &engine.renderer));

        const SPACE_BETWEEN: f32 = 2.0;
        const NUM_INSTANCES_PER_ROW: u32 = 1;
        for z in 0..NUM_INSTANCES_PER_ROW {
            for x in 0..NUM_INSTANCES_PER_ROW {
                let x_pos = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z_pos = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = cgmath::Vector3::new(x_pos, 0.0, z_pos);

                let rotation = if position.is_zero() {
                    cgmath::Quaternion::one()
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };

                engine.scene.spawn_node(
                    Some(cannon_model_id),
                    Transform::new().with_position(position).with_rotation(rotation)
                );
            }
        }

        let cube_id = engine.scene.spawn_node(
            Some(cube_model_id),
            Transform::new().with_position([2.0, 0.5, 2.0])
        );
        self.cube_id = Some(cube_id);

        engine.scene.spawn_node(
            Some(unit_cube_model_id),
            Transform::new()
                .with_position([0.0, -0.5, 0.0])
                .with_scale([500.0, 0.1, 500.0])
        );
    }

    fn update(&mut self, engine: &mut Engine, dt: Duration) {
        self.time_passed += dt.as_secs_f32();

        if let Some(node_id) = self.cube_id {
            let cube_node = &mut engine.scene.nodes[node_id];

            cube_node.transform.rotation = cube_node.transform.rotation * cgmath::Quaternion::from_axis_angle(
                cgmath::Vector3::unit_y(), 
                cgmath::Deg(45.0 * dt.as_secs_f32())
            );
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut window = Window::new(
        "epic window",
        640,
        480,
        WindowOptions {
            resize: true,
            ..Default::default()
        },
    ).unwrap_or_else(|e| panic!("{}", e));
    window.set_target_fps(0); // uncapped framerate
    let engine = block_on(Engine::new(window));

    let epic_game = EpicGame {
        time_passed: 0.0,
        cube_id: None,
    };

    engine.run(epic_game);
}
