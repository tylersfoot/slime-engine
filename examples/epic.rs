#![allow(unused)]
use cgmath::{num_traits::float, prelude::*, Quaternion, Rotation3, Rad, Euler, Deg};
use slime_engine::{
    App,
    Engine,
    transform::Transform, 
    window::Window,
    WindowOptions,
    node::Node,
    primitives::{Primitives, Primitive},
    model::ModelAsset,
    scene::{NodeId, ModelId, CameraId},
};
use std::time::Duration;
use pollster::block_on;

fn rand_color() -> [f32; 4] {
    [rand::random(), rand::random(), rand::random(), 1.0]
}

struct EpicGame {
    time_passed: f32,
    moving_cube_id: Option<NodeId>,
    floating_platform_id: Option<NodeId>,
    crazycorn_id: Option<NodeId>,
    camera: Option<CameraId>,
    camera2: Option<CameraId>,
}

impl App for EpicGame {
    fn start(&mut self, engine: &mut Engine) {
        let camera = engine.scene.spawn_camera(
            [0.0, 5.0, 10.0],
            -90.0,
            -20.0
        );
        self.camera = Some(camera);

        let camera2 = engine.scene.spawn_camera(
            [0.0, 20.0, 0.0],
            -90.0,
            -90.0
        );
        self.camera2 = Some(camera2);

        // let cube_model_id = block_on(engine.scene.load_model("unit_cube.obj", &engine.gfx, &engine.renderer)).unwrap();
        let cube_model = engine.scene.load_primitive(Primitive::Cube, &engine.gfx, &engine.renderer);
        let crazycorn_model = block_on(engine.scene.load_model("crazycorn/crazycorn.obj", &engine.gfx, &engine.renderer)).unwrap();

        // moving cube
        let moving_cube_id = engine.scene.spawn_node(
            Node::new(Some(cube_model)).with_transform(
                Transform::new()
                    .with_position([0.0, 20.0, -5.0])
                    .with_scale([2.0, 2.0, 2.0])
                ).with_color(rand_color())
        );
        self.moving_cube_id = Some(moving_cube_id);

        // floor grid
        let grid_size = 10;
        let tile_size = 4.0;
        let floor_y = 0.0;
        for i in 0..grid_size {
            for j in 0..grid_size {
                let x = (i as f32 - (grid_size as f32) / 2.0 + 0.5) * tile_size;
                let z = (j as f32 - (grid_size as f32) / 2.0 + 0.5) * tile_size;
                engine.scene.spawn_node(
                    Node::new(Some(cube_model)).with_transform(
                        Transform::new()
                            .with_position([x, floor_y, z])
                            .with_scale([tile_size, 2.0, tile_size])
                    ).with_color(rand_color())
                );
            }
        }

        // pillar field with varying heights
        let pillar_grid = 7;
        let spacing = 3.0;
        for i in 0..pillar_grid {
            for j in 0..pillar_grid {
                let x = (i as f32 - (pillar_grid as f32) / 2.0) * spacing;
                let z = (j as f32 - (pillar_grid as f32) / 2.0) * spacing;
                // height wobbles so it's not all uniform
                let h = 1.0 + ((i as f32 * 0.5).sin() + (j as f32 * 0.5).cos()).abs() * 3.0;
                engine.scene.spawn_node(
                    Node::new(Some(cube_model)).with_transform(
                        Transform::new()
                            .with_position([x, floor_y + h / 2.0, z])
                            .with_scale([1.0, h, 1.0])
                    ).with_color(rand_color())
                );
            }
        }

        // floating tilted platform
        let floating_platform_id = engine.scene.spawn_node(
            Node::new(Some(cube_model)).with_transform(
                Transform::new()
                    .with_position([0.0, 20.0, -5.0])
                    .with_rotation(Quaternion::from(Euler::new(Deg(30.0), Deg(45.0), Deg(0.0))))
                    .with_scale([3.0, 0.2, 3.0])
            ).with_color(rand_color())
        );
        self.floating_platform_id = Some(floating_platform_id);

        // ring of scattered cubes with varying rotation/size
        for k in 0..10 {
            let angle = k as f32 / 10.0 * std::f32::consts::TAU;
            let radius = 8.0;
            let x = angle.cos() * radius;
            let z = angle.sin() * radius - 5.0;
            let size = 0.5 + (k as f32 * 0.2);
            engine.scene.spawn_node(
                Node::new(Some(cube_model)).with_transform(
                Transform::new()
                    .with_position([x, 10.0, z])
                    .with_rotation(Quaternion::from(Euler::new(
                        Deg(k as f32 * 12.0),
                        Deg(k as f32 * 20.0),
                        Deg(0.0)
                    )))
                ).with_color(rand_color())
            );
        }

        let crazycorn_id = engine.scene.spawn_node(
            Node::new(Some(crazycorn_model)).with_transform(
                Transform::new()
                    .with_position([30.0, 0.0, 0.0])
                    .with_scale([0.02, 0.02, 0.02])
            )
        );
        self.crazycorn_id = Some(crazycorn_id);

    }

    fn update(&mut self, engine: &mut Engine, dt: Duration) {
        if let Some(camera) = self.camera && let Some(camera2) = self.camera2 {
            if engine.gfx.window.is_key_down(minifb::Key::C) {
                engine.scene.set_active_camera(camera2);
            } else {
                engine.scene.set_active_camera(camera);
            }
        }

        self.time_passed += dt.as_secs_f32();
        let delta = (dt.as_secs_f64() as f32).max(1e-6);

        let frequency = 2.0 * self.time_passed; // controls how fast it oscillates
        let amplitude = 1.5; // how far from center it moves
        let offset1 = (frequency).sin() * amplitude;
        let offset2 = (frequency * 1.5).cos() * amplitude;
        let offset3 = (frequency * 2.0).sin() * amplitude;

        if let Some(node_id) = self.moving_cube_id
            && let Some(node) = engine.scene.nodes.get_mut(node_id) {
            node.transform.position = [offset1, 13.0 + offset2, -5.0 + offset3].into();
        }

        // slowly spin the floating platform
        if let Some(node_id) = self.floating_platform_id
            && let Some(node) = engine.scene.nodes.get_mut(node_id) {
            let spin = Quaternion::from_angle_x(Rad(7.0_f32.to_radians() * delta))
                * Quaternion::from_angle_y(Rad(10.0_f32.to_radians() * delta))
                * Quaternion::from_angle_z(Rad(5.0_f32.to_radians() * delta));   
            node.transform.rotation = node.transform.rotation * spin;
        }

        if let Some(node_id) = self.crazycorn_id
            && let Some(node) = engine.scene.nodes.get_mut(node_id) {
            let scale = 0.01;
            let spin = Quaternion::from_angle_x(Rad((60.0_f32).to_radians() * delta))
                * Quaternion::from_angle_y(Rad(200.0_f32.to_radians() * delta))
                * Quaternion::from_angle_z(Rad(120.0_f32.to_radians() * delta));
            let new_rotation = node.transform.rotation * spin;

            node.transform = Transform::new()
                .with_position([
                    5.0,
                    5.0 + (self.time_passed * 4.0).sin() * scale * 5.0,
                    0.0
                ])
                .with_scale([
                    scale + (self.time_passed * 3.0).sin() * (scale/10.0),
                    scale + (self.time_passed * 3.0).sin() * (scale/10.0),
                    scale + (self.time_passed * 3.0).sin() * (scale/10.0)
                ])
                .with_rotation(new_rotation);
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
        moving_cube_id: None,
        floating_platform_id: None,
        crazycorn_id: None,
        camera: None,
        camera2: None,
    };

    engine.run(epic_game);
}
