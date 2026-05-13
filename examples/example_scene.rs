use cgmath::{Quaternion, Rotation3, Rad, Euler, Deg};
use slime_engine::{
    App,
    Engine,
    transform::Transform3D, 
    window::Window,
    WindowOptions,
    node::Node3D,
    primitives::Primitive,
    scene::{NodeId, CameraId},
    input::{Key},
    pollster::block_on,
    env_logger,
};
use std::time::Duration;

fn rand_color() -> [f32; 4] {
    [rand::random(), rand::random(), rand::random(), 1.0]
}

struct ExampleScene {
    time_passed: f32,
    camera: Option<CameraId>,
    camera2: Option<CameraId>,
    moving_cube_id: Option<NodeId>,
    floating_platform_id: Option<NodeId>,
    crazycorn_id: Option<NodeId>,
    planet_id: Option<NodeId>,
    moons: Vec<NodeId>,
    last_moon_toggle: f32,
    hidden_moon_index: usize,
    planet_visible: bool,
    last_planet_toggle: f32,
}

impl App for ExampleScene {
    fn start(&mut self, engine: &mut Engine) {
        let camera = engine.scene.spawn_camera(
            [0.0, 5.0, 20.0],
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

        let cube_model = engine.scene.load_primitive(Primitive::Cube, &engine.gfx, &engine.renderer);
        let crazycorn_model =  if let Some(model) = block_on(
            engine.scene.load_model("crazycorn/crazycorn.obj", &engine.gfx, &engine.renderer)
        ) {
            model
        } else {
            engine.scene.load_primitive(Primitive::Cube, &engine.gfx, &engine.renderer)
        };

        // moving cube
        let moving_cube_id = engine.scene.spawn_node(
            Node3D::new(Some(cube_model)).with_transform(
                Transform3D::new()
                    .with_position([-5.0, 5.0, 0.0])
                    .with_scale([1.0, 1.0, 1.0])
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
                    Node3D::new(Some(cube_model)).with_transform(
                        Transform3D::new()
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
                    Node3D::new(Some(cube_model)).with_transform(
                        Transform3D::new()
                            .with_position([x, floor_y + h / 2.0, z])
                            .with_scale([1.0, h, 1.0])
                    ).with_color(rand_color())
                );
            }
        }

        // floating tilted platform
        let floating_platform_id = engine.scene.spawn_node(
            Node3D::new(Some(cube_model)).with_transform(
                Transform3D::new()
                    .with_position([-8.0, 5.0, 2.0])
                    .with_rotation(Quaternion::from(Euler::new(Deg(30.0), Deg(45.0), Deg(0.0))))
                    .with_scale([3.0, 0.2, 3.0])
            ).with_color(rand_color())
        );
        self.floating_platform_id = Some(floating_platform_id);

        let crazycorn_id = engine.scene.spawn_node(
            Node3D::new(Some(crazycorn_model)).with_transform(
                Transform3D::new()
                    .with_position([30.0, 0.0, 0.0])
                    .with_scale([0.02, 0.02, 0.02])
            )
        );
        self.crazycorn_id = Some(crazycorn_id);

        let planet_id = engine.scene.spawn_node(
            Node3D::new(Some(cube_model))
                .with_transform(Transform3D::new()
                    .with_position([0.0, 7.0, 0.0])
                    .with_scale([1.5, 1.5, 1.5])
                )
                .with_color(rand_color())
        );
        self.planet_id = Some(planet_id);

        // extra cubes to make planet look cooler
        let gr = std::f32::consts::GOLDEN_RATIO;
        let gr_sq = gr * gr;
        engine.scene.spawn_node( // 72 deg
            Node3D::new(Some(cube_model))
                .with_parent(planet_id)
                .with_transform(Transform3D::new()
                    .with_rotation(Quaternion::from(Euler::new(
                        Deg(gr.atan().to_degrees()),
                        Deg((gr / 2.0).asin().to_degrees()),
                        Deg(-(1.0 / gr).atan().to_degrees())
                    )))
                )
                .with_color(rand_color())
        );
        engine.scene.spawn_node( // 144 deg
            Node3D::new(Some(cube_model))
                .with_parent(planet_id)
                .with_transform(Transform3D::new()
                    .with_rotation(Quaternion::from(Euler::new(
                        Deg(180.0 - (1.0 / gr_sq).atan().to_degrees()),
                        Deg(30.0),
                        Deg(-180.0 + gr_sq.atan().to_degrees())
                    )))
                )
                .with_color(rand_color())
        );
        engine.scene.spawn_node( // 216 deg
            Node3D::new(Some(cube_model))
                .with_parent(planet_id)
                .with_transform(Transform3D::new()
                    .with_rotation(Quaternion::from(Euler::new(
                        Deg(-180.0 + (1.0 / gr_sq).atan().to_degrees()),
                        Deg(-30.0),
                        Deg(-180.0 + gr_sq.atan().to_degrees())
                    )))
                )
                .with_color(rand_color())
        );
        engine.scene.spawn_node( // 288 deg
            Node3D::new(Some(cube_model))
                .with_parent(planet_id)
                .with_transform(Transform3D::new()
                    .with_rotation(Quaternion::from(Euler::new(
                        Deg(-gr.atan().to_degrees()),
                        Deg((-gr / 2.0).asin().to_degrees()),
                        Deg(-(1.0 / gr).atan().to_degrees())
                    )))
                )
                .with_color(rand_color())
        );

        // spawn moons in a ring around the planet
        for i in 0..20 {
            let angle = (i as f32 / 20.0) * std::f32::consts::TAU;
            let dist = 2.5;
            let moon_id = engine.scene.spawn_node(
                Node3D::new(Some(cube_model))
                    .with_parent(planet_id)
                    .with_transform(Transform3D::new()
                        .with_position([angle.cos() * dist, 0.0, angle.sin() * dist])
                        .with_rotation(Quaternion::from(Euler::new(
                            Deg(i as f32 * 12.0),
                            Deg(i as f32 * 20.0),
                            Deg(0.0)
                        )))
                        .with_scale([0.2, 0.2, 0.2])
                    ).with_color(rand_color())
            );
            self.moons.push(moon_id);
        }

    }

    fn update(&mut self, engine: &mut Engine, dt: Duration) {
        if let Some(camera) = self.camera && let Some(camera2) = self.camera2 {
            if engine.input.is_key_down(Key::C) {
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
            node.transform.position = [-5.0 + offset1, 4.0 + offset2, 0.0 + offset3].into();
        }

        // slowly spin the floating platform
        if let Some(node_id) = self.floating_platform_id
            && let Some(node) = engine.scene.nodes.get_mut(node_id) {
            let spin = Quaternion::from_angle_x(Rad(7.0_f32.to_radians() * delta))
                * Quaternion::from_angle_y(Rad(10.0_f32.to_radians() * delta))
                * Quaternion::from_angle_z(Rad(5.0_f32.to_radians() * delta));   
            node.transform.rotation = node.transform.rotation * spin;
        }

        // crazycorn go crazy
        if let Some(node_id) = self.crazycorn_id
            && let Some(node) = engine.scene.nodes.get_mut(node_id) {
            let scale = 0.01;
            let spin = Quaternion::from_angle_x(Rad((60.0_f32).to_radians() * delta))
                * Quaternion::from_angle_y(Rad(200.0_f32.to_radians() * delta))
                * Quaternion::from_angle_z(Rad(120.0_f32.to_radians() * delta));
            let new_rotation = node.transform.rotation * spin;

            node.transform = Transform3D::new()
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

        // rotate the planet (and therefore the moons)
        if let Some(p_id) = self.planet_id && let Some(planet) = engine.scene.nodes.get_mut(p_id) {
            let rot = Quaternion::from_angle_y(Rad(45.0_f32.to_radians() * delta));
            planet.transform.rotation = planet.transform.rotation * rot;
            
            // toggle planet visibility
            if self.time_passed - self.last_planet_toggle > 2.0 {
                self.planet_visible = !self.planet_visible;
                planet.visibility = self.planet_visible;
                self.last_planet_toggle = self.time_passed;
                if !self.planet_visible {
                    self.last_planet_toggle -= 1.5; // shorter time off
                }
            }
        }

        // cycle through moon visibility
        if self.time_passed - self.last_moon_toggle > 0.1 {
            self.last_moon_toggle = self.time_passed;
            
            // reset all to visible first
            for &m_id in &self.moons {
                if let Some(moon) = engine.scene.nodes.get_mut(m_id) {
                    moon.visibility = true;
                }
            }
            
            // hide the current moon
            let current_id = self.moons[self.hidden_moon_index];
            if let Some(moon) = engine.scene.nodes.get_mut(current_id) {
                moon.visibility = false;
            }

            self.hidden_moon_index = (self.hidden_moon_index + 1) % self.moons.len();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut window = Window::new(
        "example_scene",
        1000,
        800,
        WindowOptions {
            resize: true,
            ..Default::default()
        },
    ).unwrap_or_else(|e| panic!("{}", e));
    window.set_target_fps(60); // uncapped framerate
    let engine = block_on(Engine::new(window));

    let program = ExampleScene {
        time_passed: 0.0,
        camera: None,
        camera2: None,
        moving_cube_id: None,
        floating_platform_id: None,
        crazycorn_id: None,
        planet_id: None,
        moons: Vec::new(),
        last_moon_toggle: 0.0,
        hidden_moon_index: 0,
        planet_visible: true,
        last_planet_toggle: 0.0,
    };

    engine.run(program);
}
