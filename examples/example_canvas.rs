use slime_engine::{
    App,
    Engine,
    transform::Transform2D, 
    window::Window,
    WindowOptions,
    node::Node2D,
    scene::NodeId,
    pollster::block_on,
    env_logger,
};
use std::time::Duration;

struct ExampleCanvas {
    time_passed: f32,
    squishy_cube_id: Option<NodeId>,
    spinny_cube_id: Option<NodeId>,
}

impl App for ExampleCanvas {
    fn start(&mut self, engine: &mut Engine) {
        let width = engine.canvas.window_width as f32;
        let height = engine.canvas.window_height as f32;

        // moving squishy cube at the top
        let squishy_cube = engine.canvas.nodes.insert(
            Node2D::new()
                .with_transform(Transform2D::new().with_position([400.0, 70.0]).with_scale([50.0, 50.0]))
                .with_color([0.0, 1.0, 1.0, 1.0])
        );
        self.squishy_cube_id = Some(squishy_cube);

        // background for squishy cube
        let padding = 20.0;
        engine.canvas.nodes.insert(
            Node2D::new()
                .with_transform(Transform2D::new()
                    .with_position([width / 2.0, 50.0 + padding])
                    .with_scale([width - (padding * 2.0), 100.0])
                )
                .with_color([0.1, 0.1, 0.15, 0.8])
                .with_z_index(-5)
        );

        // red square
        engine.canvas.nodes.insert(
            Node2D::new()
                .with_transform(Transform2D::new().with_position([80.0, height - 150.0]).with_scale([100.0, 100.0]))
                .with_color([1.0, 0.0, 0.0, 0.7])
                .with_z_index(0)
        );
        // green square
        engine.canvas.nodes.insert(
            Node2D::new()
                .with_transform(Transform2D::new().with_position([110.0, height - 120.0]).with_scale([100.0, 100.0]))
                .with_color([0.0, 1.0, 0.0, 0.7])
                .with_z_index(1)
        );
        // blue square
        engine.canvas.nodes.insert(
            Node2D::new()
                .with_transform(Transform2D::new().with_position([140.0, height - 90.0]).with_scale([100.0, 100.0]))
                .with_color([0.0, 0.5, 1.0, 0.7]) 
                .with_z_index(2)
        );

        // yellow spinny cube
        let spinny_cube = engine.canvas.nodes.insert(
            Node2D::new()
                .with_transform(Transform2D::new().with_position([width - 100.0, 200.0]).with_scale([75.0, 75.0]))
                .with_color([1.0, 1.0, 0.0, 1.0])
        );
        self.spinny_cube_id = Some(spinny_cube);

    }

    fn update(&mut self, engine: &mut Engine, dt: Duration) {
        self.time_passed += dt.as_secs_f32();
        let delta = (dt.as_secs_f64() as f32).max(1e-6);

        fn ease(x: f32) -> f32 {
            // convert to triangle wave (0 to 1)
            let v = if (x.floor() as i32) % 2 > 0 {
                1.0 - x.fract()
            } else {
                x.fract()
            };

            // cubic in-out easing
            if v < 0.5 {
                4.0 * v * v * v
            } else {
                let f = -2.0 * v + 2.0;
                1.0 - (f * f * f) / 2.0
            }
        }

        fn ease_derivative(x: f32) -> f32 {
            // returns cubic in-out easing derivative/velocity scaled to 0.0 - 1.0
            let v = if (x.floor() as i32) % 2 != 0 {
                1.0 - x.fract()
            } else {
                x.fract()
            };

            if v < 0.5 {
                (12.0 * v * v) / 3.0
            } else {
                (12.0 * (1.0 - v) * (1.0 - v)) / 3.0
            }
        }

        let padding = 20.0;
        let ease = ease(self.time_passed * 1.0);
        let velocity = ease_derivative(self.time_passed * 1.0);
        let color = [1.0 - velocity, velocity, 0.0, 1.0];
        let squish = (velocity - 0.5) * 1.0;

        // animate the squishy cube
        if let Some(id) = self.squishy_cube_id
            && let Some(node) = engine.canvas.nodes.get_mut(id) {
            node.transform.scale.x = 50.0 * (1.0 + squish);
            node.transform.scale.y = 50.0 * (1.0 - squish);

            let x_min = padding + node.transform.scale.x / 2.0;
            let x_max = engine.canvas.window_width as f32 - padding - node.transform.scale.x / 2.0;

            node.transform.position.x = ease * (x_max - x_min) + x_min;
            node.color = color;
        }

        // spin the spinny cube
        if let Some(id) = self.spinny_cube_id
            && let Some(node) = engine.canvas.nodes.get_mut(id) {
            node.transform.rotation += 20.0 * (self.time_passed * 1.0).sin() * delta; 
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut window = Window::new(
        "example_canvas",
        1200,
        600,
        WindowOptions {
            resize: true,
            ..Default::default()
        },
    ).unwrap_or_else(|e| panic!("{}", e));
    window.set_target_fps(0);
    
    let engine = block_on(Engine::new(window));

    let game = ExampleCanvas {
        time_passed: 0.0,
        spinny_cube_id: None,
        squishy_cube_id: None,
    };

    engine.run(game);
}