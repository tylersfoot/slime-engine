
#![allow(unused)]
use slime_engine::*;
use slime_engine::window::*;
use slime_engine::object_import::*;
use std::time::{Instant, Duration};
use std::io::{self, Write};
use std::collections::VecDeque;

const WIDTH: usize = 400;
const HEIGHT: usize = 300;
const MIDDLE: (f32, f32) = (WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0);
const BENCHMARK: bool = false; // run benchmark (no window) instead

fn main() {
    if BENCHMARK {
        benchmark();
    }
    let mut buffer = Buffer::new(WIDTH, HEIGHT);
 
    let mut window = Window::new(
        "slime-engine",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            scale: minifb::Scale::X4,
            scale_mode: minifb::ScaleMode::AspectRatioStretch,
            ..WindowOptions::default()
        },
    ).unwrap_or_else(|e| {
        panic!("failed to create window: {e}");
    });
    window.set_position(700, 100);

    window.set_target_fps(1000);
    // window.set_cursor_visibility(false);

    let camera = Camera::new(
        Position { x: 10.0, y: 10.0, z: 10.0 },
        Rotation { pitch: 30.0, yaw: -45.0, roll: 0.0 },
        Scale { x: 1.0, y: 1.0, z: 1.0 },
        90.0, // field of view in degrees
        0.01, // near clipping plane
        10000.0, // far clipping plane
    );

    let (mut scene, moving_cube_id, floating_platform_id) = build_scene(camera);

    load_object_into_scene(
        &mut scene, 
        "objects/crazycorn.obj", 
        "objects/crazycorn.mtl",
        "crazycorn",
        Transform {
            position: Position { x: 30.0, y: 0.0, z: 0.0 },
            scale: Scale { x: 0.05, y: 0.05, z: 0.05 },
            ..Default::default()
        }
    );

    // orthographic example
    // scene.camera.set_projection(Projection::Orthographic);
    // scene.camera.set_ortho(50.0);

    // window.set_mouse_pos(MIDDLE.0, MIDDLE.1);

    let mut last_frame = Instant::now();
    let mut frame_times: VecDeque<Instant> = VecDeque::new();
    let mut last_print = Instant::now();
    let start = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Backspace) {
        let now = Instant::now();
        let frame_time = now - last_frame;
        last_frame = now;
        // instantaneous FPS (protect against zero delta)
        let inst_fps = 1.0 / frame_time.as_secs_f64().max(1e-6);
        // push this frame timestamp and drop anything older than 1 second
        frame_times.push_back(now);
        let cutoff = now - Duration::from_secs(1);
        while let Some(&front) = frame_times.front() {
            if front < cutoff {
                frame_times.pop_front();
            } else {
                break;
            }
        }
        // compute average FPS over the actual window
        let window_secs = if let (Some(&first), Some(&last)) =
            (frame_times.front(), frame_times.back())
        {
            (last - first).as_secs_f64().max(1e-6)
        } else {
            1.0 // fallback before enough samples
        };

        // throttle output so it updates every 100ms
        if now - last_print >= Duration::from_millis(100) {
            let fps_avg = (frame_times.len() as f64) / window_secs;
            print!(
                "\rFPS: {inst_fps:.2} | {fps_avg:.2}   "
            );
            io::stdout().flush().unwrap();
            last_print = now;
        }

        let delta = frame_time.as_secs_f64() as f32;
        let delta = delta.max(1e-6); // protect against zero delta
        let elapsed_time = (now - start).as_secs_f64() as f32;

        // handle mouse movement
        // let mouse_delta = if window.is_active() {
        //     let mouse_position = window.get_unscaled_mouse_pos(MouseMode::Discard)
        //         .unwrap_or((MIDDLE.0, MIDDLE.1));
        //     window.set_mouse_pos(MIDDLE.0, MIDDLE.1);
        //     (mouse_position.0 - MIDDLE.0, mouse_position.1 - MIDDLE.1)
        // } else {
        //     (0.0, 0.0)
        // };
        let mouse_delta = (0.0, 0.0);


        let mut camera_movement = (0.0, 0.0, 0.0);
        let mut camera_rotation = (0.0, 0.0, 0.0);
        let mut speed = 4.0;
        if window.is_key_down(Key::LeftShift) {
            speed *= 2.0;
        }

        if window.is_key_down(Key::W) {
            // move forward relative to camera's y-axis rotation
            let yaw = scene.camera.yaw().to_radians();
            camera_movement.0 += speed * yaw.sin();
            camera_movement.2 += -speed * yaw.cos();
        }
        if window.is_key_down(Key::S) {
            // move backward relative to camera's y-axis rotation
            let yaw = scene.camera.yaw().to_radians();
            camera_movement.0 += -speed * yaw.sin();
            camera_movement.2 += speed * yaw.cos();
        }
        if window.is_key_down(Key::A) {
            // move left relative to camera's y-axis rotation
            let yaw = scene.camera.yaw().to_radians();
            camera_movement.0 += -speed * yaw.cos();
            camera_movement.2 += -speed * yaw.sin();
        }
        if window.is_key_down(Key::D) {
            // move right relative to camera's y-axis rotation
            let yaw = scene.camera.yaw().to_radians();
            camera_movement.0 += speed * yaw.cos();
            camera_movement.2 += speed * yaw.sin();
        }
        if window.is_key_down(Key::Space) {
            camera_movement.1 += speed;
        }
        if window.is_key_down(Key::LeftCtrl) {
            camera_movement.1 -= speed;
        }

        if window.is_key_down(Key::Up) {
            camera_rotation.0 -= 1.0;
        }
        if window.is_key_down(Key::Down) {
            camera_rotation.0 += 1.0;
        }
        if window.is_key_down(Key::Left) {
            camera_rotation.1 -= 1.0;
        }
        if window.is_key_down(Key::Right) {
            camera_rotation.1 += 1.0;
        }
        if window.is_key_down(Key::Q) {
            camera_rotation.2 -= 1.0;
        }
        if window.is_key_down(Key::E) {
            camera_rotation.2 += 1.0;
        }
        if let Some(scroll) = window.get_scroll_wheel() {
            if scene.camera.projection() == Projection::Orthographic {
                // zoom in/out for orthographic camera
                let zoom = -(scroll.1 / 12.0);
                scene.camera.set_ortho(scene.camera.ortho() + zoom);
            } else {
                let fov = -(scroll.1 / 12.0);
                scene.camera.set_fov(scene.camera.fov() + fov);
            }
        }

        // apply mouse movement to camera rotation
        camera_rotation.0 += mouse_delta.1 * 0.2;
        camera_rotation.1 += mouse_delta.0 * 0.2;

        scene.camera.rotate(
            camera_rotation.0 * delta * 100.0,
            camera_rotation.1 * delta * 150.0,
            camera_rotation.2 * delta * 100.0
        );

        scene.camera.r#move(
            camera_movement.0 * delta,
            camera_movement.1 * delta,
            camera_movement.2 * delta
        );

        let frequency = 2.0 * elapsed_time; // controls how fast it oscillates
        let amplitude = 1.5; // how far from center it moves
        let offset1 = (frequency).sin() * amplitude;
        let offset2 = (frequency * 1.5).cos() * amplitude;
        let offset3 = (frequency * 2.0).sin() * amplitude;

        scene.move_node(moving_cube_id, (offset1, 13.0 + offset2, -5.0 + offset3));

        // slowly spin the floating platform
        scene.rotate_node(floating_platform_id, (30.0 * delta, 100.0 * delta, 60.0 * delta));
        let crazycorn_id = scene.get_node_id("crazycorn");
        if let Some(crazycorn_id) = crazycorn_id {
            scene.set_node_position(crazycorn_id, Position { x: 30.0, y: 5.0 + (elapsed_time*4.0).sin() * 5.0, z: 0.0 });
            scene.rotate_node(crazycorn_id, (30.0 * delta, 100.0 * delta, 60.0 * delta));
            scene.set_node_scale(crazycorn_id, Scale { x: 0.05 + (elapsed_time*3.0).sin() * 0.01, y: 0.05 + (elapsed_time*3.0).sin() * 0.01, z: 0.05 + (elapsed_time*3.0).sin() * 0.01 });
        }
        // scene.rotate_node(crazycorn_id, (30.0 * delta, 100.0 * delta, 60.0 * delta));

        buffer.clear();
        // buffer.reset_depth();

        // buffer = zoom_crop(&buffer, 2);
        // buffer = zoom_nearest(&buffer, 1.005);
        // bloom(&mut buffer, 20, 100, 0.05);
        // change_brightness(&mut buffer, 0.995);

        scene.render(&mut buffer);

        // let zoomed = zoom_nearest(&buffer, 1.2);
        // compress_color_space(&mut buffer, 8);
        // dither_floyd_steinberg(&mut buffer, 16);
        // ordered_dither(&mut buffer, 32);
        // background_gradient(&mut buffer, elapsed_time);

        let raw_buffer = buffer.to_raw();
        window
            .update_with_buffer(&raw_buffer, WIDTH, HEIGHT)
            .unwrap();
    }
}

fn benchmark() {
    let mut buffer = Buffer::new(WIDTH, HEIGHT);
    let camera = Camera::new(
        Position { x: 10.0, y: 10.0, z: 10.0 },
        Rotation { pitch: 30.0, yaw: -45.0, roll: 0.0 },
        Scale { x: 1.0, y: 1.0, z: 1.0 },
        90.0,
        0.01,
        10000.0,
    );

    let (mut scene, moving_cube_id, floating_platform_id) = build_scene(camera);

    load_object_into_scene(
        &mut scene, 
        "objects/crazycorn.obj", 
        "objects/crazycorn.mtl",
        "crazycorn",
        Transform {
            position: Position { x: 30.0, y: 0.0, z: 0.0 },
            scale: Scale { x: 0.05, y: 0.05, z: 0.05 },
            ..Default::default()
        }
    );

    let mut last_frame = Instant::now();
    let mut frame_times: VecDeque<Instant> = VecDeque::new();
    let mut last_print = Instant::now();
    let start = Instant::now();
    let mut frame_count = 0;
    let mut elapsed_time = 0.0;

    while elapsed_time < 10.0 {
        let now = Instant::now();
        let frame_time = now - last_frame;
        last_frame = now;
        // instantaneous FPS (protect against zero delta)
        let inst_fps = 1.0 / frame_time.as_secs_f64().max(1e-6);
        // push this frame timestamp and drop anything older than 1 second
        frame_times.push_back(now);
        let cutoff = now - Duration::from_secs(1);
        while let Some(&front) = frame_times.front() {
            if front < cutoff {
                frame_times.pop_front();
            } else {
                break;
            }
        }
        // compute average FPS over the actual window
        let window_secs = if let (Some(&first), Some(&last)) =
            (frame_times.front(), frame_times.back())
        {
            (last - first).as_secs_f64().max(1e-6)
        } else {
            1.0 // fallback before enough samples
        };

        // throttle output so it updates every 100ms
        if now - last_print >= Duration::from_millis(100) {
            let fps_avg = (frame_times.len() as f64) / window_secs;
            print!(
                "\rFPS: {inst_fps:.2} | {fps_avg:.2}   "
            );
            io::stdout().flush().unwrap();
            last_print = now;
        }

        let delta = frame_time.as_secs_f64() as f32;
        let delta = delta.max(1e-6); // protect against zero delta
        elapsed_time = (now - start).as_secs_f64() as f32;

        let frequency = 2.0 * elapsed_time; // controls how fast it oscillates
        let amplitude = 1.5; // how far from center it moves
        let offset1 = (frequency).sin() * amplitude;
        let offset2 = (frequency * 1.5).cos() * amplitude;
        let offset3 = (frequency * 2.0).sin() * amplitude;

        scene.move_node(moving_cube_id, (offset1, 13.0 + offset2, -5.0 + offset3));

        // slowly spin the floating platform
        let crazycorn_id = scene.get_node_id("crazycorn");
        if let Some(crazycorn_id) = crazycorn_id {
            scene.set_node_position(crazycorn_id, Position { x: 30.0, y: 5.0 + (elapsed_time*4.0).sin() * 5.0, z: 0.0 });
            scene.rotate_node(crazycorn_id, (30.0 * delta, 100.0 * delta, 60.0 * delta));
            scene.set_node_scale(crazycorn_id, Scale { x: 0.05 + (elapsed_time*3.0).sin() * 0.01, y: 0.05 + (elapsed_time*3.0).sin() * 0.01, z: 0.05 + (elapsed_time*3.0).sin() * 0.01 });
        }

        buffer.clear();
        scene.render(&mut buffer);
        let raw_buffer = buffer.to_raw();

        frame_count += 1;
    }

    let total_fps = frame_count as f32 / elapsed_time;
    println!("\nTotal FPS: {total_fps} ({frame_count} frames)");
    std::process::exit(0);
}
fn background_gradient(buffer: &mut Buffer, elapsed_time: f32) {
    for y in 0..buffer.height {
        for x in 0..buffer.width {
            let r = ((x * 255) / buffer.width) as u8;
            let g = ((y * 255) / buffer.height) as u8;
            let b = 128;
            // color shift by [frame] degrees
            let frame_shift = (elapsed_time * 50.0) % 360.0;
            let (r, g, b) = hue_shift(r/2, g/2, b/2, frame_shift);

            buffer.draw_pixel(x, y, (r, g, b, 255), 100.0);
        }
    }
}

fn build_scene(camera: Camera) -> (Scene, NodeId, NodeId) {
    let mut scene = Scene::new(camera);

    // moving cube
    let moving_cube_transform = Transform::at_position_raw(0.0, 20.0, -5.0);
    let moving_cube_id = scene.add_prism("Moving Cube", moving_cube_transform, (2.0, 2.0, 2.0), None);


    // floor grid
    let grid_size = 10;
    let tile_size = 4.0;
    let floor_y = 0.0;
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = (i as f32 - (grid_size as f32) / 2.0 + 0.5) * tile_size;
            let z = (j as f32 - (grid_size as f32) / 2.0 + 0.5) * tile_size;
            let tile_transform = Transform::at_position_raw(x, floor_y, z);
            scene.add_prism("Floor Tile", tile_transform, (tile_size, 0.5, tile_size), None);
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
            let pillar_transform = Transform::at_position_raw(x, floor_y + h / 2.0, z);
            scene.add_prism("Pillar", pillar_transform, (1.0, h, 1.0), None);
        }
    }

    // floating tilted platform
    let floating_platform_transform = Transform {
        position: Position { x: 0.0, y: 20.0, z: -5.0 },
        rotation: Rotation { pitch: 30.0, yaw: 45.0, roll: 0.0 },
        scale: Scale { x: 1.0, y: 1.0, z: 1.0 }, // Scale is now separate from size
    };
    let floating_platform_id = scene.add_prism(
        "Floating Platform",
        floating_platform_transform,
        (3.0, 0.2, 3.0),
        None
    );

    // ring of scattered cubes with varying rotation/size
    for k in 0..10 {
        let angle = k as f32 / 10.0 * std::f32::consts::TAU;
        let radius = 8.0;
        let x = angle.cos() * radius;
        let z = angle.sin() * radius - 5.0;
        let size = 0.5 + (k as f32 * 0.2);
        let cube_transform = Transform {
            position: Position { x, y: 10.0, z },
            rotation: Rotation { pitch: k as f32 * 12.0, yaw: k as f32 * 20.0, roll: 0.0 },
            ..Default::default()
        };
        scene.add_prism("Ring Cube", cube_transform, (size, size, size), None);
    }

    (scene, moving_cube_id, floating_platform_id)
}
