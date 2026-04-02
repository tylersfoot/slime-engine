#![allow(unused)]

use std::mem::ManuallyDrop;
use minifb::{KeyRepeat, MouseButton};
use wgpu::util::DeviceExt;
use image::GenericImageView;
pub use minifb::{CursorStyle, Key, MouseMode, WindowOptions};
use cgmath::prelude::*;
use std::time::{Duration, Instant};

pub mod window;
pub mod core;
pub mod render;
pub mod scene;
pub mod texture;
pub mod model;
pub mod resources;
pub mod camera;
pub mod transform;
pub mod node;

use crate::window::Window;
use crate::core::GraphicsContext;
use crate::render::Renderer;
use crate::scene::Scene;
use crate::transform::Transform;

// a struct to hold real-time debug information
struct DebugInfo {
    last_update: Instant,
    frame_count: u32, // for fps calc
    fps: f32,
    pub total_frames: u32,
}

impl DebugInfo {
    fn new() -> Self {
        Self {
            last_update: Instant::now(),
            frame_count: 0,
            fps: 0.0,
            total_frames: 0,
        }
    }

    // this will be called every frame from our main loop
    fn update(&mut self) {
        self.frame_count += 1;
        self.total_frames += 1;
        let elapsed = self.last_update.elapsed().as_secs_f32();

        // update the stats every half-second
        if elapsed >= 0.5 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.last_update = Instant::now();
        }
    }
}

pub trait App {
    // called once before the loop starts
    // meant for stuff like loading models and spawning initial nodes
    fn start(&mut self, app: &mut Engine);

    // called every frame
    // meant for game-loop stuff like moving nodes
    fn update(&mut self, app: &mut Engine, dt: Duration);
}

pub struct Engine<'a> {
    pub gfx: GraphicsContext<'a>,
    pub renderer: Renderer,
    pub scene: Scene,

    debug: DebugInfo,
    window_active: bool,
}

impl Engine<'_> {
    pub async fn new(window: Window) -> Self {
        // init GPU backend
        let gfx = GraphicsContext::new(window).await;
        
        // build the renderer (pipelines, layouts)
        let renderer = Renderer::new(&gfx);

        // build the scene (for camera, lights, models)
        let mut scene = Scene::new(&gfx, &renderer).await;

        let mut engine = Engine {
            gfx,
            renderer,
            scene,
            debug: DebugInfo::new(),
            window_active: false,
        };

        // apply the config to the surface
        // tells GPU to create a swap chain (set of textures to draw to) with the w/h/format
        let (width, height) = engine.gfx.window.get_size();
        engine.resize(width as u32, height as u32);

        engine
    }

    pub fn draw_frame(&mut self) {
        let (width, height) = self.gfx.window.get_size();
        let frame = match self.gfx.surface.get_current_texture() {
            // request a buffer/texture/frame from the swap chain
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.resize(width as u32, height as u32);
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                // skip this frame
                return;
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                // window resized or something else made the swap chain obsolete
                self.resize(width as u32, height as u32);
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                // swap chain was lost for a serious reason (like display driver reset)
                log::error!("Swapchain has been lost!");
                self.resize(width as u32, height as u32);
                return;
            }
        };
        
        // we draw to a TextureView instead of directly to the texture
        // a TextureView is like a lens or interpretation of a texture, it describes
        // how we want to look at and use the texture (eg. mipmap levels or array layers to use)
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // delegate drawing to renderer
        self.renderer.render(&self.gfx, &self.scene, &view);

        // tell the swap chain we're done drawing this frame, ready to be presented to the screen
        frame.present();

    }

    pub fn window(&self) -> &Window {
        &self.gfx.window
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.scene.resize(width, height);
        self.gfx.configure_surface(width, height);
        self.renderer.resize(&self.gfx);
    }

    pub fn run(mut self, mut app: impl App) {
        // run user's setup code first
        app.start(&mut self);

        let mut last_render_time = Instant::now();

        // main program loop
        loop {
            // processes events like key presses, mouse movements, close button, etc.
            self.gfx.window.update();

            // handle special keys
            let keys = self.gfx.window.get_keys();
            let mouse_pressed = self.gfx.window.get_mouse_down(MouseButton::Left);
            if keys.contains(&Key::Backspace) { return }

            if self.window_active &&
                (keys.contains(&Key::Escape) || !self.gfx.window.is_active()) {
                // unlock mouse if hit esc or unfocus window
                self.gfx.window.set_cursor_visibility(true);
                self.window_active = false;
            }
            if !self.window_active && mouse_pressed {
                // lock mouse if clicked on
                self.gfx.window.set_cursor_visibility(false);
                self.window_active = true;

                // snap mouse immediately to center to camera doesn't jump
                let (width, height) = self.gfx.window.get_size();
                self.gfx.window.set_mouse_pos((width / 2) as f32, (height / 2) as f32);
            }
            
            // handle infinite mouse movement
            let mut mouse_delta = (0.0, 0.0);
            if self.window_active && self.gfx.window.is_active()
                && let Some((x, y)) = self.gfx.window.get_mouse_pos(MouseMode::Pass) {
                    let (width, height) = self.gfx.window.get_size();
                    let center_x = (width / 2) as f32;
                    let center_y = (height / 2) as f32;

                    let dx = x - center_x;
                    let dy = y - center_y;

                    // only update and snap is mouse actually moved
                    if dx != 0.0 || dy != 0.0 {
                        mouse_delta = (dx, dy);
                        self.gfx.window.set_mouse_pos(center_x, center_y);
                    }
            }

            // handle scroll wheel
            let scroll_wheel_delta = self.gfx.window.get_scroll_wheel().unwrap_or((0.0, 0.0)).1;

            // handle inputs
            self.scene.camera_controller.handle_keys(&keys);
            self.scene.camera_controller.handle_mouse(mouse_delta.0, mouse_delta.1);
            self.scene.camera_controller.handle_mouse_scroll(scroll_wheel_delta);

            // calculate delta time
            let now = Instant::now();
            let dt = now - last_render_time;
            last_render_time = now;

            // run user per-frame logic
            app.update(&mut self, dt);

            // update internal engine states
            self.scene.update(dt, &self.gfx.device, &self.gfx.queue);

            if !self.gfx.window.is_open() {
                return; // exit if window is closed
            }

            self.draw_frame();

            self.debug.update();
            print!(
                "\rFPS: {:.2} | Total Frames: {:<6} | Node count: {}   ",
                self.debug.fps,
                self.debug.total_frames,
                self.scene.nodes.len()
            );
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    }
}
