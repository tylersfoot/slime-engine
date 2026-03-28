use cgmath::prelude::*;
use cgmath::{Point3, Matrix4, Vector3, Deg};
use crate::Key;


#[derive(Debug)]
pub struct Camera {
    // position of the camera in 3D world space
    pub eye: Point3<f32>,
    // the point in space the camera is looking at
    // the direction the camera is facing = the vector from eye to target
    pub target: Point3<f32>,
    // defines the "up" direction for the camera so it doesn't roll on its side
    pub up: Vector3<f32>,
    // aspect ratio of the screen (w/h) to prevent stretched/squished image
    pub aspect: f32,
    // vertical field of view in degrees; basically zoom
    pub fovy: f32,
    // near/far clipping planes; any geometry outside this range will not be drawn
    pub znear: f32,
    pub zfar: f32,
}
 
impl Camera {
    fn build_view_projection_matrix(&self) -> Matrix4<f32> {
        // world space -> view space
        // moves and rotates the whole world so the camera is at (0,0,0) looking down -Z axis
        let view = Matrix4::look_at_rh(self.eye, self.target, self.up);

        // view space -> ndc space (normalized device coordinates)
        // squashes the viewing frustrum (a pyramid-ish) into a perfect cube (the ndc)
        // warps the scene to account for depth (like far objects look closer to the middle)
        let proj = cgmath::perspective(Deg(self.fovy), self.aspect, self.znear, self.zfar);

        #[rustfmt::skip]
        // wgpu's normalized device coordinates have the y-axis/x-axis range -1.0 to +1.0
        // and z-axis range 0.0 to +1.0; cgmath uses OpenGL's coordinate system, so
        // this matrix scales/translates to account for that
        pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0,
        );

        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}
 
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    // convert cgmath Matrix4 -> 4x4 f32 array
    pub view_proj: [[f32; 4]; 4],
}
 
impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
        }
    }
 
    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}
 
pub struct CameraController {
    pub speed: f32,
    pub is_up_pressed: bool,
    pub is_down_pressed: bool,
    pub is_forward_pressed: bool,
    pub is_backward_pressed: bool,
    pub is_left_pressed: bool,
    pub is_right_pressed: bool,
}
 
impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_up_pressed: false,
            is_down_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }
 
    pub fn handle_keys(&mut self, keys: &[Key]) {
        // handles key inputs
        self.is_up_pressed = keys.contains(&Key::Space);
        self.is_down_pressed = keys.contains(&Key::LeftShift);
        self.is_forward_pressed = keys.contains(&Key::W) | keys.contains(&Key::Up);
        self.is_left_pressed = keys.contains(&Key::A) | keys.contains(&Key::Left);
        self.is_backward_pressed = keys.contains(&Key::S) | keys.contains(&Key::Down);
        self.is_right_pressed = keys.contains(&Key::D) | keys.contains(&Key::Right);
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        // move the camera's eye based on inputs
        let forward = (camera.target - camera.eye);
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();
        let right = forward_norm.cross(camera.up);

        // prevents glitching when camera gets too close to the center of the scene
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}