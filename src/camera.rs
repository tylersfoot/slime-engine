use cgmath::*;
use std::time::Duration;
use std::f32::consts::FRAC_PI_2;
use crate::Key;

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

#[derive(Debug)]
pub struct Camera {
    // position of the camera in 3D world space
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
}
 
impl Camera {
    pub fn new<
        V: Into<Point3<f32>>,
        Y: Into<Rad<f32>>,
        P: Into<Rad<f32>>,
    >(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Matrix4::look_to_rh(
            self.position,
            Vector3::new(
                cos_pitch * cos_yaw,
                sin_pitch,
                cos_pitch * sin_yaw
            ).normalize(),
            Vector3::unit_y(),
        )
    }
}
 
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_position: [f32; 4],
    // convert cgmath Matrix4 -> 4x4 f32 array
    pub view_proj: [[f32; 4]; 4],
}
 
impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: Matrix4::identity().into(),
        }
    }
 
    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
    }
}
 
pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}
 
impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
        }
    }
 
    pub fn handle_keys(&mut self, keys: &[Key]) {
        // handles key inputs
        self.amount_forward = f32::from((keys.contains(&Key::W) | keys.contains(&Key::Up)) as u8);
        self.amount_backward = f32::from((keys.contains(&Key::S) | keys.contains(&Key::Down)) as u8);
        self.amount_left = f32::from((keys.contains(&Key::A) | keys.contains(&Key::Left)) as u8);
        self.amount_right = f32::from((keys.contains(&Key::D) | keys.contains(&Key::Right)) as u8);
        self.amount_up = f32::from(keys.contains(&Key::Space) as u8);
        self.amount_down = f32::from(keys.contains(&Key::LeftShift) as u8);
        self.amount_forward = f32::from((keys.contains(&Key::W) | keys.contains(&Key::Up)) as u8);
    }

    pub fn handle_mouse(&mut self, mouse_dx: f32, mouse_dy: f32) {
        self.rotate_horizontal = mouse_dx;
        self.rotate_vertical = mouse_dy;
    }

    pub fn handle_mouse_scroll(&mut self, delta: f32) {
        self.scroll = delta;
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let dt = dt.as_secs_f32(); // delta time for consistent speed over time

        // move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;

        // move in/out (zoom)
        // note: this isn't actual zoom, the camera's position changes when zooming
        // added to make it easier to get closer to an object to focus on
        let (pitch_sin, pitch_cos) = camera.pitch.0.sin_cos();
        let scrollward = Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        // move up/down
        // since we dont use rooo, we can modify the y coordinate directly
        camera.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        // rotate
        camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        // safeguard
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // keep the camera's angle from going too high/low
        if camera.pitch < -Rad(SAFE_FRAC_PI_2) {
            camera.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if camera.pitch > Rad(SAFE_FRAC_PI_2) {
            camera.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}

pub struct Projection {
    // aspect ratio of the screen (w/h) to prevent stretched/squished image
    aspect: f32,
    // vertical field of view in degrees; basically zoom
    fovy: Rad<f32>,
    // near/far clipping planes; any geometry outside this range will not be drawn
    znear: f32,
    zfar: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(
        width: u32,
        height: u32,
        fovy: F,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        // view space -> ndc space (normalized device coordinates)
        // squashes the viewing frustrum (a pyramid-ish) into a perfect cube (the ndc)
        // warps the scene to account for depth (like far objects look closer to the middle)
        let projection = perspective(self.fovy, self.aspect, self.znear, self.zfar);

        // wgpu's normalized device coordinates have the y-axis/x-axis range -1.0 to +1.0
        // and z-axis range 0.0 to +1.0; cgmath uses OpenGL's coordinate system, so
        // this matrix scales/translates to account for that
        #[rustfmt::skip]
        pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0,
        );

        OPENGL_TO_WGPU_MATRIX * projection
    }
}