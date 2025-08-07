// #![allow(unused)]
pub use minifb::{CursorStyle, Key, MouseMode, WindowOptions};
use nalgebra::{Matrix4, Vector4};
use rand::Rng;
use std::any::Any;

pub mod window;

// region codes, used for clipping
const INSIDE: u32 = 0b0000;
const LEFT: u32 = 0b0001;
const RIGHT: u32 = 0b0010;
const BOTTOM: u32 = 0b0100;
const TOP: u32 = 0b1000;

const EPSILON: f32 = 1e-6;

pub type Color = (u8, u8, u8, u8); // RGBA format
pub type Point3D = (f32, f32, f32); // 3D coordinates
pub type Triangle = [Point3D; 3]; // 3D triangle defined by 3 vertices
pub type Line3D = (Point3D, Point3D); // 3D line segment defined by end points


#[derive(Clone, Copy, Default)]
pub struct Pixel {
    pub color: Color, // RGBA format
    pub depth: f32, // depth value for z-buffering
}

#[derive(Clone, Default)]
pub struct Buffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Pixel>,
    pub matrix: Matrix4<f32>, // transformation matrix
}

impl Buffer {
    const CLEAR_PIXEL: Pixel = Pixel {
        color: (0, 0, 0, 0), // transparent black
        depth: f32::INFINITY, // max depth
    };
    pub fn new(width: usize, height: usize) -> Self {
        let mut buffer = Self {
            width,
            height,
            pixels: vec![Self::CLEAR_PIXEL; width * height],
            matrix: Matrix4::identity()
        };
        buffer.update_matrix();
        buffer
    }

    fn update_matrix(&mut self) {
        let (w, h) = (self.width as f32, self.height as f32);
        // scale matrix: scale x by w/2, and y by -h/2
        let scale_matrix: Matrix4<f32> = Matrix4::new(
            w / 2.0, 0.0, 0.0, 0.0,
            0.0, -h / 2.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );

        // translation matrix: translate by (w/2, h/2)
        let translation_matrix: Matrix4<f32> = Matrix4::new(
            1.0, 0.0, 0.0, w / 2.0,
            0.0, 1.0, 0.0, h / 2.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );

        self.matrix = translation_matrix * scale_matrix;
    }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> usize {
        // px coords -> 1D array index
        y * self.width + x
    }

    pub fn draw_pixel(&mut self, x: usize, y: usize, color: Color, depth: f32) {
        if x < self.width && y < self.height {
            let idx = self.index(x, y);
            let pixel = &mut self.pixels[idx];
            if depth < pixel.depth {
                pixel.color = color;
                pixel.depth = depth;
            }
        }
    }

    pub fn get_pixel(&self, x: usize, y: usize) -> Option<&Pixel> {
        if x < self.width && y < self.height {
            let idx = self.index(x, y);
            Some(&self.pixels[idx])
        } else {
            None
        }
    }

    pub fn get_pixel_mut(&mut self, x: usize, y: usize) -> Option<&mut Pixel> {
        if x < self.width && y < self.height {
            let idx = self.index(x, y);
            Some(&mut self.pixels[idx])
        } else {
            None
        }
    }

    pub fn merge(&mut self, other: &Buffer) {
        if self.width != other.width || self.height != other.height {
            panic!("Buffers must have the same dimensions to merge.");
        }

        for (self_pixel, other_pixel) in self.pixels.iter_mut().zip(other.pixels.iter()) {
            if other_pixel.depth < self_pixel.depth {
                *self_pixel = *other_pixel;
            }
        }
    }

    pub fn clear(&mut self) {
        self.pixels.fill(Self::CLEAR_PIXEL);
    }

    pub fn clear_color(&mut self, color: Color) {
        // clear buffer with a specific color
        self.pixels.fill(Pixel {
            color,
            depth: f32::INFINITY, // reset depth to max
        })
    }

    pub fn reset_depth(&mut self) {
        // reset depth values to max
        for pixel in &mut self.pixels {
            pixel.depth = f32::INFINITY;
        }
    }

    pub fn to_raw(&self) -> Vec<u32> {
        // convert to flat array of u32 in ARGB format
        self.pixels
            .iter()
            .map(|pixel| {
                  (pixel.color.3 as u32) << 24
                | (pixel.color.0 as u32) << 16
                | (pixel.color.1 as u32) << 8
                | (pixel.color.2 as u32)
            })
            .collect()
    }
}

#[derive(Default)]
pub struct Scene {
    pub camera: Camera,
    pub objects: Vec<Box<dyn Object>>,
}

impl Scene {
    pub fn new(camera: Camera, objects: Vec<Box<dyn Object>>) -> Self {
        Self { camera, objects }
    }

    pub fn render(&self, buffer: &mut Buffer) {
        const TYPE: &str = "full"; // full, wireframe, points

        // Render each object in the scene
        for object in &self.objects {
            let tris = object.tris();
            let mut tri_colors = [[(0, 0, 0, 255); 3]; 12];
            for (i, tri) in tri_colors.iter_mut().enumerate().take(tris.len()) {
                for (color_idx, color) in tri.iter_mut().enumerate() {
                    *color = random_color_seeded((i * 3 + color_idx) as u64);
                }
            }

            if object.is::<RectangularPrism>() {
                tri_colors = object.downcast_ref::<RectangularPrism>()
                    .unwrap()
                    .tri_colors;
            }

            if TYPE == "wireframe" {
                // draw wiremesh edges
                for tri in tris.iter() {
                    // For a triangle, the edges are always (0,1), (1,2), (2,0)
                    let tri_edges = [(0, 1), (1, 2), (2, 0)];
                    for &(i0, i1) in tri_edges.iter() {
                        let v1 = tri[i0];
                        let v2 = tri[i1];

                        // Try to clip the line in 3D space before projection
                        if let Some((clipped_v1, clipped_v2)) = clip_line_to_camera_plane(v1, v2, &self.camera) {
                            // Project the clipped endpoints
                            if let (Some(p1), Some(p2)) = (
                                project_to_screen_space(&clipped_v1, &self.camera, buffer),
                                project_to_screen_space(&clipped_v2, &self.camera, buffer)
                            ) {
                                let color: Color = (255, 255, 255, 255);
                                draw_line(buffer, (p1.0, p1.1), (p2.0, p2.1), color);
                            }
                        }
                    }
                }
            }

            // draw triangles
            if TYPE == "full" {
                for (i, tri) in tris.iter().enumerate() {
                    render_tri(buffer, &self.camera, tri, tri_colors[i % tri_colors.len()]);
                }
            }


        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    Perspective,
    Orthographic,
}

#[derive(Clone, Copy)]
pub struct Camera {
    transform: Transform,
    fov: f32, // field of view in degrees
    near: f32, // near clipping plane
    far: f32,  // far clipping plane
    projection: Projection, // projection type
    ortho_size: f32, // orthographic size (for orthographic projection)
    transform_matrix: Matrix4<f32>,
    projection_matrix: Matrix4<f32>,
}

impl Camera {
    pub fn new(position: Position, rotation: Rotation, scale: Scale, fov: f32, near: f32, far: f32) -> Self {
        let mut camera = Self { 
            transform: Transform {
                position,
                rotation,
                scale
            },
            transform_matrix: Matrix4::identity(),
            projection_matrix: Matrix4::identity(),
            fov,
            near,
            far,
            ortho_size: 50.0,
            projection: Projection::Perspective
        };
        camera.update_matrix();
        camera
    }
    
    fn update_matrix(&mut self) {
        // translation matrix
        let translation_matrix: Matrix4<f32> = Matrix4::new(
            1.0, 0.0, 0.0, -self.transform.position.x,
            0.0, 1.0, 0.0, -self.transform.position.y,
            0.0, 0.0, 1.0, -self.transform.position.z,
            0.0, 0.0, 0.0, 1.0
        );

        // yaw (rotation around y-axis)
        let (yaw_sin, yaw_cos) = self.transform.rotation.yaw.to_radians().sin_cos();
        let yaw_matrix: Matrix4<f32> = Matrix4::new(
            yaw_cos, 0.0, yaw_sin, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -yaw_sin, 0.0, yaw_cos, 0.0,
            0.0, 0.0, 0.0, 1.0
        );

        // pitch (rotation around x-axis)
        let (pitch_sin, pitch_cos) = self.transform.rotation.pitch.to_radians().sin_cos();
        let pitch_matrix: Matrix4<f32> = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, pitch_cos, -pitch_sin, 0.0,
            0.0, pitch_sin, pitch_cos, 0.0,
            0.0, 0.0, 0.0, 1.0
        );

        // roll (rotation around z-axis)
        let (roll_sin, roll_cos) = self.transform.rotation.roll.to_radians().sin_cos();
        let roll_matrix: Matrix4<f32> = Matrix4::new(
            roll_cos, -roll_sin, 0.0, 0.0,
            roll_sin, roll_cos, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
        self.transform_matrix = roll_matrix * pitch_matrix * yaw_matrix * translation_matrix;

        // projection matrix
        if self.projection == Projection::Orthographic {
            // orthographic projection
            let size = self.ortho_size; // vertical size of view
            let top = size / 2.0;
            let bottom = -top;
            let right = top;
            let left = -right;
            self.projection_matrix = make_ortho_projection(left, right, bottom, top);
        } else {
            // perspective projection
            let f = 1.0 / (self.fov.to_radians() / 2.0).tan();
            let n = self.near;
            let far = self.far;
            self.projection_matrix = Matrix4::new(
                f,   0.0, 0.0, 0.0,
                0.0, f,   0.0, 0.0,
                0.0, 0.0, (far + n) / (n - far), (2.0 * far * n) / (n - far),
                0.0, 0.0, -1.0, 0.0,
            );
        }
    }

    pub fn set_projection(&mut self, projection: Projection) {
        self.projection = projection;
        self.update_matrix();
    }

    pub fn projection(&self) -> Projection {
        self.projection
    }

    pub fn set_fov(&mut self, fov: f32) {
        self.fov = fov.clamp(1.0, 179.0); // clamp to valid range
        self.update_matrix();
    }

    pub fn fov(&self) -> f32 {
        self.fov
    }

    pub fn set_near(&mut self, near: f32) {
        self.near = near.max(0.01);
        self.update_matrix();
    }
    pub fn near(&self) -> f32 {
        self.near
    }
    pub fn set_far(&mut self, far: f32) {
        self.far = far.min(0.01);
        self.update_matrix();
    }
    pub fn far(&self) -> f32 {
        self.far
    }

    pub fn set_position(&mut self, position: Position) {
        self.transform.position = position;
        self.update_matrix();
    }

    pub fn position(&self) -> Position {
        self.transform.position
    }

    pub fn set_rotation(&mut self, rotation: Rotation) {
        self.transform.rotation = Rotation {
            // clamp camera pitch to prevent flipping
            pitch: rotation.pitch.clamp(-89.0, 89.0),
            yaw: rotation.yaw,
            roll: rotation.roll
        };
        self.update_matrix();
    }

    pub fn rotation(&self) -> Rotation {
        self.transform.rotation
    }
    pub fn pitch(&self) -> f32 { self.transform.rotation.pitch }
    pub fn yaw(&self) -> f32 { self.transform.rotation.yaw }
    pub fn roll(&self) -> f32 { self.transform.rotation.roll }

    pub fn r#move(&mut self, x: f32, y: f32, z: f32) {
        let position = self.transform.position;
        self.set_position(Position {
            x: position.x + x,
            y: position.y + y,
            z: position.z + z,
        });
    }

    pub fn rotate(&mut self, pitch: f32, yaw: f32, roll: f32) {
        let rotation = self.transform.rotation;
        self.set_rotation(Rotation {
            pitch: rotation.pitch + pitch,
            yaw: rotation.yaw + yaw,
            roll: rotation.roll + roll,
        });
    }

    pub fn ortho(&self) -> f32 {
        self.ortho_size
    }

    pub fn set_ortho(&mut self, size: f32) {
        self.ortho_size = size;
        self.update_matrix();
    }
}

impl Default for Camera {
    fn default() -> Self {
        let mut camera = Camera {
            transform: Transform::default(),
            fov: 90.0,
            near: 0.01,
            far: 100.0,
            transform_matrix: Matrix4::identity(),
            projection_matrix: Matrix4::identity(),
            ortho_size: 50.0,
            projection: Projection::Perspective,
        };
        camera.update_matrix();
        camera
    }
}

#[derive(Clone, Copy, Default)]
pub struct Transform {
    // 3D transformation properties
    pub position: Position,
    pub rotation: Rotation,
    pub scale: Scale,
}

impl Transform {
    // create a Transform with just position, using default rotation and scale
    pub fn at_position(position: Position) -> Self {
        Self {
            position,
            rotation: Rotation::default(),
            scale: Scale::default(),
        }
    }

    // create a Transform at specific coordinates with default rotation and scale
    pub fn at_coords(x: f32, y: f32, z: f32) -> Self {
        Self::at_position(Position { x, y, z })
    }

    // functions to get the raw values easier
    pub fn position(&self) -> (f32, f32, f32) {
        (self.position.x, self.position.y, self.position.z)
}
    pub fn pos_x(&self) -> f32 { self.position.x }
    pub fn pos_y(&self) -> f32 { self.position.y }
    pub fn pos_z(&self) -> f32 { self.position.z }

    pub fn rotation(&self) -> (f32, f32, f32) {
        (self.rotation.pitch, self.rotation.yaw, self.rotation.roll)
    }
    pub fn pitch(&self) -> f32 { self.rotation.pitch }
    pub fn yaw(&self) -> f32 { self.rotation.yaw }
    pub fn roll(&self) -> f32 { self.rotation.roll }

    pub fn scale(&self) -> (f32, f32, f32) {
        (self.scale.x, self.scale.y, self.scale.z)
    }
    pub fn scale_x(&self) -> f32 { self.scale.x }
    pub fn scale_y(&self) -> f32 { self.scale.y }
    pub fn scale_z(&self) -> f32 { self.scale.z }
}

#[derive(Clone, Copy, Default)]
pub struct Position {
    // 3D position coordinates
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Default)]
pub struct Rotation {
    pub pitch: f32, // up/down rotation
    pub yaw: f32,   // left/right rotation
    pub roll: f32,  // tilt rotation
}

#[derive(Clone, Copy)]
pub struct Scale {
    // 3D scale factors in each direction
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Default for Scale {
    fn default() -> Self {
        Self { x: 1.0, y: 1.0, z: 1.0 }
    }
}

// downcast to allow checking and casting to specific object types
trait ObjectDowncast {
    fn is<T: 'static>(&self) -> bool;
    fn downcast_ref<T: 'static>(&self) -> Option<&T>;
}

impl<T: Object + ?Sized> ObjectDowncast for T {
    fn is<U: 'static>(&self) -> bool {
        self.as_any().is::<U>()
    }
    fn downcast_ref<U: 'static>(&self) -> Option<&U> {
        self.as_any().downcast_ref::<U>()
    }
}

pub trait Object {
    fn as_any(&self) -> &dyn Any;

    fn r#move(&mut self, delta: (f32, f32, f32)) {
        let position = self.transform_mut().position;
        self.set_position(Position {
            x: position.x + delta.0,
            y: position.y + delta.1,
            z: position.z + delta.2,
        });
    }

    fn rotate(&mut self, delta: (f32, f32, f32)) {
        let rotation = self.transform_mut().rotation;
        self.set_rotation(Rotation {
            pitch: rotation.pitch + delta.0,
            yaw: rotation.yaw + delta.1,
            roll: rotation.roll + delta.2,
        });
    }

    fn scale(&mut self, delta: (f32, f32, f32)) {
        let scale = self.transform_mut().scale;
        self.set_scale(Scale {
            x: scale.x + delta.0,
            y: scale.y + delta.1,
            z: scale.z + delta.2,
        });
    }

    // getters/setters for properties directly
    fn set_position(&mut self, position: Position) {
        self.transform_mut().position = position;
    }
    fn get_position(&self) -> Position {
        self.transform().position
    }
    fn set_rotation(&mut self, rotation: Rotation) {
        self.transform_mut().rotation = rotation;
    }
    fn get_rotation(&self) -> Rotation {
        self.transform().rotation
    }
    fn set_scale(&mut self, scale: Scale) {
        self.transform_mut().scale = scale;
    }
    fn get_scale(&self) -> Scale {
        self.transform().scale
    }

    fn transform(&self) -> &Transform;
    fn transform_mut(&mut self) -> &mut Transform;
    fn vertices(&self) -> Vec<Point3D>;
    fn tris(&self) -> Vec<Triangle>;
}


#[derive(Clone, Copy, Default)]
pub struct RectangularPrism {
    pub transform: Transform,
    pub size: (f32, f32, f32), // width, height, depth
    pub tri_colors: [[Color; 3]; 12],
}

impl Object for RectangularPrism {
    fn as_any(&self) -> &dyn Any { self }
    fn transform(&self) -> &Transform {
        &self.transform
    }
    fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }

    fn vertices(&self) -> Vec<Point3D> {
        // returns the 8 corner vertices of the rectangular prism
        let (w, h, d) = self.size;
        let (x, y, z) = self.transform.position();
        vec![
            (x - w / 2.0, y - h / 2.0, z - d / 2.0), // back  bottom left
            (x + w / 2.0, y - h / 2.0, z - d / 2.0), // back  bottom right
            (x - w / 2.0, y + h / 2.0, z - d / 2.0), // back  top    left
            (x + w / 2.0, y + h / 2.0, z - d / 2.0), // back  top    right
            (x - w / 2.0, y - h / 2.0, z + d / 2.0), // front bottom left
            (x + w / 2.0, y - h / 2.0, z + d / 2.0), // front bottom right
            (x - w / 2.0, y + h / 2.0, z + d / 2.0), // front top    left
            (x + w / 2.0, y + h / 2.0, z + d / 2.0), // front top    right
        ]
    }

    fn tris(&self) -> Vec<Triangle> {
        let vertices = self.vertices();
        vec![
            // back face (normal -Z)
            [ vertices[0], vertices[2], vertices[1] ],
            [ vertices[1], vertices[2], vertices[3] ],
            // front face (normal +Z)
            [ vertices[4], vertices[5], vertices[6] ],
            [ vertices[5], vertices[7], vertices[6] ],
            // left face (normal -X)
            [ vertices[0], vertices[4], vertices[2] ],
            [ vertices[4], vertices[6], vertices[2] ],
            // right face (normal +X)
            [ vertices[1], vertices[3], vertices[5] ],
            [ vertices[5], vertices[3], vertices[7] ],
            // top face (normal +Y)
            [ vertices[2], vertices[6], vertices[3] ],
            [ vertices[3], vertices[6], vertices[7] ],
            // bottom face (normal -Y)
            [ vertices[0], vertices[1], vertices[4] ],
            [ vertices[1], vertices[5], vertices[4] ],
        ]
    }
}

impl RectangularPrism {
    pub fn new(transform: Transform, size: (f32, f32, f32)) -> Self {
        let mut colors = [[(0, 0, 0, 255); 3]; 12];
        for tri in colors.iter_mut() {
            for color in tri.iter_mut() {
                *color = random_color();
            }
        }
        Self { transform, size, tri_colors: colors }
    }

    pub fn new_cube(transform: Transform, size: f32) -> Self {
        // create a cube with equal sides
        Self::new(transform, (size, size, size))
    }

    pub fn new_cube_at(position: Position, size: f32) -> Self {
        Self::new_cube(
            Transform {
                position,
                rotation: Rotation::default(),
                scale: Scale::default(),
            },
            size
        )
    }

    /// Create a cube at the specified coordinates with default rotation and scale
    pub fn new_cube_at_coords(x: f32, y: f32, z: f32, size: f32) -> Self {
        Self::new_cube_at(Position { x, y, z }, size)
    }

    /// Create a rectangular prism at the specified position with default rotation and scale
    pub fn new_at(position: Position, size: (f32, f32, f32)) -> Self {
        Self::new(
            Transform {
                position,
                rotation: Rotation::default(),
                scale: Scale::default(),
            },
            size
        )
    }

    /// Create a rectangular prism at the specified coordinates with default rotation and scale
    pub fn new_at_coords(x: f32, y: f32, z: f32, size: (f32, f32, f32)) -> Self {
        Self::new_at(Position { x, y, z }, size)
    }
    
    pub fn edges_idx(&self) -> &'static [(usize, usize); 12] {
        // returns the index pairs of vertices for edges/lines
        &[
            (0, 2), (2, 6), (6, 4), (4, 0), // back face
            (1, 3), (3, 7), (7, 5), (5, 1), // front face
            (0, 1), (2, 3), (4, 5), (6, 7), // side edges connecting front ↔ back
        ]
    }

    pub fn edges(&self) -> Vec<(usize, usize)> {
        // returns the edges of the rectangular prism as pairs of vertex indices
        let mut edges = Vec::with_capacity(12);
        for &(start, end) in self.edges_idx().iter() {
            edges.push((start, end));
        }
        edges
    }
}


pub fn compute_region_code(x: f32, y: f32, x_min: f32, x_max: f32, y_min: f32, y_max: f32) -> u32 {
    // initialized as being inside
    let mut code = INSIDE;
    if x < x_min {
        code |= LEFT; // to the left
    }
    if x > x_max {
        code |= RIGHT; // to the right
    }
    if y < y_min {
        code |= BOTTOM; // below
    }
    if y > y_max {
        code |= TOP; // above
    }

    code
}

pub fn clip_line(p1: (f32, f32), p2: (f32, f32), bounds: (f32, f32, f32, f32)) -> Option<(f32, f32, f32, f32)> {
    // Cohen–Sutherland line clipping algorithm
    // https://www.geeksforgeeks.org/dsa/line-clipping-set-1-cohen-sutherland-algorithm/

    let (x_min, x_max, y_min, y_max) = bounds;
    let (mut x1, mut y1) = p1;
    let (mut x2, mut y2) = p2;
    // compute region codes for p1, p2
    let mut code1 = compute_region_code(x1, y1, x_min, x_max, y_min, y_max);
    let mut code2 = compute_region_code(x2, y2, x_min, x_max, y_min, y_max);

    // initialize line as outside the window
    let mut accept = false;

    loop {
        if code1 | code2 == 0b0000 {
            // both endpoints lie within window
            accept = true;
            break;
        }
        else if code1 & code2 != 0b0000 {
            // both endpoints are outside window in same region
            break;
        }
        else {
            // some segment of line lies within the window
            let (mut x, mut y) = (0.0, 0.0);

            // at least one endpoint is outside the window, pick it
            let code_out = if code1 != 0b0000 {
                code1
            } else {
                code2
            };

            // find intersection point;
            // using formulas y = y1 + slope * (x - x1),
            // x = x1 + (1 / slope) * (y - y1)
            if code_out & TOP != 0b0000 {
                // point is above the window
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1);
                y = y_max;
            }
            else if code_out & BOTTOM != 0b0000 {
                // point is below the window
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1);
                y = y_min;
            }
            else if code_out & RIGHT != 0b0000 {
                // point is to the right of window
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1);
                x = x_max;
            }
            else if code_out & LEFT != 0b0000 {
                // point is to the left of window
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1);
                x = x_min;
            }

            // now intersection point x, y is found
            // replace point outside window by intersection point
            if code_out == code1 {
                x1 = x;
                y1 = y;
                code1 = compute_region_code(x1, y1, x_min, x_max, y_min, y_max);
            } else {
                x2 = x;
                y2 = y;
                code2 = compute_region_code(x2, y2, x_min, x_max, y_min, y_max);
            }
        }
    }
    if accept {
        Some((x1, y1, x2, y2)) // return the clipped line segment
    } else {
        None // line is completely outside the window
    }
}

pub fn clip_line_to_camera_plane(v1: (f32, f32, f32), v2: (f32, f32, f32), camera: &Camera) -> Option<Line3D> {
    // clip a 3D line against the camera's near plane (z = -near_plane in camera space)
    // transform vertices to camera space
    let transform_vertex = |vertex: (f32, f32, f32)| -> (f32, f32, f32) {
        let (mut x, mut y, mut z) = vertex;
        
        // apply camera translation
        x -= camera.transform.position.x;
        y -= camera.transform.position.y;
        z -= camera.transform.position.z;

        // yaw (y axis)
        let (sin_yaw, cos_yaw) = camera.transform.rotation.yaw.to_radians().sin_cos();
        let x1 = cos_yaw * x + sin_yaw * z;
        let z1 = -sin_yaw * x + cos_yaw * z;

        // pitch (x axis)
        let (sin_pitch, cos_pitch) = camera.transform.rotation.pitch.to_radians().sin_cos();
        let y2 = cos_pitch * y - sin_pitch * z1;
        let z2 = sin_pitch * y + cos_pitch * z1;

        // roll (z axis)
        let (sin_roll, cos_roll) = camera.transform.rotation.roll.to_radians().sin_cos();
        let x3 = cos_roll * x1 - sin_roll * y2;
        let y3 = sin_roll * x1 + cos_roll * y2;

        (x3, y3, z2)
    };

    let cam_v1 = transform_vertex(v1);
    let cam_v2 = transform_vertex(v2);
    
    let near_plane = -0.01; // very close to camera
    
    // check if both points are behind camera
    if cam_v1.2 >= near_plane && cam_v2.2 >= near_plane {
        return None; // both behind camera
    }
    
    // check if both points are in front of camera
    if cam_v1.2 < near_plane && cam_v2.2 < near_plane {
        return Some((v1, v2)); // both in front, no clipping needed
    }
    
    // one point is behind, one is in front - need to clip
    let (front_point, behind_point, front_cam, behind_cam) = if cam_v1.2 < near_plane {
        (v1, v2, cam_v1, cam_v2)
    } else {
        (v2, v1, cam_v2, cam_v1)
    };
    
    // calculate intersection with near plane
    let t = (near_plane - front_cam.2) / (behind_cam.2 - front_cam.2);
    
    // interpolate to find intersection point in world space
    let intersect = (
        front_point.0 + t * (behind_point.0 - front_point.0),
        front_point.1 + t * (behind_point.1 - front_point.1),
        front_point.2 + t * (behind_point.2 - front_point.2),
    );
    
    Some((front_point, intersect))
}


pub fn draw_line(buffer: &mut Buffer, p1: (f32, f32), p2: (f32, f32), color: Color) {
    let bounds = (0.0, buffer.width as f32, 0.0, buffer.height as f32);
    let line = clip_line(p1, p2, bounds);
    let (x0, y0, x1, y1) = match line {
        Some(l) => l,
        None => {
            // line is completely outside the window, do not draw
            return;
        }
    };

    let algorithm = 1; // 1 = Bresenham, 2 = DDA
    match algorithm {
        1 => {
            let (mut x0, mut y0, x1, y1) = (
                x0 as i32, y0 as i32, x1 as i32, y1 as i32
            );

            let dx = (x1 - x0).abs();
            let dy = -(y1 - y0).abs();
            let sx = if x0 < x1 { 1 } else { -1 };
            let sy = if y0 < y1 { 1 } else { -1 };

            let mut err = dx + dy;

            let mut i = 0;
            loop {
                if i > 10000 {
                    break; // prevent infinite loop
                } else {
                    i += 1;
                }
                if x0 >= 0 && y0 >= 0 && (x0 as usize) < buffer.width && (y0 as usize) < buffer.height {
                    buffer.draw_pixel(x0 as usize, y0 as usize, color, 2.0);
                }
                if x0 == x1 && y0 == y1 {
                    break;
                }

                let err2 = err * 2;
                if err2 >= dy {
                    err += dy;
                    x0 += sx;
                }
                if err2 <= dx {
                    err += dx;
                    y0 += sy;
                }
            }
        }
        2 => {
            let dx = x1 - x0;
            let dy = y1 - y0;

            let steps = dx.abs().max(dy.abs()) as usize;
            if steps == 0 { return; }

            let x_inc = dx / (steps as f32);
            let y_inc = dy / (steps as f32);
            let (mut x, mut y) = (x0, y0);

            for _ in 0..steps {
                x += x_inc;
                y += y_inc;
                if x >= 0.0 && y >= 0.0 && (x as usize) < buffer.width && (y as usize) < buffer.height {
                    buffer.draw_pixel(x as usize, y as usize, color, 2.0);
                }
            }
        }

        _ => {}
    }
}

pub fn project_to_screen_space(vertex: &(f32, f32, f32), camera: &Camera, buffer: &Buffer) -> Option<(f32, f32)> {
    let mut point = Vector4::new(vertex.0, vertex.1, vertex.2, 1.0);
    point = project_world_to_view(&point, camera);
    point = project_view_to_clip(&point, buffer, camera);
    point = project_clip_to_ndc(&point);
    point = project_ndc_to_screen(&point, buffer);
    Some((point.x, point.y))
}

// ================================ MATRIX RENDERING ================================

pub fn render_tri(buffer: &mut Buffer, camera: &Camera, tri: &Triangle, colors: [Color; 3]) {
    // tri points assumed to be in world space for now

    // convert triangle vertices to homogeneous coordinates (x, y, z, w)
    // this allows for easier matrix transformations
    let tri_hom: [Vector4<f32>; 3] = [
        Vector4::new(tri[0].0, tri[0].1, tri[0].2, 1.0),
        Vector4::new(tri[1].0, tri[1].1, tri[1].2, 1.0),
        Vector4::new(tri[2].0, tri[2].1, tri[2].2, 1.0),
    ];

    // world space -> view space
    let tri_view: [Vector4<f32>; 3] = [
        project_world_to_view(&tri_hom[0], camera),
        project_world_to_view(&tri_hom[1], camera),
        project_world_to_view(&tri_hom[2], camera),
    ];

    // backface culling
    if is_backface(&tri_view, camera) {
        return; // skip rendering
    }

    let clip_result = clip_triangle_against_near_plane(&tri_view, camera);
    if clip_result.is_empty() {
        return; // skip rendering if triangle is fully outside near plane
    }
    for clipped in clip_result {
        // view space -> clip space
        let tri_clip: [Vector4<f32>; 3] = [
            project_view_to_clip(&clipped[0], buffer, camera),
            project_view_to_clip(&clipped[1], buffer, camera),
            project_view_to_clip(&clipped[2], buffer, camera),
        ];

        if is_fully_outside_clip_space(tri_clip) {
            continue; // skip rendering if triangle is fully outside clip space
        }

        let clip_w = (tri_clip[0].w, tri_clip[1].w, tri_clip[2].w);

        // clip space -> NDC (Normalized Device Coordinates)
        let tri_ndc: [Vector4<f32>; 3] = [
            project_clip_to_ndc(&tri_clip[0]),
            project_clip_to_ndc(&tri_clip[1]),
            project_clip_to_ndc(&tri_clip[2]),
        ];

        // check if triangle is fully outside NDC space
        if is_fully_outside_ndc(&tri_ndc) {
            continue;
        }

        // check for invalid projections
        for vertex in &tri_ndc {
            if !vertex.x.is_finite() || !vertex.y.is_finite() || !vertex.z.is_finite() {
                continue; // skip invalid triangles
            }
        }

        // NDC -> screen space (pixel coordinates)
        let tri_screen: [Vector4<f32>; 3] = [
            project_ndc_to_screen(&tri_ndc[0], buffer),
            project_ndc_to_screen(&tri_ndc[1], buffer),
            project_ndc_to_screen(&tri_ndc[2], buffer),
        ];

        let edge_function = |a: (f32, f32), b: (f32, f32), c: (f32, f32)| -> f32 {
            // determines which side of line ab point c is on (sign)
            // and the barycentric area (magnitude)
            (c.0 - a.0) * (b.1 - a.1) - (c.1 - a.1) * (b.0 - a.0)
        };

        let (a, b, c) = (tri_screen[0], tri_screen[1], tri_screen[2]);
        let area = edge_function((a.x, a.y), (b.x, b.y), (c.x, c.y));
        if area.abs() < EPSILON {
            continue; // degenerate triangle
        }
        let area_inv = 1.0 / area;

        // triangle bounding box to limit pixel checks
        let bounding_box = compute_clamped_bbox(&tri_screen, buffer.width as f32, buffer.height as f32);

        for x in bounding_box.0 as usize..bounding_box.2 as usize {
            for y in bounding_box.1 as usize..bounding_box.3 as usize {
                // check if point p is inside the triangle using edge function
                let p = (x as f32 + 0.5, y as f32 + 0.5);
                let mut w0 = edge_function((b.x, b.y), (c.x, c.y), p);
                let mut w1 = edge_function((c.x, c.y), (a.x, a.y), p);
                let mut w2 = edge_function((a.x, a.y), (b.x, b.y), p);
                if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0) {
                    // point is inside the triangle
                    w0 *= area_inv;
                    w1 *= area_inv;
                    w2 *= area_inv;
                    let bary = (w0, w1, w2);

                    // interpolate z-depth and color using barycentric weights
                    let depth = interp_perspective_scalar(
                        (a.z, b.z, c.z),
                        clip_w,
                        bary,
                    );
                    // color gradient
                    let pixel_color = interp_perspective_color(colors, clip_w, bary);

                    buffer.draw_pixel(x, y, pixel_color, depth);
                }
            }
        }
    }
}

pub fn project_world_to_view(point: &Vector4<f32>, camera: &Camera) -> Vector4<f32> {
    // projects a 3D point in world space into view/camera space (camera = origin, facing -z)
    camera.transform_matrix * point
}

pub fn project_view_to_clip(point: &Vector4<f32>, buffer: &Buffer, camera: &Camera) -> Vector4<f32> {
    // projects a 3D point in view space (camera = origin, facing -z)
    // into 3D clip space (x, y, z) where z is the depth

    let aspect_ratio = buffer.width as f32 / buffer.height as f32;
    let aspect_scale_matrix: Matrix4<f32> = Matrix4::new(
        1.0 / aspect_ratio, 0.0, 0.0, 0.0,
        0.0, 1.0,      0.0, 0.0,
        0.0, 0.0,      1.0, 0.0,
        0.0, 0.0,      0.0, 1.0,
    );

    aspect_scale_matrix * camera.projection_matrix * point
}

pub fn project_clip_to_ndc(point: &Vector4<f32>) -> Vector4<f32> {
    // projects a 3D point in clip space (x, y, z) into normalized device coordinates (NDC)
    // where x and y are in [-1.0, 1.0] range
    if point.w.abs() < EPSILON {
        // avoid division by zero
        return Vector4::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    }
    let x_ndc = point.x / point.w;
    let y_ndc = point.y / point.w;
    let z_ndc = point.z / point.w;

    Vector4::new(x_ndc, y_ndc, z_ndc, 1.0)
}

pub fn project_ndc_to_screen(point: &Vector4<f32>, buffer: &Buffer) -> Vector4<f32> {
    // projects a 3D point in normalized device coordinates (NDC)
    // into screen space (pixel coordinates)
    buffer.matrix * point
}

pub fn is_backface(tri: &[Vector4<f32>; 3], camera: &Camera) -> bool {
    // checks if a triangle is a backface (normal facing away from the camera)
    let edge1 = tri[1].xyz() - tri[0].xyz();
    let edge2 = tri[2].xyz() - tri[0].xyz();
    let normal = edge1.cross(&edge2); // unnormalized face normal

    if camera.projection == Projection::Orthographic {
        // orthographic view: camera direction is always -Z
        let view_direction = nalgebra::Vector3::new(0.0, 0.0, -1.0);
        normal.dot(&view_direction) >= 0.0
    } else {
        // perspective view: use view direction from camera to triangle centroid
        let centroid = (tri[0].xyz() + tri[1].xyz() + tri[2].xyz()) / 3.0;
        let view_direction = -centroid;
        normal.dot(&view_direction) <= 0.0 // normal points away from camera → backface
    }
}

pub fn clip_triangle_against_near_plane(tri: &[Vector4<f32>; 3], camera: &Camera) -> Vec<[Vector4<f32>; 3]> {
    // triangle is [A, B, C], each with view-space position (x,y,z)
    let near = if camera.projection == Projection::Perspective {
        camera.near()
    } else {
        -100000.0 // large value for orthographic projection
    };
    // keeps points in front of camera
    let inside_test = |p: Vector4<f32>| p.z <= -near;
    let mut result_vertices = Vec::new();

    for (p, q) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
        let p_in = inside_test(p);
        let q_in = inside_test(q);

        if p_in {
            result_vertices.push(p);
        }

        if p_in ^ q_in {
            // edge crosses plane, compute intersection
            let t = (-near - p.z) / (q.z - p.z);
            // interpolate all components (including any attributes)
            let i = p + t * (q - p);
            result_vertices.push(i);
        }
    }

    // result_vertices is 0..4 points; re-triangulate:
    if result_vertices.len() < 3 {
        // fully outside
        vec![]
    } else if result_vertices.len() == 3 {
        vec![[
            result_vertices[0],
            result_vertices[1],
            result_vertices[2],
        ]]// single triangle
    } else if result_vertices.len() == 4 {
        // two triangles
        vec![
            [
                result_vertices[0],
                result_vertices[1],
                result_vertices[2],
            ],
            [
                result_vertices[0],
                result_vertices[2],
                result_vertices[3],
            ],
        ]
    } else {
        // more than 4 points, which shouldn't happen in a triangle
        vec![]
    }
}


pub fn is_fully_outside_clip_space(tri: [Vector4<f32>; 3]) -> bool {
    const FRUSTUM_PLANES_CLIP: [(usize, isize); 6] = [
        // axis (x, y, z), side
        (0, -1), // x' < -w'
        (0, 1), // x' >  w'
        (1, -1), // y' < -w'
        (1, 1), // y' >  w'
        (2, -1), // z' < -w'
        (2, 1), // z' >  w'
    ];
    for (axis, side) in FRUSTUM_PLANES_CLIP {
        let mut all_out = true;
        for vertex in &tri {
            let value = vertex[axis];
            let w = vertex.w;
            if (side == -1 && value >= -w) || (side == 1 && value <= w) {
                all_out = false;
            }
        }
        if all_out {
            // entirely outside this one plane -> reject
            return true;
        }
    }
    // potentially partially inside
    false
}

pub fn is_fully_outside_ndc(tri: &[Vector4<f32>; 3]) -> bool {
    // ndc space frustum culling
    if tri[0].x < -1.0 && tri[1].x < -1.0 && tri[2].x < -1.0 {
        return true; // all x_ndc < -1
    }
    if tri[0].x > 1.0 && tri[1].x > 1.0 && tri[2].x > 1.0 {
        return true; // all x_ndc > 1
    }
    if tri[0].y < -1.0 && tri[1].y < -1.0 && tri[2].y < -1.0 {
        return true; // all y_ndc < -1
    }
    if tri[0].y > 1.0 && tri[1].y > 1.0 && tri[2].y > 1.0 {
        return true; // all y_ndc > 1
    }
    if tri[0].z < -1.0 && tri[1].z < -1.0 && tri[2].z < -1.0 {
        return true; // all z_ndc < -1
    }
    if tri[0].z > 1.0 && tri[1].z > 1.0 && tri[2].z > 1.0 {
        return true; // all z_ndc > 1
    }

    false
}

fn compute_clamped_bbox(tri: &[Vector4<f32>; 3], screen_width: f32, screen_height: f32) -> (f32, f32, f32, f32) {
    let mut xmin = (tri[0].x.min(tri[1].x).min(tri[2].x)).floor();
    let mut xmax = (tri[0].x.max(tri[1].x).max(tri[2].x)).ceil();
    let mut ymin = (tri[0].y.min(tri[1].y).min(tri[2].y)).floor();
    let mut ymax = (tri[0].y.max(tri[1].y).max(tri[2].y)).ceil();
    xmin = xmin.max(0.0);
    ymin = ymin.max(0.0);
    xmax = xmax.min(screen_width - 1.0);
    ymax = ymax.min(screen_height - 1.0);

    (xmin, ymin, xmax, ymax)
}

fn interp_perspective_scalar(a: (f32, f32, f32), w_clip: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    // performs perspective-correct interpolation of a scalar attribute
    let inv0 = 1.0 / w_clip.0;
    let inv1 = 1.0 / w_clip.1;
    let inv2 = 1.0 / w_clip.2;

    let num = b.0 * (a.0 * inv0) + b.1 * (a.1 * inv1) + b.2 * (a.2 * inv2);
    let den = b.0 * inv0 + b.1 * inv1 + b.2 * inv2;
    if den.abs() < EPSILON {
        return 0.0;
    }
    num / den
}

fn interp_perspective_color(colors: [Color; 3], w_clip: (f32, f32, f32), bary: (f32, f32, f32)) -> Color {
    // performs perspective-correct interpolation of a color
    let (b0, b1, b2) = bary;
    let (w0_clip, w1_clip, w2_clip) = w_clip;
    let inv0 = 1.0 / w0_clip;
    let inv1 = 1.0 / w1_clip;
    let inv2 = 1.0 / w2_clip;

    // helper to do one color channel
    let interp_channel = |c0: u8, c1: u8, c2: u8| -> u8 {
        let a0 = c0 as f32;
        let a1 = c1 as f32;
        let a2 = c2 as f32;
        let num = b0 * (a0 * inv0) + b1 * (a1 * inv1) + b2 * (a2 * inv2);
        let den = b0 * inv0 + b1 * inv1 + b2 * inv2;
        if den.abs() < EPSILON {
            return 0;
        }
        let v = num / den;
        // clamp to [0,255] and round
        v.clamp(0.0, 255.0).round() as u8
    };

    let r = interp_channel(colors[0].0, colors[1].0, colors[2].0);
    let g = interp_channel(colors[0].1, colors[1].1, colors[2].1);
    let b = interp_channel(colors[0].2, colors[1].2, colors[2].2);
    let a = interp_channel(colors[0].3, colors[1].3, colors[2].3);
    (r, g, b, a)
}

pub fn make_ortho_projection(left: f32, right: f32, bottom: f32, top: f32) -> Matrix4<f32> {
    // creates an orthographic projection matrix
    let near = -100000.0; // near plane
    let far = 100000.0; // far plane
    Matrix4::new(
        2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left),
        0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom),
        0.0, 0.0, 2.0 / (near - far), (far + near) / (near - far),
        0.0, 0.0, 0.0, 1.0,
    )
}

// ================================================================

pub fn hue_shift(r: u8, g: u8, b: u8, hue_deg: f32) -> (u8, u8, u8) {
    let (h, s, v) = rgb_to_hsv(r, g, b);
    let new_h = (h + hue_deg) % 360.0;
    hsv_to_rgb(new_h, s, v)
}

pub fn random_color() -> Color {
    let mut rng = rand::rng();
    let hsv = (
        rng.random_range(0.0..=360.0),
        rng.random_range(0.5..=1.0),
        rng.random_range(0.9..=1.0),
    );
    let (r, g, b) = hsv_to_rgb(hsv.0, hsv.1, hsv.2);
    (r, g, b, 255)
}

pub fn random_color_seeded(seed: u64) -> Color {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(seed);
    (
        rng.random_range(0..=255),
        rng.random_range(0..=255),
        rng.random_range(0..=255),
        255, // alpha (fully opaque)
    )
}

pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    // convert RGB to HSV color space
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let mut h = 0.0;
    if delta > 0.0 {
        if max == r {
            h = (g - b) / delta + if g < b { 6.0 } else { 0.0 };
        } else if max == g {
            h = (b - r) / delta + 2.0;
        } else {
            h = (r - g) / delta + 4.0;
        }
        h /= 6.0; // normalize to [0, 1]
    }

    let s = if max == 0.0 { 0.0 } else { delta / max };
    let v = max;

    (h * 360.0, s, v) // return hue in degrees
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    // convert HSV to RGB color space
    let h = h / 360.0; // normalize hue to [0, 1]
    let c = v * s; // chroma
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 1.0 / 6.0 {
        (c, x, 0.0)
    } else if h < 2.0 / 6.0 {
        (x, c, 0.0)
    } else if h < 3.0 / 6.0 {
        (0.0, c, x)
    } else if h < 4.0 / 6.0 {
        (0.0, x, c)
    } else if h < 5.0 / 6.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (
        ((r + m) * 255.0).round() as u8,
        ((g + m) * 255.0).round() as u8,
        ((b + m) * 255.0).round() as u8,
    )
}
