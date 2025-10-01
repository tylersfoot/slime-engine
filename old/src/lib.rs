// #![allow(unused)]
pub use minifb::{CursorStyle, Key, MouseMode, WindowOptions};
use nalgebra::{Matrix4, Point3, Rotation3, UnitQuaternion, Vector3, Vector4};
use rand::Rng;
use slotmap::{SlotMap, new_key_type};
use std::ops::Mul;

pub mod object_import;
pub mod window;
pub mod gpu;

use object_import::*;

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


// Unique IDs for various objects
new_key_type! {
    pub struct NodeId;
    pub struct MeshId;
    pub struct PrismId;
    pub struct MaterialId;
}

#[derive(Clone, Default)]
pub struct Buffer {
    pub width: usize,
    pub height: usize,
    pub color_buffer: Vec<Color>, // RGBA format
    pub depth_buffer: Vec<f32>, // depth value for z-buffering
    pub matrix: Matrix4<f32>, // transformation matrix
}

impl Buffer {
    const CLEAR_COLOR: Color = (0, 0, 0, 0);  // transparent black
    const CLEAR_DEPTH: f32 = f32::INFINITY; // max depth

    pub fn new(width: usize, height: usize) -> Self {
        let mut buffer = Self {
            width,
            height,
            color_buffer: vec![Self::CLEAR_COLOR; width * height],
            depth_buffer: vec![Self::CLEAR_DEPTH; width * height],
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
            let pixel = &mut self.color_buffer[idx];
            if depth < self.depth_buffer[idx] {
                *pixel = color;
                self.depth_buffer[idx] = depth;
            }
        }
    }

    pub fn get_pixel(&self, x: usize, y: usize) -> Option<&Color> {
        if x < self.width && y < self.height {
            let idx = self.index(x, y);
            Some(&self.color_buffer[idx])
        } else {
            None
        }
    }

    pub fn merge(&mut self, other: &Buffer) {
        if self.width != other.width || self.height != other.height {
            panic!("Buffers must have the same dimensions to merge.");
        }

        for i in 0..self.color_buffer.len() {
            if other.depth_buffer[i] < self.depth_buffer[i] {
                self.color_buffer[i] = other.color_buffer[i];
                self.depth_buffer[i] = other.depth_buffer[i];
            }
        }
    }

    pub fn clear(&mut self) {
        self.color_buffer.fill(Self::CLEAR_COLOR);
        self.depth_buffer.fill(Self::CLEAR_DEPTH);
    }

    pub fn clear_color(&mut self, color: Color) {
        // clear buffer with a specific color
        self.color_buffer.fill(color);
        self.depth_buffer.fill(f32::INFINITY); // reset depth to max
    }

    pub fn reset_depth(&mut self) {
        // reset depth values to max
        self.depth_buffer.fill(f32::INFINITY);
    }

    pub fn to_raw(&self) -> Vec<u32> {
        // convert to flat array of u32 in ARGB format
        self.color_buffer
            .iter()
            .map(|pixel| {
                  (pixel.3 as u32) << 24
                | (pixel.0 as u32) << 16
                | (pixel.1 as u32) << 8
                | (pixel.2 as u32)
            })
            .collect()
    }
}

#[derive(Default)]
pub struct Scene {
    pub camera: Camera,
    pub nodes: SlotMap<NodeId, Node>,
    pub meshes: SlotMap<MeshId, MeshComponent>,
    pub prisms: SlotMap<PrismId, RectangularPrismComponent>,
    pub materials: SlotMap<MaterialId, Material>,
}

impl Scene {
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            ..Default::default()
        }
    }

    pub fn add_node(
        &mut self,
        name: &str,
        transform: Transform,
        kind: Option<ObjectKind>,
        parent: Option<NodeId>,
    ) -> NodeId {
        // Adds a new node to the scene and returns its unique ID
        let node = Node {
            name: name.to_string(),
            transform,
            parent,
            children: Vec::new(),
            kind,
        };
        // Insert node into slotmap and get its key/ID
        let id = self.nodes.insert(node);

        // If parent exists, update the parent's children list
        if let Some(parent_id) = parent {
            self.nodes[parent_id].children.push(id);
        }

        id
    }

    pub fn get_node_id(&self, name: &str) -> Option<NodeId> {
        self.nodes.iter().find_map(|(id, node)| {
            if node.name == name {
                Some(id)
            } else {
                None
            }
        })
    }

    pub fn get_world_transform(&self, id: NodeId) -> Transform {
        // Recursively calculates the world transform of a node by walking up the parent chain
        // get the node or return identity transform
        if let Some(node) = self.nodes.get(id) {
            let mut transform = node.transform;
            let mut current_parent = node.parent;

            // loop up the heirarchy, multiplying transforms
            while let Some(parent_id) = current_parent {
                if let Some(parent_node) = self.nodes.get(parent_id) {
                    transform = parent_node.transform * transform;
                    current_parent = parent_node.parent;
                } else {
                    break; // parent not found
                }
            }
            transform
        } else {
            Transform::identity()
        }
    }

    pub fn delete_node(&mut self, id: NodeId) {
        // Recursively deletes a node and all of its children
        // clone the list of children IDs
        let children_to_delete = self.nodes.get(id).map_or(Vec::new(), |n| n.children.clone());

        for child_id in children_to_delete {
            // recursively delete children
            self.delete_node(child_id);
        }

        // remove the node itself
        if let Some(removed_node) = self.nodes.remove(id) {
            // remove node from parent's children list
            if let Some(parent_id) = removed_node.parent
                && let Some(parent_node) = self.nodes.get_mut(parent_id) {
                    parent_node.children.retain(|&child_id| child_id != id);
                }
        }
    }

    pub fn move_node(&mut self, id: NodeId, delta: (f32, f32, f32)) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.transform.position.x += delta.0;
            node.transform.position.y += delta.1;
            node.transform.position.z += delta.2;
        }
    }

    pub fn set_node_position(&mut self, id: NodeId, position: Position) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.transform.position = position;
        }
    }

    pub fn get_node_position(&self, id: NodeId) -> Option<Position> {
        self.nodes.get(id).map(|node| node.transform.position)
    }

    pub fn rotate_node(&mut self, id: NodeId, rotation: (f32, f32, f32)) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.transform.rotation.pitch += rotation.0;
            node.transform.rotation.yaw += rotation.1;
            node.transform.rotation.roll += rotation.2;
        }
    }

    pub fn set_node_rotation(&mut self, id: NodeId, rotation: Rotation) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.transform.rotation = rotation;
        }
    }

    pub fn get_node_rotation(&self, id: NodeId) -> Option<Rotation> {
        self.nodes.get(id).map(|node| node.transform.rotation)
    }

    pub fn scale_node(&mut self, id: NodeId, scale: (f32, f32, f32)) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.transform.scale.x *= scale.0;
            node.transform.scale.y *= scale.1;
            node.transform.scale.z *= scale.2;
        }
    }

    pub fn set_node_scale(&mut self, id: NodeId, scale: Scale) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.transform.scale = scale;
        }
    }

    pub fn get_node_scale(&self, id: NodeId) -> Option<Scale> {
        self.nodes.get(id).map(|node| node.transform.scale)
    }

    // TODO: add rotation + scale

    pub fn get_node_world_tris(&self, id: NodeId) -> Option<Vec<Triangle>> {
        let node = self.nodes.get(id)?;
        let kind = node.kind?;

        // get the local space tris based on the object kind
        let local_tris = match kind {
            ObjectKind::Mesh(mesh_id) => {
                let mesh = self.meshes.get(mesh_id)?;
                mesh.get_local_tris()
            }
            ObjectKind::RectangularPrism(prism_id) => {
                let prism = self.prisms.get(prism_id)?;
                prism.tris()
            }
        };

        // get the node's final world transform matrix
        let world_transform_matrix = self.get_world_transform_matrix(id);

        // apply the world transform to every vertex of every tri
        let world_tris = local_tris.into_iter().map(|tri| {
            let p0 = world_transform_matrix.transform_point(&Point3::new(tri[0].0, tri[0].1, tri[0].2));
            let p1 = world_transform_matrix.transform_point(&Point3::new(tri[1].0, tri[1].1, tri[1].2));
            let p2 = world_transform_matrix.transform_point(&Point3::new(tri[2].0, tri[2].1, tri[2].2));
            [
                (p0.x, p0.y, p0.z),
                (p1.x, p1.y, p1.z),
                (p2.x, p2.y, p2.z),
            ]
        }).collect();

        Some(world_tris)
    }

    pub fn get_world_transform_matrix(&self, node_id: NodeId) -> Matrix4<f32> {
        // Calculates the final world-space transformation matrix for a given node
        // It does this by traversing up the scene graph from the node to the root,
        // accumulating transforms along the way

        // start with the local transform of the requested node
        // if the ID is invalid, start with an identity matrix
        let mut current_transform = self.nodes.get(node_id)
            .map_or(Matrix4::identity(), |n| n.transform.to_matrix());

        // get the parent of the starting node
        let mut maybe_parent_id = self.nodes.get(node_id).and_then(|n| n.parent);

        // loop up the heirarchy until we reach a node with no parent (root node)
        while let Some(parent_id) = maybe_parent_id {
            if let Some(parent_node) = self.nodes.get(parent_id) {
                // pre-multiply by the parent's transform
                current_transform = parent_node.transform.to_matrix() * current_transform;

                maybe_parent_id = parent_node.parent;
            } else {
                // parent id was invalid for some reason so stop
                break;
            }
        }

        current_transform
    }

    pub fn add_prism(
        &mut self,
        name: &str,
        transform: Transform,
        size: (f32, f32, f32),
        parent: Option<NodeId>,
    ) -> NodeId {
        // create the data component
        let prism_component = RectangularPrismComponent {
            size,
            ..Default::default()
        };

        // add the component to its arena to get an ID
        let prism_id = self.prisms.insert(prism_component);

        // create the node that links the transform to the data
        let node = Node {
            name: name.to_string(),
            transform,
            parent,
            children: Vec::new(),
            kind: Some(ObjectKind::RectangularPrism(prism_id)),
        };

        // add the node to the scene graph and update its parent
        let node_id = self.nodes.insert(node);
        if let Some(parent_id) = parent
            && let Some(parent_node) = self.nodes.get_mut(parent_id) {
                parent_node.children.push(node_id);
            }

        node_id
    }

    pub fn render(&self, buffer: &mut Buffer) {
        const TYPE: &str = "full"; // "full", "wireframe", "points"

        let view_projection = create_view_projection_matrix(buffer, &self.camera);

        // iterate through every node in the scene graph
        for (node_id, node) in self.nodes.iter() {
            // skip nodes that don't have any geometry
            let Some(object_kind) = node.kind else {
                continue;
            };

            let local_tris: Vec<Triangle>;
            // color per vertex of one triangle
            let mut tri_colors: Option<[Color; 3]> = None;

            match object_kind {
                ObjectKind::Mesh(mesh_id) => {
                    if let Some(mesh) = self.meshes.get(mesh_id) {
                        local_tris = mesh.get_local_tris();

                        // apply material color
                        if let Some(material_id) = mesh.material_id
                            && let Some(material) = self.materials.get(material_id) {
                                // use the diffuse color (Kd) for shading
                                // convert from [0.0, 1.0] float to (u8, u8, u8, u8)
                                let r = (material.diffuse_color[0] * 255.0).round() as u8;
                                let g = (material.diffuse_color[1] * 255.0).round() as u8;
                                let b = (material.diffuse_color[2] * 255.0).round() as u8;
                                
                                // for now, apply the same color to all vertices of the triangle
                                tri_colors = Some([(r, g, b, 255), (r, g, b, 255), (r, g, b, 255)]);
                            }
                    } else {
                        continue;
                    }
                }
                ObjectKind::RectangularPrism(prism_id) => {
                    if let Some(prism) = self.prisms.get(prism_id) {
                        local_tris = prism.tris();
                    } else {
                        continue;
                    };
                }
            }

            // get the final transformation matrix for this node
            let world_transform_matrix = self.get_world_transform_matrix(node_id);

            // draw wiremesh edges
            if TYPE == "wireframe" {
                // apply the world transform to the local triangle
                for tri in &local_tris {
                    let p0 = world_transform_matrix
                        .transform_point(&Point3::new(tri[0].0, tri[0].1, tri[0].2));
                    let p1 = world_transform_matrix
                        .transform_point(&Point3::new(tri[1].0, tri[1].1, tri[1].2));
                    let p2 = world_transform_matrix
                        .transform_point(&Point3::new(tri[2].0, tri[2].1, tri[2].2));
                    let world_tri = [(p0.x, p0.y, p0.z), (p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z)];

                    // For a triangle, the edges are always (0,1), (1,2), (2,0)
                    let tri_edges = [(0, 1), (1, 2), (2, 0)];
                    for &(i0, i1) in tri_edges.iter() {
                        let v1 = world_tri[i0];
                        let v2 = world_tri[i1];

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

            // draw filled triangles
            if TYPE == "full" {
                // apply the world transform to the local triangle
                for (i, tri) in local_tris.iter().enumerate() {
                    let p0 = world_transform_matrix
                        .transform_point(&Point3::new(tri[0].0, tri[0].1, tri[0].2));
                    let p1 = world_transform_matrix
                        .transform_point(&Point3::new(tri[1].0, tri[1].1, tri[1].2));
                    let p2 = world_transform_matrix
                        .transform_point(&Point3::new(tri[2].0, tri[2].1, tri[2].2));
                    let world_tri = [(p0.x, p0.y, p0.z), (p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z)];

                    let color_data = match object_kind {
                        ObjectKind::Mesh(_) => {
                            // use the material color if available, otherwise fallback
                            // tri_colors.unwrap_or_else(|| [
                            //     random_color_seeded(i as u64 * 3),
                            //     random_color_seeded(i as u64 * 3 + 1),
                            //     random_color_seeded(i as u64 * 3 + 2),
                            // ])
                            tri_colors.unwrap_or([
                                (0, 0, 0, 255),
                                (0, 0, 0, 255),
                                (0, 0, 0, 255),
                            ])
                        }
                        ObjectKind::RectangularPrism(prism_id) => {
                            // use the prism's specific per-triangle colors
                            self.prisms[prism_id].tri_colors[i % 12]
                        }
                    };
                    render_tri(buffer, &self.camera, &world_tri, color_data, &view_projection);
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
                scale,
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
        self.far = far.max(0.01);
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
            roll: rotation.roll,
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

#[derive(Clone, Copy, Default, Debug)]
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
    pub fn at_position_raw(x: f32, y: f32, z: f32) -> Self {
        Self::at_position(Position { x, y, z })
    }

    pub fn from_matrix(matrix: &Matrix4<f32>) -> Self {
        // decomposes a 4x4 matrix into position, rotation, and scale
        Self {
            position: Position::from_matrix(matrix),
            rotation: Rotation::from_matrix(matrix),
            scale: Scale::from_matrix(matrix),
        }
    }

    pub fn to_matrix(&self) -> Matrix4<f32> {
        let t = self.position.to_matrix();
        let r = self.rotation.to_matrix();
        let s = self.scale.to_matrix();
        t * r * s
    }

    pub fn identity() -> Self {
        Self {
            position: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: Rotation {
                pitch: 0.0,
                yaw: 0.0,
                roll: 0.0,
            },
            scale: Scale {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
        }
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

impl Mul for Transform {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // combine two transforms by multiplying their matrices
        let self_matrix = self.to_matrix();
        let other_matrix = other.to_matrix();
        Self::from_matrix(&(self_matrix * other_matrix))
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Position {
    // 3D position coordinates
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Position {
    pub fn from_matrix(matrix: &Matrix4<f32>) -> Self {
        // translation/position vector is the first 3 components of the 4th column
        let pos_vec = matrix.column(3).xyz();
        Self { x: pos_vec.x, y: pos_vec.y, z: pos_vec.z }
    }

    pub fn to_matrix(&self) -> Matrix4<f32> {
        // position/translation matrix
        Matrix4::new_translation(&Vector3::new(self.x, self.y, self.z))
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Rotation {
    pub pitch: f32, // up/down rotation
    pub yaw: f32,   // left/right rotation
    pub roll: f32,  // tilt rotation
}

impl Rotation {
    pub fn from_matrix(matrix: &Matrix4<f32>) -> Self {
        // extract the scale to normalize the rotation
        let sx = matrix.column(0).magnitude();
        let sy = matrix.column(1).magnitude();
        let sz = matrix.column(2).magnitude();

        // check to avoid division by zero
        if sx == 0.0 || sy == 0.0 || sz == 0.0 {
            return Self { pitch: 0.0, yaw: 0.0, roll: 0.0 };
        }

        // create a pure rotation matrix by removing the scale
        let rotation_matrix = nalgebra::Matrix3::from_columns(&[
            matrix.column(0).xyz() / sx,
            matrix.column(1).xyz() / sy,
            matrix.column(2).xyz() / sz,
        ]);

        let rotation = Rotation3::from_matrix_unchecked(rotation_matrix);

        // convert the pure rotation matrix to a quaternion
        let rotation_quat = UnitQuaternion::from_rotation_matrix(&rotation);

        // convert the quaternion to Euler angles (roll, pitch, yaw)
        let (roll, pitch, yaw) = rotation_quat.euler_angles();

        Self {
            pitch: pitch.to_degrees(),
            yaw: yaw.to_degrees(),
            roll: roll.to_degrees(),
        }
    }

    pub fn to_matrix(&self) -> Matrix4<f32> {
        // convert degrees to radians for nalgebra functions
        let pitch_rad = self.pitch.to_radians();
        let yaw_rad = self.yaw.to_radians();
        let roll_rad = self.roll.to_radians();

        // create a quaternion from Euler angles
        let quaternion = UnitQuaternion::from_euler_angles(roll_rad, pitch_rad, yaw_rad);

        // convert the quaternion to a 4x4 matrix
        quaternion.to_homogeneous()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale {
    // 3D scale factors in each direction
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Scale {
    pub fn from_matrix(matrix: &Matrix4<f32>) -> Self {
        // scale is the magnitude (length) of each of the first three column vectors
        let sx = matrix.column(0).magnitude();
        let sy = matrix.column(1).magnitude();
        let sz = matrix.column(2).magnitude();
        Self { x: sx, y: sy, z: sz }
    }
    
    pub fn to_matrix(&self) -> Matrix4<f32> {
        Matrix4::new_nonuniform_scaling(&Vector3::new(self.x, self.y, self.z))
    }
}

impl Default for Scale {
    fn default() -> Self {
        Self { x: 1.0, y: 1.0, z: 1.0 }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ObjectKind {
    // what kind of data a Node is associated with
    Mesh(MeshId),
    RectangularPrism(PrismId),
}

#[derive(Debug)]
pub struct Node {
    pub name: String,
    pub transform: Transform,
    // hierarchical structure
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    // link to the actual geometry
    pub kind: Option<ObjectKind>,
}

#[derive(Clone, Default)]
pub struct MeshComponent {
    pub vertices: Vec<Point3D>,
    pub faces: Vec<Face>,
    pub material_id: Option<MaterialId>,
}

impl MeshComponent {
    pub fn get_local_tris(&self) -> Vec<Triangle> {
        // converts the indexed faces of the mesh into
        // a flat list of triangles in local space
        let mut tris = Vec::new();

        // iterate over each face defined in the mesh component
        for face in &self.faces {
            // a face must have at least 3 vertices to form a triangle
            if face.vertex_indices.len() < 3 {
                continue;
            }

            // use a "fan" triangulation method for polygons (faces with >3 vertices)
            // we anchor the fan on the first vertex of the face
            let v0_index = face.vertex_indices[0] as usize;
            let v0 = self.vertices[v0_index];

            // create tris by connecting the anchor to
            // every subsequent pair of vertices
            // for a quad (v0, v1, v2, v3) this creates tris (v0, v1, v2) and (v0, v2, v3)
            for i in 1..(face.vertex_indices.len() - 1) {
                let v1_index = face.vertex_indices[i] as usize;
                let v2_index = face.vertex_indices[i + 1] as usize;

                let v1 = self.vertices[v1_index];
                let v2 = self.vertices[v2_index];

                tris.push([v0, v1, v2]);
            }
        }
        tris
    }
}

#[derive(Clone, Copy)]
pub struct RectangularPrismComponent {
    pub size: (f32, f32, f32), // width, height, depth
    pub tri_colors: [[Color; 3]; 12],
}

impl Default for RectangularPrismComponent {
    fn default() -> Self {
        let mut colors = [[(0, 0, 0, 255); 3]; 12];
        for tri in colors.iter_mut() {
            for color in tri.iter_mut() {
                *color = random_color();
            }
        }
        Self {
            size: (0.0, 0.0, 0.0),
            tri_colors: colors,
        }
    }
}

impl RectangularPrismComponent {
    fn vertices(&self) -> [Point3D; 8] {
        // returns the 8 corner vertices of the rectangular prism in local space
        let (w, h, d) = self.size;
        [
            (-w / 2.0, -h / 2.0, -d / 2.0), // back  bottom left
            ( w / 2.0, -h / 2.0, -d / 2.0), // back  bottom right
            (-w / 2.0,  h / 2.0, -d / 2.0), // back  top    left
            ( w / 2.0,  h / 2.0, -d / 2.0), // back  top    right
            (-w / 2.0, -h / 2.0,  d / 2.0), // front bottom left
            ( w / 2.0, -h / 2.0,  d / 2.0), // front bottom right
            (-w / 2.0,  h / 2.0,  d / 2.0), // front top    left
            ( w / 2.0,  h / 2.0,  d / 2.0), // front top    right
        ]
    }

    pub fn tris(&self) -> Vec<Triangle> {
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

pub fn render_tri(
    buffer: &mut Buffer,
    camera: &Camera,
    tri: &Triangle,
    colors: [Color; 3],
    view_projection: &Matrix4<f32>
) {
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
        // view space -> clip space using precomputed matrix
        let tri_clip: [Vector4<f32>; 3] = [
            view_projection * clipped[0],
            view_projection * clipped[1],
            view_projection * clipped[2],
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
        if tri_ndc.iter().any(|v| !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite()) {
            continue;
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

        for y in bounding_box.1..bounding_box.3 {
            for x in bounding_box.0..bounding_box.2 {
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

                    // use w-buffering instead of z-buffering for better perspective accuracy
                    // linearly interpolate 1/w, then invert to get w for depth testing
                    let inv_w_interp = interp_linear_scalar(
                        (1.0 / clip_w.0, 1.0 / clip_w.1, 1.0 / clip_w.2), 
                        bary,
                    );
                    // use the reciprocal as depth - smaller values = closer
                    let depth = 1.0 / inv_w_interp;
                    // color gradient (perspective-correct interpolation)
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

    let view_projection = create_view_projection_matrix(buffer, camera);
    view_projection * point
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

pub fn create_view_projection_matrix(buffer: &Buffer, camera: &Camera) -> Matrix4<f32> {
    let aspect_ratio = buffer.width as f32 / buffer.height as f32;
    let aspect_scale_matrix: Matrix4<f32> = Matrix4::new(
        1.0 / aspect_ratio, 0.0, 0.0, 0.0,
        0.0, 1.0,      0.0, 0.0,
        0.0, 0.0,      1.0, 0.0,
        0.0, 0.0,      0.0, 1.0,
    );
    aspect_scale_matrix * camera.projection_matrix
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

fn compute_clamped_bbox(tri: &[Vector4<f32>; 3], screen_width: f32, screen_height: f32) -> (usize, usize, usize, usize) {
    let mut xmin = (tri[0].x.min(tri[1].x).min(tri[2].x)).floor();
    let mut xmax = (tri[0].x.max(tri[1].x).max(tri[2].x)).ceil();
    let mut ymin = (tri[0].y.min(tri[1].y).min(tri[2].y)).floor();
    let mut ymax = (tri[0].y.max(tri[1].y).max(tri[2].y)).ceil();
    xmin = xmin.max(0.0);
    ymin = ymin.max(0.0);
    xmax = xmax.min(screen_width - 1.0);
    ymax = ymax.min(screen_height - 1.0);

    (xmin as usize, ymin as usize, xmax as usize, ymax as usize)
}

fn _interp_perspective_scalar(a: (f32, f32, f32), w_clip: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
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

fn interp_linear_scalar(a: (f32, f32, f32), bary: (f32, f32, f32)) -> f32 {
    // performs linear interpolation of a scalar attribute using barycentric coordinates
    let (b0, b1, b2) = bary;
    a.0 * b0 + a.1 * b1 + a.2 * b2
}

fn interp_perspective_color(colors: [Color; 3], w_clip: (f32, f32, f32), bary: (f32, f32, f32)) -> Color {
    // performs perspective-correct interpolation of a color
    use glam::Vec4;

    // use glam for SIMD
    let c0 = Vec4::new(colors[0].0 as f32, colors[0].1 as f32, colors[0].2 as f32, colors[0].3 as f32);
    let c1 = Vec4::new(colors[1].0 as f32, colors[1].1 as f32, colors[1].2 as f32, colors[1].3 as f32);
    let c2 = Vec4::new(colors[2].0 as f32, colors[2].1 as f32, colors[2].2 as f32, colors[2].3 as f32);

    let (b0, b1, b2) = bary;
    let (w0_clip, w1_clip, w2_clip) = w_clip;

    // calculate perspective-correct vertex attributes (Color / w)
    let c0_persp = c0 / w0_clip;
    let c1_persp = c1 / w1_clip;
    let c2_persp = c2 / w2_clip;

    // interpolate the (Color / w) attributes using barycentric coordinates
    let num_vec = c0_persp.mul_add(Vec4::splat(b0), c1_persp.mul_add(Vec4::splat(b1), c2_persp * b2));

    // calculate the interpolated 1/w for the fragment
    let interpolated_inv_w = (1.0 / w0_clip) * b0 + (1.0 / w1_clip) * b1 + (1.0 / w2_clip) * b2;

    if interpolated_inv_w.abs() < EPSILON {
        return (0, 0, 0, 0);
    }

    // final perspective-correct color by dividing by interpolated 1/w
    let final_color_vec = num_vec / interpolated_inv_w;
    
    // lamp the result to the valid color range [0, 255] and convert back to u8 tuple
    let clamped = final_color_vec.clamp(Vec4::ZERO, Vec4::splat(255.0)).round();
    (clamped.x as u8, clamped.y as u8, clamped.z as u8, clamped.w as u8)
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
