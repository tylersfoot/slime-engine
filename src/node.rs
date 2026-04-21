use crate::transform::{Transform3D, Transform2D};
use cgmath::{Matrix4};
use crate::scene::{NodeId, ModelId, TextureId};

pub struct Node3D {
    // local transform relative to parent
    pub transform: Transform3D,
    // world transform matrix after walking up node tree
    pub global_transform: Matrix4<f32>,
    // optional model ID to draw
    pub model_id: Option<ModelId>,
    // color tint to this node's instance
    pub color: [f32; 4],
    // optional index of the parent node
    pub parent: Option<NodeId>,
    // indices of child nodes
    pub children: Vec<NodeId>,
}

impl Node3D {
    pub fn new(model_id: Option<ModelId>) -> Self {
        Self {
            transform: Transform3D::new(),
            global_transform: Matrix4::from_scale(1.0),
            model_id,
            color: [1.0, 1.0, 1.0, 1.0],
            parent: None,
            children: Vec::new(),
        }
    }

    pub fn with_transform(mut self, transform: Transform3D) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }
}

impl Default for Node3D {
    fn default() -> Self {
        Self {
            transform: Transform3D::new(),
            global_transform: Matrix4::from_scale(1.0),
            model_id: None,
            color: [1.0, 1.0, 1.0, 1.0],
            parent: None,
            children: Vec::new(),
        }
    }
}

pub struct Node2D {
    // local transform relative to parent
    pub transform: Transform2D,
    // world transform matrix after walking up node tree
    pub global_transform: Matrix4<f32>,
    // optional texture ID to draw
    pub texture_id: Option<TextureId>,
    // draw order
    pub z_index: i32,
    // color tint to this node's instance
    pub color: [f32; 4],
    // optional index of the parent node
    pub parent: Option<NodeId>,
    // indices of child nodes
    pub children: Vec<NodeId>,
}

impl Node2D {
    pub fn new() -> Self {
        Self {
            transform: Transform2D::new(),
            global_transform: Matrix4::from_scale(1.0),
            texture_id: None,
            z_index: 0,
            color: [1.0, 1.0, 1.0, 1.0],
            parent: None,
            children: Vec::new(),
        }
    }

    pub fn with_transform(mut self, transform: Transform2D) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    pub fn with_z_index(mut self, z_index: i32) -> Self {
        self.z_index = z_index;
        self
    }
}

impl Default for Node2D {
    fn default() -> Self {
        Self {
            transform: Transform2D::new(),
            global_transform: Matrix4::from_scale(1.0),
            texture_id: None,
            z_index: 0,
            color: [1.0, 1.0, 1.0, 1.0],
            parent: None,
            children: Vec::new(),
        }
    }
}