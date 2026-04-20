use crate::transform::Transform3D;
use cgmath::{Matrix4};
use crate::scene::{Node3DId, ModelId};

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
    pub parent: Option<Node3DId>,
    // indices of child nodes
    pub children: Vec<Node3DId>,
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
