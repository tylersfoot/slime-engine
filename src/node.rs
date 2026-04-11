use crate::transform::Transform;
use cgmath::{Matrix4};
use crate::scene::{NodeId, ModelId};

pub struct Node {
    // local transform relative to parent
    pub transform: Transform,
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

impl Node {
    pub fn new(model_id: Option<ModelId>) -> Self {
        Self {
            transform: Transform::new(),
            global_transform: Matrix4::from_scale(1.0),
            model_id,
            color: [1.0, 1.0, 1.0, 1.0],
            parent: None,
            children: Vec::new(),
        }
    }

    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }
}