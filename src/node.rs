use crate::transform::Transform;
use cgmath::{Matrix4};

pub struct Node {
    // local transform relative to parent
    pub transform: Transform,
    // world transform matrix after walking up node tree
    pub global_transform: Matrix4<f32>,
    // optional model ID to draw
    pub model_id: Option<usize>,
    // optional index of the parent node
    pub parent: Option<usize>,
    // indices of child nodes
    pub children: Vec<usize>,
}

impl Node {
    pub fn new(model_id: Option<usize>) -> Self {
        Self {
            transform: Transform::new(),
            global_transform: Matrix4::from_scale(1.0),
            model_id,
            parent: None,
            children: Vec::new(),
        }
    }

    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }
}