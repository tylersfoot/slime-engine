use cgmath::{Vector3, Quaternion, Matrix4, Zero, One};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale: Vector3<f32>,
}

impl Transform {
    pub fn new() -> Self {
        Self {
            position: Vector3::zero(), // at origin
            rotation: Quaternion::one(), // identity quaternion (no rotation)
            scale: Vector3::new(1.0, 1.0, 1.0), // 100% scale
        }
    }

    // calculates the 4x4 matrix used by the GPU to place vertices in the world
    pub fn calc_matrix(&self) -> Matrix4<f32> {
        // in CG, matrix mult. is applied right -> left
        // order: scale -> rotate -> translate
        Matrix4::from_translation(self.position)
            * Matrix4::from(self.rotation)
            * Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z)
    }

    // helper builders, for example:
    // Transform::new().with_position([2.0, 0.5, 2.0]).with_scale([5.0, 5.0, 5.0])
    pub fn with_position(mut self, position: impl Into<Vector3<f32>>) -> Self {
        self.position = position.into();
        self
    }
    pub fn with_rotation(mut self, rotation: impl Into<Quaternion<f32>>) -> Self {
        self.rotation = rotation.into();
        self
    }
    pub fn with_scale(mut self, scale: impl Into<Vector3<f32>>) -> Self {
        self.scale = scale.into();
        self
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::new()
    }
}