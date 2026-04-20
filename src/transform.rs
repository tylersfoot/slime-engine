use cgmath::{Vector2, Vector3, Quaternion, Matrix4, Zero, One, Rad};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform3D {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale: Vector3<f32>,
}

impl Transform3D {
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
    // Transform3D::new().with_position([2.0, 0.5, 2.0]).with_scale([5.0, 5.0, 5.0])
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

impl Default for Transform3D {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    pub position: Vector2<f32>,
    pub rotation: f32, // in degrees
    pub scale: Vector2<f32>,
}

impl Transform2D {
    pub fn new() -> Self {
        Self {
            position: Vector2::zero(),
            rotation: 0.0,
            scale: Vector2::new(1.0, 1.0),
        }
    }

    // calculates the 4x4 matrix used by the GPU to place vertices in the world
    pub fn calc_matrix(&self) -> Matrix4<f32> {
        // z stays at 0, z_index is used for draw order instead
        Matrix4::from_translation(Vector3::new(self.position.x, self.position.y, 0.0))
            // 2D rotation is around z axis
            * Matrix4::from_angle_z(Rad(self.rotation))
            // z scale stays at 1
            * Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, 1.0)
    }

    pub fn with_position(mut self, position: impl Into<Vector2<f32>>) -> Self {
        self.position = position.into();
        self
    }
    pub fn with_rotation(mut self, rotation: f32) -> Self {
        self.rotation = rotation;
        self
    }
    pub fn with_scale(mut self, scale: impl Into<Vector2<f32>>) -> Self {
        self.scale = scale.into();
        self
    }
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::new()
    }
}