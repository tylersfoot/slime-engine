use crate::model::{
    Model,
    ModelVertex,
    Material,
    MaterialTextures,
    MaterialUniforms,
};

#[derive(Debug)]
pub enum Primitive {
    Quad,
    Cube,
}

pub struct Primitives;

impl Primitives {
    pub fn quad(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        color: [f32; 3],
    ) -> Model {
        // flat plane facing up
        let vertices = [
            ModelVertex { // top left
                position: [-0.5, 0.0, -0.5],
                tex_coords: [0.0, 1.0],
                normal: [0.0, 1.0, 0.0]
            },
            ModelVertex { // top right
                position: [ 0.5, 0.0, -0.5],
                tex_coords: [1.0, 1.0],
                normal: [0.0, 1.0, 0.0]
            },
            ModelVertex { // bottom right
                position: [ 0.5, 0.0,  0.5],
                tex_coords: [1.0, 0.0],
                normal: [0.0, 1.0, 0.0]
            },
            ModelVertex { // bottom left
                position: [-0.5, 0.0,  0.5],
                tex_coords: [0.0, 0.0],
                normal: [0.0, 1.0, 0.0]
            },
        ];

        // two tris
        let indices = [
            0, 1, 2,
            0, 2, 3
        ];

        // default texture/material, with custom color
        let textures = MaterialTextures::default(device, queue).unwrap();
        let mut uniforms = MaterialUniforms {
            diffuse_color: color,
            ..Default::default()
        };

        let material = Material::new(
            device,
            "quad_material",
            textures,
            layout,
            uniforms
        );

        Model::from_raw(device, "quad", &vertices, &indices, material)
    }

    pub fn cube(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        color: [f32; 3],
    ) -> Model {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let face_data = [
            // front face (+Z)
            ([ 0.0,  0.0,  1.0], [[-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]]),
            // back face (-Z)
            ([ 0.0,  0.0, -1.0], [[ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5]]),
            // right face (+X)
            ([ 1.0,  0.0,  0.0], [[ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5]]),
            // left face (-X)
            ([-1.0,  0.0,  0.0], [[-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5]]),
            // top face (+Y)
            ([ 0.0,  1.0,  0.0], [[-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5]]),
            // bottom face (-Y)
            ([ 0.0, -1.0,  0.0], [[-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5]]),
        ];

        let mut offset = 0;
        for (normal, corners) in face_data {
            let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
            
            for i in 0..4 {
                vertices.push(ModelVertex {
                    position: corners[i],
                    tex_coords: uvs[i],
                    normal,
                });
            }

            indices.extend_from_slice(&[offset, offset + 1, offset + 2, offset, offset + 2, offset + 3]);
            offset += 4;
        }

        let textures = MaterialTextures::default(device, queue).unwrap();
        let mut uniforms = MaterialUniforms {
            diffuse_color: color,
            ..Default::default()
        };
        uniforms.diffuse_color = color;

        let material = Material::new(device, "cube_material", textures, layout, uniforms);

        Model::from_raw(device, "cube", &vertices, &indices, material)
    }
}