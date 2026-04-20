use std::ops::Range;
use wgpu::util::DeviceExt;
use crate::texture;

pub trait Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// the raw data of the instance to pass to the GPU
pub struct InstanceRaw {
    pub model: [[f32; 4]; 4],
    pub normal: [[f32; 3]; 3],
    pub color: [f32; 4],
}
 
impl InstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // we use a step mode of Instance, where our shaders will only change 
            // to use the next instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // a mat4 takes up 4 vertex slots as it is technically 4 Vec4's
                // we need to define a slot for each vec4 and reassemble in the shader
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // normal: 3 Vec3's
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color: Vec4
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 25]>() as wgpu::BufferAddress,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// bytemuck and repr(C) ensure the data is ordered correctly for the GPU
// repr(C) fixes ordering: do not reorder fields in memory, and use same padding as C compiler
// bytemuck fixes types: rust can't convert &[Vertex] to &[u8]
// Pod (plain old data) promises that the struct is just simple bytes, and doesn't change size
// Zeroable promises that a memory pattern of all zeroes is valid
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
// the raw vertex bytes to send to the GPU
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

impl Vertex for ModelVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        // creates a description to give to the render pipeline, which tells it
        // how to interpret the raw byte stream in the vertex_buffer
        wgpu::VertexBufferLayout {
            // defines how many bytes to jump forward in the buffer to get from the
            // start of one vertex to the start of the next one; 
            // in this case, it's just the size of Vertex struct
            array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            // tells the GPU when to advance to the next vertex in the buffer
            // in this case, for each vertex read the next block of data
            step_mode: wgpu::VertexStepMode::Vertex,
            // describes each field in the struct
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0, // starts at byte 0
                    shader_location: 0, // will be available in the shader at @location(0)
                    format: wgpu::VertexFormat::Float32x3, // three 32-bit floats
                },
                // texture coordinates
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniforms {
    // name ~ keyword - type/range - default
    pub ambient_color: [f32; 3],  // Ka ~ 3*[0.0, 1.0] ~ [1.0, 1.0, 1.0]
    pub dissolve: f32,            // d ~ [0.0, 1.0] ~ 1.0 (Tr = 1.0 - d)
    pub diffuse_color: [f32; 3],  // Kd ~ 3*[0.0, 1.0] ~ [1.0, 1.0, 1.0]
    pub specular_exponent: f32,   // Ns ~ [0.0, 1000.0] ~ 10.0 (Pr = roughness)
    pub specular_color: [f32; 3], // Ks ~ 3*[0.0, 1.0] ~ [0.0, 0.0, 0.0]
    pub _padding1: f32,
    pub emissive_color: [f32; 3], // Ke ~ 3*[0.0, 1.0] ~ [0.0, 0.0, 0.0]
    pub _padding2: f32,
}

impl MaterialUniforms {
    pub fn new(
        ambient_color: [f32; 3],
        diffuse_color: [f32; 3],
        specular_color: [f32; 3],
        emissive_color: [f32; 3],
        dissolve: f32,
        specular_exponent: f32,
    ) -> Self {
        Self {
            ambient_color,
            diffuse_color,
            specular_color,
            emissive_color,
            dissolve,
            specular_exponent,
            _padding1: 0.0,
            _padding2: 0.0,
        }
    }
}

impl Default for MaterialUniforms {
    fn default() -> Self {
        // init with defaults, can edit individual values after
        Self {
            ambient_color: [1.0, 1.0, 1.0],
            diffuse_color: [1.0, 1.0, 1.0],
            specular_color: [0.5, 0.5, 0.5],
            emissive_color: [0.0, 0.0, 0.0],
            dissolve: 1.0,
            specular_exponent: 50.0,
            _padding1: 0.0,
            _padding2: 0.0,
        }
    }
}

pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub uniforms: MaterialUniforms,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl Material {
    pub fn new(
        device: &wgpu::Device,
        name: &str,
        textures: MaterialTextures,
        layout: &wgpu::BindGroupLayout,
        uniforms: MaterialUniforms,
    ) -> Self {
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?}-uniform-buffer", name)),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&textures.diffuse.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&textures.diffuse.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                }
            ],
            label: Some(name),
        });

        Self {
            name: String::from(name),
            diffuse_texture: textures.diffuse,
            uniforms,
            uniform_buffer,
            bind_group,
        }
    }

    pub fn update_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }
}

pub struct MaterialTextures {
    pub diffuse: texture::Texture, // map_Kd
}

impl MaterialTextures {
    pub fn new(
        diffuse: texture::Texture,
    ) -> Self {
        Self {
            diffuse,
        }
    }

    // generates default/fallback 1x1 textures
    pub fn default(
        device: &wgpu::Device,
        queue: &wgpu::Queue
    ) -> anyhow::Result<Self> {
        // white - identity multiplier for standard maps
        let diffuse = texture::Texture::from_color(
            device,
            queue,
            [255, 255, 255, 255],
            "default_diffuse",
            false
        )?;

        Ok(Self {
            diffuse,
        })
    }

    // creates a MaterialTextures using a diffuse map, with fallbacks for other textures
    pub fn from_diffuse(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        diffuse: texture::Texture,
    ) -> anyhow::Result<Self> {
        let mut textures = MaterialTextures::default(device, queue)?;
        textures.diffuse = diffuse;
        Ok(textures)
    }
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl Model {
    // builds a model from raw data
    pub fn from_raw(
        device: &wgpu::Device,
        name: &str,
        vertices: &[ModelVertex],
        indices: &[i32],
        material: Material,
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{name:?}_vertex_buffer")),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{name:?}_index_buffer")),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        log::info!("Created raw model: {} | {} vertices", name, vertices.len());
        let mesh = Mesh {
            name: name.to_string(),
            vertex_buffer,
            index_buffer,
            num_elements: indices.len() as u32,
            material: 0, // first (and only) material in array
        };

        Self {
            meshes: vec![mesh],
            materials: vec![material],
        }
    }
}

// represents a loaded 3D asset + buffer to draw instances
pub struct ModelAsset {
    pub model: Model,
    // the buffer holding the InstanceRaw data for the model
    pub instance_buffer: wgpu::Buffer,
    // how many nodes are currently using this model
    pub instance_count: u32,
    // how many instances the buffer can currently hold
    pub capacity: u32,
}

impl ModelAsset {
    pub fn new(device: &wgpu::Device, model: Model) -> Self {
        // allocate for 10 instances to start
        let capacity = 10;
        let instance_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("model_instance_buffer"),
                size: (std::mem::size_of::<InstanceRaw>() * capacity) as wgpu::BufferAddress,
                // COPY_DST so we can write to the buffer every frame
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        Self {
            model,
            instance_buffer,
            instance_count: 0,
            capacity: capacity as u32,
        }
    }

    pub fn resize_buffer_if_needed(&mut self, device: &wgpu::Device, required_capacity: u32) {
        // checks if the model instance buffer needs a higher instance capacity
        if required_capacity > self.capacity {
            // double capacity
            let mut new_capacity = self.capacity * 2;
            
            // if still not enough, just set to the required capacity
            if new_capacity < required_capacity {
                new_capacity = required_capacity;
            }

            // create the new buffer
            self.instance_buffer = device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("model_instance_buffer"),
                    size: (std::mem::size_of::<InstanceRaw>() as u32 * new_capacity) as wgpu::BufferAddress,
                    // COPY_DST so we can write to the buffer every frame
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            );

            self.capacity = new_capacity;
        }
    }
}

pub trait DrawShadow<'a> {
    fn draw_shadow_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        instances: Range<u32>,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_shadow_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        light_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawShadow<'b> for wgpu::RenderPass<'a> where 'b: 'a {
    fn draw_shadow_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        instances: Range<u32>,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_shadow_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            self.draw_shadow_mesh_instanced(mesh, instances.clone(), light_bind_group);
        }
    }
}

pub trait DrawModel<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
        shadow_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
        shadow_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
        shadow_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
        shadow_bind_group: &'a wgpu::BindGroup,
    );

}

impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
        shadow_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, camera_bind_group, light_bind_group, shadow_bind_group);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
        shadow_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, shadow_bind_group, &[]);
        self.set_bind_group(1, &material.bind_group, &[]);
        self.set_bind_group(2, camera_bind_group, &[]);
        self.set_bind_group(3, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }
    
    fn draw_model(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
        shadow_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_model_instanced(model, 0..1, camera_bind_group, light_bind_group, shadow_bind_group);
    }

    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
        shadow_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone(), camera_bind_group, light_bind_group, shadow_bind_group);
        }
    }
}

pub trait DrawLight<'a> {
    fn draw_light_mesh(
        &mut self,
        mesh: &'a Mesh,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_light_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_light_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );

}

impl<'a, 'b> DrawLight<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_light_mesh(
        &mut self,
        mesh: &'b Mesh,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_light_mesh_instanced(mesh, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }
    
    fn draw_light_model(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup
    ) {
        self.draw_light_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_light_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            self.draw_light_mesh_instanced(mesh, instances.clone(), camera_bind_group, light_bind_group);
        }
    }
}