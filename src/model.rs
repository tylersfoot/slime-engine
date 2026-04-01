use std::ops::Range;
use wgpu::util::DeviceExt;
use crate::texture;
use crate::transform::Transform;

pub trait Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

pub struct Instance {
    pub transform: Transform,
}
 
impl Instance {
    pub fn new(transform: Transform) -> Self {
        Self { transform }
    }
    pub fn to_raw(&self) -> InstanceRaw {
        let model_matrix = self.transform.calc_matrix();
        InstanceRaw {
            model: model_matrix.into(),
            normal: cgmath::Matrix3::from(self.transform.rotation).into(),  
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// the raw data of the instance to pass to the GPU
pub struct InstanceRaw {
    pub model: [[f32; 4]; 4],
    pub normal: [[f32; 3]; 3],
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
    // note: reordered for byte alignment
    // name ~ keyword - type/range - default
    // Main keywords
    pub ambient_color: [f32; 3],        // Ka ~ 3*[0.0, 1.0] ~ [1.0, 1.0, 1.0]
    pub dissolve: f32,                  // d ~ [0.0, 1.0] ~ 1.0 (Tr = 1.0 - d)
    pub diffuse_color: [f32; 3],        // Kd ~ 3*[0.0, 1.0] ~ [1.0, 1.0, 1.0]
    pub specular_exponent: f32,         // Ns ~ [0.0, 1000.0] ~ 10.0 (Pr = roughness)
    pub specular_color: [f32; 3],       // Ks ~ 3*[0.0, 1.0] ~ [0.0, 0.0, 0.0]
    pub optical_density: f32,           // Ni ~ [0.001, 10.0] ~ 1.0
    pub emissive_color: [f32; 3],       // Ke ~ 3*[0.0, 1.0] ~ [0.0, 0.0, 0.0]
    pub reflection_sharpness: f32,      // sharpness ~ [0.0, 1000.0] ~ 60.0
    pub transmission_filter: [f32; 3],  // Tf ~ 3*[0.0, 1.0] ~ [1.0, 1.0, 1.0]

    // PBR extensions
    // pub roughness: f32, // Pr ~ [0.0, 1.0] ~ 0.5
    pub metallic: f32, // Pm ~ [0.0, 1.0] ~ 0.0
    pub sheen: f32, // Ps ~ [0.0, 1.0] ~ 0.0
    pub clearcoat_thickness: f32, // Pc ~ [0.0, 1.0] ~ 0.0
    pub clearcoat_roughness: f32, // Pcr ~ [0.0, 1.0] ~ 0.0
    pub anisotropy: f32, // aniso ~ [0.0, 1.0] ~ 0.0
    pub anisotropy_rotation: f32, // anisor ~ [0.0, 1.0] ~ 0.0

    // Illumination state
    pub illumination_model: u32, // illum ~ [0, 10] ~ 2
    // 0. Color on and Ambient off
    // 1. Color on and Ambient on
    // 2. Highlight on
    // 3. Reflection on and Ray trace on
    // 4. Transparency: Glass on, Reflection: Ray trace on
    // 5. Reflection: Fresnel on and Ray trace on
    // 6. Transparency: Refraction on, Reflection: Fresnel off and Ray trace on
    // 7. Transparency: Refraction on, Reflection: Fresnel on and Ray trace on
    // 8. Reflection on and Ray trace off
    // 9. Transparency: Glass on, Reflection: Ray trace off
    // 10. Casts shadows onto invisible surfaces

    pub _padding: [f32; 2],
}

impl MaterialUniforms {
    pub fn new(
        ambient_color: [f32; 3],
        diffuse_color: [f32; 3],
        specular_color: [f32; 3],
        emissive_color: [f32; 3],
        transmission_filter: [f32; 3],
        dissolve: f32,
        specular_exponent: f32,
        optical_density: f32,
        reflection_sharpness: f32,
        metallic: f32,
        sheen: f32,
        clearcoat_thickness: f32,
        clearcoat_roughness: f32,
        anisotropy: f32,
        anisotropy_rotation: f32,
        illumination_model: u32,
    ) -> Self {
        Self {
            ambient_color,
            diffuse_color,
            specular_color,
            emissive_color,
            transmission_filter,
            dissolve,
            specular_exponent,
            optical_density,
            reflection_sharpness,
            metallic,
            sheen,
            clearcoat_thickness,
            clearcoat_roughness,
            anisotropy,
            anisotropy_rotation,
            illumination_model,
            _padding: [0.0; 2],
        }
    }
    
    pub fn default() -> Self {
        // init with defaults, can edit individual values after
        Self {
            ambient_color: [1.0, 1.0, 1.0],
            diffuse_color: [1.0, 1.0, 1.0],
            specular_color: [0.0, 0.0, 0.0],
            emissive_color: [0.0, 0.0, 0.0],
            transmission_filter: [1.0, 1.0, 1.0],
            dissolve: 1.0,
            specular_exponent: 10.0,
            optical_density: 1.0,
            reflection_sharpness: 60.0,
            metallic: 0.0,
            sheen: 0.0,
            clearcoat_thickness: 0.0,
            clearcoat_roughness: 0.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            illumination_model: 2,
            _padding: [0.0; 2],
        }
    }
}

pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub normal_texture: texture::Texture,
    pub specular_texture: texture::Texture,
    pub dissolve_texture: texture::Texture,
    pub ambient_texture: texture::Texture,
    pub roughness_texture: texture::Texture,
    pub metal_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

impl Material {
    pub fn new(
        device: &wgpu::Device,
        name: &str,
        textures: MaterialTextures,
        layout: &wgpu::BindGroupLayout,
        material_uniforms: MaterialUniforms,
    ) -> Self {
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?}-uniform-buffer", name)),
            contents: bytemuck::cast_slice(&[material_uniforms]),
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
                    resource: wgpu::BindingResource::TextureView(&textures.normal.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&textures.normal.sampler),
                },

                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&textures.specular.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&textures.specular.sampler),
                },

                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&textures.dissolve.view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&textures.dissolve.sampler),
                },

                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&textures.ambient.view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&textures.ambient.sampler),
                },

                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&textures.roughness.view),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::Sampler(&textures.roughness.sampler),
                },

                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(&textures.metal.view),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::Sampler(&textures.metal.sampler),
                },

                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: uniform_buffer.as_entire_binding(),
                }
            ],
            label: Some(name),
        });

        Self {
            name: String::from(name),
            diffuse_texture: textures.diffuse,
            normal_texture: textures.normal,
            specular_texture: textures.specular,
            dissolve_texture: textures.dissolve,
            ambient_texture: textures.ambient,
            roughness_texture: textures.roughness,
            metal_texture: textures.metal,
            bind_group,
        }
    }
}

pub struct MaterialTextures {
    pub diffuse: texture::Texture, // map_Kd
    pub normal: texture::Texture, // norm/bump/map_bump
    pub specular: texture::Texture, // map_Ks
    pub dissolve: texture::Texture, // map_d
    pub ambient: texture::Texture, // map_Ka
    pub roughness: texture::Texture, // map_Pr, 1 - map_Ns
    pub metal: texture::Texture, // map_Pm
    // pub sheen: texture::Texture, // map_Ps
    // pub emissive: texture::Texture, // map_Ke
    // pub displacement: texture::Texture, // disp
    // pub reflection: texture::Texture, // refl
    // pub decal: texture::Texture, // decal
}

impl MaterialTextures {
    pub fn new(
        diffuse: texture::Texture,
        normal: texture::Texture,
        specular: texture::Texture,
        dissolve: texture::Texture,
        ambient: texture::Texture,
        roughness: texture::Texture,
        metal: texture::Texture,
    ) -> Self {
        Self {
            diffuse,
            normal,
            specular,
            dissolve,
            ambient,
            roughness,
            metal,
        }
    }

    /// generates default/fallback 1x1 textures
    pub fn default(
        device: &wgpu::Device,
        queue: &wgpu::Queue
    ) -> anyhow::Result<Self> {
        // white - identity multiplier for standard maps
        let diffuse = texture::Texture::from_color(device, queue, [255, 255, 255, 255], "default_diffuse", false)?;
        let specular = texture::Texture::from_color(device, queue, [255, 255, 255, 255], "default_specular", false)?;
        let dissolve = texture::Texture::from_color(device, queue, [255, 255, 255, 255], "default_dissolve", false)?;
        let ambient = texture::Texture::from_color(device, queue, [255, 255, 255, 255], "default_ambient", false)?;
        let roughness = texture::Texture::from_color(device, queue, [255, 255, 255, 255], "default_roughness", false)?;
        // flat normal points straight up in tangent space (0.5, 0.5, 1.0)
        let normal = texture::Texture::from_color(device, queue, [128, 128, 255, 255], "default_normal", true)?;
        // black - default to non-metallic
        let metal = texture::Texture::from_color(device, queue, [0, 0, 0, 255], "default_metal", false)?;

        Ok(Self {
            diffuse,
            normal,
            specular,
            dissolve,
            ambient,
            roughness,
            metal,
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
        // allocate for 100 instances to start
        // TODO: resize when count > capacity
        let capacity = 100;
        let instance_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("model_instancce_buffer"),
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
}

pub trait DrawModel<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
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
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, &material.bind_group, &[]);
        self.set_bind_group(1, camera_bind_group, &[]);
        self.set_bind_group(2, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }
    
    fn draw_model(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup
    ) {
        self.draw_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone(), camera_bind_group, light_bind_group);
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