use crate::core::GraphicsContext;
use crate::render::Renderer;
use crate::camera::{Camera, CameraUniform, CameraController, Projection};
use crate::model::{Instance, InstanceRaw, Material, MaterialTextures, MaterialUniforms, Mesh, Model, ModelVertex};
use crate::texture::{Texture};
use crate::resources;
use cgmath::prelude::*;
use wgpu::util::DeviceExt;


// scene represents "the what"
pub struct Scene {
    pub camera: Camera,
    pub projection: Projection,
    pub camera_controller: CameraController,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,

    pub light_uniform: LightUniform,
    pub light_buffer: wgpu::Buffer,
    pub light_bind_group: wgpu::BindGroup,

    pub obj_model: Model,
    pub obj_instances: Vec<Instance>,
    pub obj_instance_buffer: wgpu::Buffer,
    pub cube_model: Model,
    pub cube_instance_buffer: wgpu::Buffer,
    pub ground_model: Model,
    pub ground_instance_buffer: wgpu::Buffer,
}

impl Scene {
    pub async fn new(gfx: &GraphicsContext<'_>, renderer: &Renderer) -> Self {
       let camera = Camera::new(
            (0.0, 5.0, 10.0),
            cgmath::Deg(-90.0),
            cgmath::Deg(-20.0)
        );
        let projection = Projection::new(
            gfx.config.width,
            gfx.config.height,
            cgmath::Deg(45.0),
            0.02,
            100.0
        );
        let camera_controller = CameraController::new(2.0, 0.4);
 
        // create camera uniform so we can use our camera data in shaders
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);
            
        let camera_buffer = gfx.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("camera_buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group = gfx.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &renderer.camera_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }],
                label: Some("camera_bind_group"),
            }
        );
        
        // instancing
        const SPACE_BETWEEN: f32 = 2.0;
        const NUM_INSTANCES_PER_ROW: u32 = 1;
        let obj_instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = cgmath::Vector3 { x, y: 0.0, z };

                let rotation = if position.is_zero() {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };

                Instance {
                    position, rotation,
                }
            })
        }).collect::<Vec<_>>();
 
        let obj_instance_data = obj_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let obj_instance_buffer = gfx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instance_buffer"),
            contents: bytemuck::cast_slice(&obj_instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let obj_model = resources::load_model(
            "cannon/cannon.obj",
            &gfx.device,
            &gfx.queue,
            &renderer.texture_bind_group_layout
        ).await.unwrap();

        let cube_model = resources::load_model(
            "cube2/cube.obj",
            &gfx.device,
            &gfx.queue,
            &renderer.texture_bind_group_layout
        ).await.unwrap();
        let cube_instance_buffer = gfx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube-instance-buffer"),
            contents: bytemuck::cast_slice(&[
                    Instance {
                        position: [2.0, 0.5, 2.0].into(),
                        rotation: cgmath::Quaternion::one(),
                    }.to_raw()
                ]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let ground_vertices = [
            ModelVertex { position: [-500.0, -0.5, -500.0], tex_coords: [0.0, 0.0], normal: [0.0, 1.0, 0.0] },
            ModelVertex { position: [ 500.0, -0.5, -500.0], tex_coords: [1000.0, 0.0], normal: [0.0, 1.0, 0.0] },
            ModelVertex { position: [ 500.0, -0.5,  500.0], tex_coords: [1000.0, 1000.0], normal: [0.0, 1.0, 0.0] },
            ModelVertex { position: [-500.0, -0.5,  500.0], tex_coords: [0.0, 1000.0], normal: [0.0, 1.0, 0.0] },
        ];
        let ground_indices: [u32; 6] = [0, 2, 1, 0, 3, 2];

        let ground_v_buf = gfx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ground_vertex_buffer"),
            contents: bytemuck::cast_slice(&ground_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ground_i_buf = gfx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ground_index_buffer"),
            contents: bytemuck::cast_slice(&ground_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let ground_diffuse_texture = Texture::from_color(
            &gfx.device,
            &gfx.queue,
            [50, 50, 50, 255],
            "ground_diffuse",
            false,
        ).unwrap();

        let mut ground_material_textures = MaterialTextures::from_diffuse(
            &gfx.device,
            &gfx.queue,
            ground_diffuse_texture,
        ).unwrap();

        ground_material_textures.roughness = Texture::from_color(&gfx.device, &gfx.queue, [0, 0, 0, 255], "default_roughness", false).unwrap();

        let mut ground_material_uniforms = MaterialUniforms::default();
        ground_material_uniforms.specular_exponent = 100.0;
        ground_material_uniforms.specular_color = [1.0, 1.0, 1.0];
        let mut ground_material = Material::new(
            &gfx.device,
            "ground-material",
            ground_material_textures,
            &renderer.texture_bind_group_layout,
            ground_material_uniforms,
        );

        let ground_model = Model {
            meshes: vec![Mesh {
                name: "ground".into(),
                vertex_buffer: ground_v_buf,
                index_buffer: ground_i_buf,
                num_elements: 6,
                material: 0, 
            }],
            materials: vec![ground_material],
        };

        let ground_instance = Instance {
            position: [0.0, -0.1, 0.0].into(),
            rotation: cgmath::Quaternion::one(),
        };
        let ground_instance_buffer = gfx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ground-instance-buffer"),
            contents: bytemuck::cast_slice(&[
                    Instance {
                        position: [0.0, -0.1, 0.0].into(),
                        rotation: cgmath::Quaternion::one(),
                    }.to_raw()
                ]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // add lighting
        let light_uniform = LightUniform {
            position: [2.0, 3.0, 2.0],
            _padding: 0,
            color: [1.0, 0.96, 0.89],
            _padding2: 0
        };

        // use COPY_DST to update light's position
        let light_buffer = gfx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("light_buffer"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
 
        let light_bind_group = gfx.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &renderer.light_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.as_entire_binding(),
                }],
                label: Some("light_bind_group"),
            }
        );

        Self {
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            light_uniform,
            light_buffer,
            light_bind_group,
            obj_model,
            obj_instances,
            obj_instance_buffer,
            cube_model,
            cube_instance_buffer,
            ground_model,
            ground_instance_buffer,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.projection.resize(width, height);
    }

    pub fn update(&mut self, dt: std::time::Duration, queue: &wgpu::Queue) {
        // update camera
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.update_view_proj(&self.camera, &self.projection);
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // update light
        let old_position: cgmath::Vector3<_> = self.light_uniform.position.into();
        self.light_uniform.position = (
            cgmath::Quaternion::from_axis_angle(
                (0.0, 1.0, 0.0).into(),
                cgmath::Deg(60.0 * dt.as_secs_f32())
            ) * old_position
        ).into();

        queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// represents a colored point in space
pub struct LightUniform {
    pub position: [f32; 3],
    _padding: u32,
    pub color: [f32; 3],
    _padding2: u32,
}