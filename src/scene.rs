use crate::core::GraphicsContext;
use crate::render::Renderer;
use crate::camera::{Camera, CameraUniform, CameraController, Projection};
use crate::model::{ModelAsset, Instance, InstanceRaw, Material, MaterialTextures, MaterialUniforms, Mesh, Model, ModelVertex};
use crate::texture::{Texture};
use crate::resources;
use crate::transform::Transform;
use crate::node::Node;
use cgmath::{Matrix3, prelude::*};
use wgpu::util::DeviceExt;


// scene represents "the what"
pub struct Scene {
    pub nodes: Vec<Node>,
    pub assets: Vec<ModelAsset>,

    pub camera: Camera,
    pub projection: Projection,
    pub camera_controller: CameraController,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,

    pub light_uniform: LightUniform,
    pub light_buffer: wgpu::Buffer,
    pub light_bind_group: wgpu::BindGroup,
}

impl Scene {
    pub async fn new(gfx: &GraphicsContext<'_>, renderer: &Renderer) -> Self {
        let nodes = Vec::new();
        let assets = Vec::new();

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

        // add lighting
        let light_uniform = LightUniform::new(
            [2.0, 3.0, 2.0],
            [1.0, 0.96, 0.89],
        );

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
            nodes,
            assets,
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            light_uniform,
            light_buffer,
            light_bind_group,
        }
    }

    // loads .obj file from disk, creates buffers, returns ID handle
    pub async fn load_model(
        &mut self,
        file_path: &str,
        gfx: &GraphicsContext<'_>,
        renderer: &Renderer
    ) -> usize {
        // load raw mesh data
        let model = resources::load_model(
            file_path,
            &gfx.device,
            &gfx.queue,
            &renderer.texture_bind_group_layout
        ).await.unwrap();

        // wrap in asset container, creating empty instance buffer
        let asset = ModelAsset::new(&gfx.device, model);

        let id = self.assets.len();
        self.assets.push(asset);
        id
    }

    // spawns a node into the world, returns ID handle
    pub fn spawn_node(&mut self, model_id: Option<usize>, transform: Transform) -> usize {
        let node_id = self.nodes.len();
        let node = Node::new(model_id).with_transform(transform);
        self.nodes.push(node);
        node_id
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


        // process nodes and instancing
        for asset in &mut self.assets {
            asset.instance_count = 0;
        }

        // create temporary buckets to hold the raw GPU data for each model
        let mut instance_data = vec![Vec::new(); self.assets.len()];

        // loop through the node tree and build transforms
        for node in &mut self.nodes {
            // calculate absolute world position
            // TODO: add parent/child math
            node.global_transform = node.transform.calc_matrix();

            // if the node has a model, convert it to bytes and bucket it
            if let Some(model_id) = node.model_id {
                let raw = InstanceRaw {
                    model: node.global_transform.into(),
                    normal: Matrix3::from(node.transform.rotation).into(),
                };
                instance_data[model_id].push(raw);
                self.assets[model_id].instance_count += 1;
            }
        }

        // send the buckets to the wgpu buffers
        for (i, asset) in self.assets.iter_mut().enumerate() {
            if asset.instance_count > 0 {
                // safeguard for now
                assert!(
                    asset.instance_count <= asset.capacity,
                    "Exceeded instance capacity for model {}, max is {}", i, asset.capacity
                );

                queue.write_buffer(
                    &asset.instance_buffer,
                    0,
                    bytemuck::cast_slice(&instance_data[i]));
            }
        }
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

impl LightUniform {
    pub fn new(position: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            position,
            _padding: 0,
            color,
            _padding2: 0,
        }
    }
}