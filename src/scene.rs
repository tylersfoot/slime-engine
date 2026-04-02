use crate::core::GraphicsContext;
use crate::render::Renderer;
use crate::camera::{Camera, CameraId, CameraUniform, CameraController, Projection};
use crate::model::{ModelAsset, Instance, InstanceRaw, Material, MaterialTextures, MaterialUniforms, Mesh, Model, ModelVertex};
use crate::texture::{Texture};
use crate::resources;
use crate::transform::Transform;
use crate::node::{Node, NodeId};
use cgmath::{Matrix3, prelude::*, Point3};
use wgpu::util::DeviceExt;
use slotmap::SlotMap;


// scene represents "the what"
pub struct Scene {
    pub nodes: SlotMap<NodeId, Node>,
    pub assets: Vec<ModelAsset>,

    pub cameras: SlotMap<CameraId, Camera>,
    pub active_camera: Option<CameraId>,
    pub camera_controller: CameraController,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,

    pub light_uniform: LightUniform,
    pub light_buffer: wgpu::Buffer,
    pub light_bind_group: wgpu::BindGroup,

    pub window_width: u32,
    pub window_height: u32,
}

impl Scene {
    pub async fn new(gfx: &GraphicsContext<'_>, renderer: &Renderer) -> Self {
        let nodes = SlotMap::with_key();
        let assets = Vec::new();
        let cameras = SlotMap::with_key();
        let active_camera = None;

        let camera_controller = CameraController::new(2.0, 0.4);
 
        // create camera uniform so we can use our camera data in shaders
        let mut camera_uniform = CameraUniform::new();
            
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
            cameras,
            active_camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            light_uniform,
            light_buffer,
            light_bind_group,
            window_width: gfx.config.width,
            window_height: gfx.config.height,
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
    pub fn spawn_node(&mut self, model_id: Option<usize>, transform: Transform) -> NodeId {
        let node_id = self.nodes.len();
        let node = Node::new(model_id).with_transform(transform);
        self.nodes.insert(node)
    }

    pub fn spawn_camera<V: Into<Point3<f32>>>(&mut self, position: V, yaw_deg: f32, pitch_deg: f32) -> CameraId {
        let camera = Camera::new(
            position,
            cgmath::Deg(yaw_deg),
            cgmath::Deg(pitch_deg),
            self.window_width,
            self.window_height
        );
        let id = self.cameras.insert(camera);

        // if this is the first camera, set it to be active
        if self.active_camera.is_none() {
            self.active_camera = Some(id);
        }
        
        id
    }

    pub fn set_active_camera(&mut self, camera_id: CameraId) {
        if self.cameras.contains_key(camera_id) {
            self.active_camera = Some(camera_id);
        } else {
            log::warn!("Tried to set active camera to invalid CameraId: {:?}", camera_id);
        }
    }

    pub fn get_global_transform(&self, node_id: NodeId) -> cgmath::Matrix4<f32> {
        // recursively calculates the world transform 
        // of a node by walking up the parent chain
        let mut transform = cgmath::Matrix4::identity();

        if let Some(node) = self.nodes.get(node_id) {
            // get this node's local matrix
            transform = node.transform.calc_matrix();
            let mut current_parent = node.parent;

            // loop up the heirarchy, multiplying transforms
            while let Some(parent_id) = current_parent {
                if let Some(parent_node) = self.nodes.get(parent_id) {
                    transform = parent_node.transform.calc_matrix()  * transform;
                    current_parent = parent_node.parent;
                } else {
                    break; // parent not found
                }
            }
        }
        transform
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.window_width = width;
        self.window_height = height;

        for (_, camera) in self.cameras.iter_mut() {
            camera.projection.resize(width, height);
        }
    }

    pub fn update(&mut self, dt: std::time::Duration, device: &wgpu::Device, queue: &wgpu::Queue) {
        // update camera
        if let Some(active_id) = self.active_camera
            && let Some(camera) = self.cameras.get_mut(active_id) {
                self.camera_controller.update_camera(camera, dt);
                self.camera_uniform.update_view_proj(camera, &camera.projection);
                
                queue.write_buffer(
                    &self.camera_buffer,
                    0,
                    bytemuck::cast_slice(&[self.camera_uniform]),
                );
        };

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

        // calculate all global transforms first
        let mut global_transforms = Vec::with_capacity(self.nodes.len());
        for (id, _) in self.nodes.iter() {
            global_transforms.push((id, self.get_global_transform(id)));
        }

        // apply global transforms to nodes
        for (id, global_matrix) in global_transforms {
            let node = self.nodes.get_mut(id).unwrap();
            node.global_transform = global_matrix;

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
                // check if we need to allocate for more instances
                asset.resize_buffer_if_needed(device, asset.instance_count);

                queue.write_buffer(
                    &asset.instance_buffer,
                    0,
                    bytemuck::cast_slice(&instance_data[i])
                );
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