use crate::core::GraphicsContext;
use crate::render::Renderer;
use crate::camera::{Camera, CameraUniform, CameraController, Projection};
use crate::model::{ModelAsset, InstanceRaw, Material, MaterialTextures, MaterialUniforms, Mesh, Model, ModelVertex};
use crate::texture::{Texture};
use crate::resources;
use crate::transform::Transform;
use crate::node::Node;
use crate::primitives::{Primitives, Primitive};
use cgmath::{Matrix3, prelude::*, Point3};
use wgpu::util::DeviceExt;
use slotmap::{SlotMap, new_key_type};
use std::collections::HashMap;

// generates unique ID keys
new_key_type! {
    pub struct CameraId;
    pub struct ModelId;
    pub struct NodeId;
}

// scene represents "the what"
pub struct Scene {
    pub nodes: SlotMap<NodeId, Node>,
    pub assets: SlotMap<ModelId, ModelAsset>,

    pub cameras: SlotMap<CameraId, Camera>,
    pub active_camera: Option<CameraId>,
    pub camera_controller: CameraController,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,

    pub light_uniform: LightUniform,
    pub light_buffer: wgpu::Buffer,
    pub light_bind_group: wgpu::BindGroup,

    pub shadow_bind_group: wgpu::BindGroup,

    pub window_width: u32,
    pub window_height: u32,
}

impl Scene {
    pub async fn new(gfx: &GraphicsContext<'_>, renderer: &Renderer) -> Self {
        let nodes = SlotMap::with_key();
        let assets = SlotMap::with_key();
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

        let shadow_bind_group = gfx.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &renderer.shadow_bind_group_layout,
                entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&renderer.shadow_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&renderer.shadow_texture.sampler),
                        },
                    ],
                label: Some("camera_bind_group"),
            }
        );

        // add lighting
        let light_uniform = LightUniform::new(
            [10.0, 9.0, -6.0],
            [1.0, 0.96, 0.89],
            [-10.0, -9.0, 6.0],
            // [-0.678844, -0.61096, 0.407307],
            f32::cos(30.0_f32.to_radians()),
            f32::cos(45.0_f32.to_radians()),
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
            shadow_bind_group,
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
    ) -> Option<ModelId> {
        // load raw mesh data
        match resources::load_model(
            file_path,
            &gfx.device,
            &gfx.queue,
            &renderer.texture_bind_group_layout
        ).await {
            Ok(model) => {
                // wrap in asset container, creating empty instance buffer
                let asset = ModelAsset::new(&gfx.device, model);
                let id = self.assets.len();
                Some(self.assets.insert(asset))
            },
            Err(e) => {
                log::error!("Error loading model {}: {:?}", file_path, e);
                None
            }
        }
    }

    // loads a primitive object/model
    pub fn load_primitive(
        &mut self,
        primitive: Primitive,
        gfx: &GraphicsContext<'_>,
        renderer: &Renderer,
    ) -> ModelId {
        self.load_primitive_colored(primitive, gfx, renderer, [1.0, 1.0, 1.0])
    }

    // loads a primitive object/model with a color
    pub fn load_primitive_colored(
        &mut self,
        primitive: Primitive,
        gfx: &GraphicsContext<'_>,
        renderer: &Renderer,
        color: [f32; 3],
    ) -> ModelId {
        let model = match primitive {
            Primitive::Quad => {
                Primitives::quad(&gfx.device, &gfx.queue, &renderer.texture_bind_group_layout, color)
            }
            Primitive::Cube => {
                Primitives::cube(&gfx.device, &gfx.queue, &renderer.texture_bind_group_layout, color)
            }
        };

        let asset = ModelAsset::new(&gfx.device, model);
        let id = self.assets.len();
        self.assets.insert(asset)
    }

    // spawns a node into the world, returns ID handle
    pub fn spawn_node(&mut self, node: Node) -> NodeId {
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
        use cgmath::{Matrix4, Point3, Vector3, Rad};
        let light_view_matrix = Matrix4::look_at_rh(
            self.light_uniform.position.into(),
            Point3::from(self.light_uniform.position) + Vector3::from(self.light_uniform.direction),
            Vector3::unit_y()
        );
        let light_proj_matrix: Matrix4<f32> = cgmath::perspective(
            Rad(self.light_uniform.outer_cutoff.acos() * 2.0),
            1.0, 0.1, 1000.0
        );
        self.light_uniform.light_view_proj = (light_proj_matrix * light_view_matrix).into();

        queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );

        // process nodes and instancing
        for (_, asset) in &mut self.assets {
            asset.instance_count = 0;
        }

        // create temporary buckets to hold the raw GPU data for each model
        let mut instance_data: HashMap<ModelId, Vec<InstanceRaw>> = HashMap::new();
        for model_id in self.assets.keys() {
            instance_data.insert(model_id, Vec::new());
        }

        // calculate all global transforms first
        let mut global_transforms = Vec::with_capacity(self.nodes.len());
        for (id, _) in self.nodes.iter() {
            global_transforms.push((id, self.get_global_transform(id)));
        }

        // apply global transforms to nodes
        for (id, global_matrix) in global_transforms {
            if let Some(node) = self.nodes.get_mut(id) {
                node.global_transform = global_matrix;

                // if the node has a model, convert it to bytes and bucket it
                if let Some(model_id) = node.model_id {
                    let raw = InstanceRaw {
                        model: node.global_transform.into(),
                        normal: Matrix3::from(node.transform.rotation).into(),
                        color: node.color,
                    };

                    if let Some(bucket) = instance_data.get_mut(&model_id) {
                        bucket.push(raw);
                    }
                    if let Some(asset) = self.assets.get_mut(model_id) {
                        asset.instance_count += 1;
                    }
                }
            }
        }

        // send the buckets to the wgpu buffers
        for (model_id, asset) in self.assets.iter_mut() {
            if asset.instance_count > 0 {
                // check if we need to allocate for more instances
                asset.resize_buffer_if_needed(device, asset.instance_count);

                if let Some(bucket) = instance_data.get(&model_id) {
                    queue.write_buffer(
                        &asset.instance_buffer,
                        0,
                        bytemuck::cast_slice(bucket)
                    );
                }
            }
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// represents a colored spotlight
pub struct LightUniform {
    pub position: [f32; 3],
    _padding: f32,
    pub color: [f32; 3],
    _padding2: f32,
    // normalized vector where light is shining
    pub direction: [f32; 3],
    // angles from the direction where brightness is 100% to 0% (in cosines)
    pub inner_cutoff: f32,
    pub outer_cutoff: f32,
    _padding3: [f32; 3],
    pub light_view_proj: [[f32; 4]; 4],
}

impl LightUniform {
    pub fn new(position: [f32; 3], color: [f32; 3], 
        direction: [f32; 3], inner_cutoff: f32, outer_cutoff: f32
    ) -> Self {

        Self {
            position,
            _padding: 0.0,
            color,
            _padding2: 0.0,
            direction,
            inner_cutoff,
            outer_cutoff,
            _padding3: [0.0, 0.0, 0.0],
            light_view_proj: [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]],
        }
    }
}