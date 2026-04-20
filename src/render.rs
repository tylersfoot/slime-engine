use crate::core::GraphicsContext;
use crate::model::DrawLight;
use crate::model::{DrawModel, DrawShadow};
use crate::model::Vertex;
use crate::scene::Scene;
use crate::texture;
use crate::model;


pub struct Renderer {
    // pipeline: complete, pre-configured state object that defines how to draw
    // bundles: vertex/fragment shader, vertex data layout, type of primitive (tri),
    // settings for depth testing, color blending, etc.
    // by bundling, the GPU can swap render pipelines insanely fast
    pub render_pipeline: wgpu::RenderPipeline,
    pub light_render_pipeline: wgpu::RenderPipeline,
    pub shadow_render_pipeline: wgpu::RenderPipeline,

    pub shadow_bind_group_layout: wgpu::BindGroupLayout,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub light_bind_group_layout: wgpu::BindGroupLayout,

    pub depth_texture: texture::Texture,
    pub shadow_texture: texture::Texture,
}

impl Renderer {
    pub fn new(gfx: &GraphicsContext) -> Self {

        // create shadow mapping texture
        let shadow_texture = texture::Texture::create_shadow_texture(
            &gfx.device, &gfx.config, "shadow_texture", 4096
        );

        // a bind group describes a set of resources and how they can be accessed by the shader
        // this is separate from the layout because it allows us to swap bind groups
        // on the fly given they share the same layout
        // each texture/sampler has to be added to a bind group
        let texture_bind_group_layout = gfx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // diffuse texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        // these can only be used in fragment shader
                        // can be any bitwise combination of NONE, VERTEX, FRAGMENT, COMPUTE
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    // diffuse sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // should match filterable field from above entry
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // material uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }
                ],
                label: Some("texture_bind_group_layout"),
            }
        );

        let shadow_bind_group_layout = gfx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // shadow texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    // shadow sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
                label: Some("shadow_bind_group_layout"),
            }
        );

        let camera_bind_group_layout = gfx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // means the location of the data in the buffer can change, like if you
                        // store multiple data sets that vary in size in a single buffer
                        // if true, you have to specify the offsets later
                        has_dynamic_offset: false,
                        // the smallest size the buffer can be; don't rlly need to specify
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            }
        );

        let light_bind_group_layout = gfx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("light_bind_group_layout"),
            }
        );

        // create our depth texture for correct z rendering
        let depth_texture = texture::Texture::create_depth_texture(
            &gfx.device, &gfx.config, "depth_texture"
        );
        
        // the pipeline layout defines the interface between the pipeline
        // and the external GPU resources (textures, uniforms, storage buffers)
        let render_pipeline_layout =
            gfx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render_pipeline_layout"),
                // a bind group is a way to group resources that a shader needs access to
                // ex. a group for scene data, material-specific data
                bind_group_layouts: &[
                    Some(&shadow_bind_group_layout),
                    Some(&texture_bind_group_layout),
                    Some(&camera_bind_group_layout),
                    Some(&light_bind_group_layout),
                ],
                // a way to send very small amounts of data to shaders very quickly, but limited
                immediate_size: 0,
            });

        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("normal_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/shader.wgsl").into()),
            };
            create_render_pipeline(
                &gfx.device,
                &render_pipeline_layout,
                gfx.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                wgpu::CompareFunction::Less,
                true,
                &[model::ModelVertex::desc(), crate::model::InstanceRaw::desc()],
                shader,
                true,
                wgpu::DepthBiasState::default(),
                wgpu::Face::Back,
                Some("render_pipeline"),
            )
        };

        let light_render_pipeline = {
            let layout = gfx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("light_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&camera_bind_group_layout),
                    Some(&light_bind_group_layout)
                ],
                immediate_size: 0,
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("light_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/light.wgsl").into()),
            };
            create_render_pipeline(
                &gfx.device,
                &layout,
                gfx.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                wgpu::CompareFunction::Less,
                true,
                &[model::ModelVertex::desc()],
                shader,
                true,
                wgpu::DepthBiasState::default(),
                wgpu::Face::Back,
                Some("light_render_pipeline"),
            )
        };

        let shadow_render_pipeline = {
            let layout = gfx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&light_bind_group_layout)
                ],
                immediate_size: 0,
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("shadow_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/shadow.wgsl").into()),
            };
            create_render_pipeline(
                &gfx.device,
                &layout,
                gfx.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                wgpu::CompareFunction::Less,
                true,
                &[model::ModelVertex::desc(), crate::model::InstanceRaw::desc()],
                shader,
                false,
                wgpu::DepthBiasState {
                    constant: 0, // 2
                    slope_scale: 1.5, // 2.0
                    clamp: 0.0,
                },
                wgpu::Face::Front,
                Some("shadow_render_pipeline"),
            )
        };

        Self {
            render_pipeline,
            light_render_pipeline,
            shadow_render_pipeline,
            shadow_bind_group_layout,
            texture_bind_group_layout,
            camera_bind_group_layout,
            light_bind_group_layout,
            depth_texture,
            shadow_texture,
        }
    }

    pub fn resize(&mut self, gfx: &GraphicsContext) {
        // when the window resizes, the renderer only needs to rebuild the depth texture
        self.depth_texture = texture::Texture::create_depth_texture(
            &gfx.device, &gfx.config, "depth_texture"
        );
    }

    // draws the current frame
    pub fn render(
        &self,
        gfx: &GraphicsContext,
        scene: &Scene,
        view: &wgpu::TextureView,
    ) {
        // only draw if we have an active camera
        if scene.active_camera.is_some() {
            // create the CommandEncoder, which records a list of all commands to be sent to the GPU
            let mut encoder = gfx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

            // run an initial shadow render pass, with the depth texture set to shadow_texture
            // and render all objects to save their depths to the shadow texture
            let mut shadow_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_render_pass"),
                color_attachments: &[],
                // attach depth texture to render objects in the right order
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });

            shadow_render_pass.set_pipeline(&self.shadow_render_pipeline);

            for asset in scene.assets.values() {
                if asset.instance_count > 0 {
                    shadow_render_pass.set_vertex_buffer(1, asset.instance_buffer.slice(..));
                    shadow_render_pass.draw_shadow_model_instanced(
                        &asset.model,
                        0..asset.instance_count,
                        &scene.light_bind_group,
                    );
                }
            }

            drop(shadow_render_pass);


            // begins a render pass, a block of drawing operations that target the same set of framebuffers/attachments
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                // specifies where the final colors will be written
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        // attach the view of the frame texture
                        view,
                        resolve_target: None,
                        // tells the GPU what to do with the attachment 
                        // at the beginning (load) and end (store) of the pass
                        ops: wgpu::Operations {
                            // at the start, clear the texture to a background color
                            load: wgpu::LoadOp::Clear(
                                wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }
                            ),
                            // at the end, store the results into the texture
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })
                ],
                // attach depth texture to render objects in the right order
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });


            // draw light
            render_pass.set_pipeline(&self.light_render_pipeline);

            // get first object for now
            // TODO: dedicated light object?
            if let Some(light_asset) = scene.assets.values().next() {
                render_pass.draw_light_model(
                    &light_asset.model,
                    &scene.camera_bind_group,
                    &scene.light_bind_group,
                );
            }


            // draw scene objects
            render_pass.set_pipeline(&self.render_pipeline);

            for asset in scene.assets.values() {
                if asset.instance_count > 0 {
                    // bind specific buffer for this model
                    render_pass.set_vertex_buffer(1, asset.instance_buffer.slice(..));

                    // draw all instances
                    render_pass.draw_model_instanced(
                        &asset.model,
                        0..asset.instance_count,
                        &scene.camera_bind_group,
                        &scene.light_bind_group,
                        &scene.shadow_bind_group,
                    );
                }
            }
            
            // drop render pass manually
            drop(render_pass);

            // tell the encoder we are done recording and to package all commands into a command buffer
            // submit the command buffer with all commands to the queue
            gfx.queue.submit(Some(encoder.finish()));

        }
    }
}


fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    depth_compare: wgpu::CompareFunction,
    depth_write_enabled: bool,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    has_fragment: bool,
    depth_bias_state: wgpu::DepthBiasState,
    cull_mode: wgpu::Face,
    label: Option<&str>,
) -> wgpu::RenderPipeline {
    // brings the shaders, data layout, and state settings together into a pipeline
    let shader = device.create_shader_module(shader);
    let fragment = if !has_fragment { 
        None 
    } else {
        // configures the fragment stage of the pipeline
        Some(wgpu::FragmentState {
            module: &shader,
            // start at function 'fs_main' in the shader
            entry_point: Some("fs_main"),
            // defines what the fragment shader is writing its output to
            targets: &[
                // defines a single output buffer (texture)
                Some(wgpu::ColorTargetState {
                    // must match format of surface (sanity check)
                    format: color_format,
                    // REPLACE: when the shader outputs a color, replace the previous color
                    // ALPHABLENDING: combine new color with old color based on transparency 
                    blend: Some(wgpu::BlendState {
                        alpha: wgpu::BlendComponent::REPLACE,
                        color: wgpu::BlendComponent::REPLACE,
                    }),
                    // can write to all color channels (rgba)
                    write_mask: wgpu::ColorWrites::ALL,
                })
            ],
            compilation_options: Default::default(),
        })
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label,
        layout: Some(layout),
        // configures the vertex stage of the pipeline
        vertex: wgpu::VertexState {
            module: &shader,
            // start at function 'vs_main' in the shader
            entry_point: Some("vs_main"),
            // tells the pipeline how the vertex buffer data is laid out
            buffers: vertex_layouts,
            compilation_options: Default::default(),
        },
        fragment,
        // tells the GPU's rasterizer stage how to interpret the vertex data
        primitive: wgpu::PrimitiveState {
            // take buffer vertices 3 at a time as a list of independent triangles
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw, // winding rotation
            cull_mode: Some(cull_mode), // dont render triangles facing away
            // how to fill the triangles
            // Fill: color whole triangle
            // Line: wireframe
            // Point: draw points at vertices
            // Line/Point requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // dont clip outside depth range 0.0-1.0; requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // normally pixels are shaded if center falls inside triangle
            // this will shade if any part of pixel lies inside triangle
            // requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        // configure depth/stencil buffers
        // depth buffer: a texture that stores the depth of every pixel
        // stencil buffer: lets you perform masking operations
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(depth_write_enabled),
            // tells us when pixels are discarded
            // Less: pixels will be drawn front to back
            // other options: Never, Less, Equal, LessEqual,
            // Greater, NotEqual, GreaterEqual, Always
            depth_compare: Some(depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: depth_bias_state,
        }),
        // configures Multi-Sample Anti-Aliasing (MSAA)
        multisample: wgpu::MultisampleState {
            // amount of samples per pixel; 1 = not using MSAA
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        // render to multiple views (texture layers) in a single draw; used for VR
        multiview_mask: None,
        cache: None,
    })
}