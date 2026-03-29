#![allow(unused)]

use std::{io::Write, mem::ManuallyDrop};
use minifb::{KeyRepeat, MouseButton};
use wgpu::util::DeviceExt;
use image::GenericImageView;
pub use minifb::{CursorStyle, Key, MouseMode, WindowOptions};
use cgmath::prelude::*;
use std::time::{Duration, Instant};

mod window;
mod texture;
mod model;
mod resources;
mod camera;

use crate::window::Window;
use crate::model::{DrawModel, Vertex, Model, DrawLight};
use crate::camera::{Camera, CameraUniform, CameraController, Projection};

const NUM_INSTANCES_PER_ROW: u32 = 100;
// starting window size
const WIDTH: usize = 640;
const HEIGHT: usize = 480;

struct Instance {
    position: cgmath::Vector3<f32>,
    // quaternion is used to represent rotation
    rotation: cgmath::Quaternion<f32>,
}
 
impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        let model = cgmath::Matrix4::from_translation(self.position)
            * cgmath::Matrix4::from(self.rotation);
        InstanceRaw {
            model: model.into(),
            normal: cgmath::Matrix3::from(self.rotation).into(),  
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}
 
impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // we use a step mode of Instance, where our shaders will only change 
            // to use the next instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // a mat4 takes up 4 vertex slots as it is technically 4 vec4s
                // we need to define a slot for each vec4 and reassemble in the shader
                wgpu::VertexAttribute {
                    offset: 0,
                    // while our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex; we'll start at slot 5 not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// represents a colored point in space
struct LightUniform {
    position: [f32; 3],
    _padding: u32,
    color: [f32; 3],
    _padding2: u32,
}

// a struct to hold real-time debug information
struct DebugInfo {
    last_update: Instant,
    frame_count: u32, // for fps calc
    fps: f32,
    pub draw_calls: u32,
    pub rendered_tris: u32,
    pub total_frames: u32,
}

impl DebugInfo {
    fn new() -> Self {
        Self {
            last_update: Instant::now(),
            frame_count: 0,
            fps: 0.0,
            draw_calls: 0,
            rendered_tris: 0,
            total_frames: 0,
        }
    }

    // this will be called every frame from our main loop
    fn update(&mut self) {
        self.frame_count += 1;
        self.total_frames += 1;
        let elapsed = self.last_update.elapsed().as_secs_f32();

        // update the stats every half-second
        if elapsed >= 0.5 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.last_update = Instant::now();
        }
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    label: Option<&str>,
) -> wgpu::RenderPipeline {
    // brings the shaders, data layout, and state settings together into a pipeline
    let shader = device.create_shader_module(shader);

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
        // configures the fragment stage of the pipeline
        fragment: Some(wgpu::FragmentState {
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
        }),
        // tells the GPU's rasterizer stage how to interpret the vertex data
        primitive: wgpu::PrimitiveState {
            // take buffer vertices 3 at a time as a list of independent triangles
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw, // winding rotation
            cull_mode: Some(wgpu::Face::Back), // dont render triangles facing away
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
            depth_write_enabled: Some(true),
            // tells us when pixels are discarded
            // Less: pixels will be drawn front to back
            // other options: Never, Less, Equal, LessEqual,
            // Greater, NotEqual, GreaterEqual, Always
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
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

struct Application<'a> {
    // the minifb window object onscreen
    window: Window,

    // the drawable part of the window, like the "canvas"
    surface: ManuallyDrop<wgpu::Surface<'a>>,

    // how the pixels on the surface are stored in memory
    surface_format: wgpu::TextureFormat,

    // handle to the physical GPU hardware
    adapter: wgpu::Adapter,

    // the software interface connection to the GPU
    // to send commands or create resources (buffers/textures/pipelines)
    device: wgpu::Device,

    // the channel to submit commands (like drawing instructions) for the GPU to execute
    queue: wgpu::Queue,

    // configuration for the surface, like width, height,
    // surface_format, present_mode (vsync) etc.
    // needs to be updated when anything changes (like resizing)
    config: wgpu::SurfaceConfiguration,

    // complete, pre-configured state object that defines how to draw
    // bundles: vertex/fragment shader, vertex data layout, type of primitive (tri),
    // settings for depth testing, color blending, etc.
    // by bundling, the GPU can swap render pipelines insanely fast
    render_pipeline: wgpu::RenderPipeline,

    camera: Camera,
    projection: Projection,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    instances: Vec<Instance>,
    #[allow(dead_code)]
    instance_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,

    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline,

    obj_model: Model,

    debug: DebugInfo,

    window_active: bool,
}

impl Drop for Application<'_> {
    // the surface object is fundamentally linked to the window;
    // the surface contains pointers to the OS level window.
    // we NEED the surface to be dropped before the window, or crashes occur,
    // so we define a manual drop method here to drop the surface first
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.surface);
        }
    }
}

impl Application<'_> {
    async fn new() -> Self {
        // create the physical window
        let mut window = Window::new(
            "awesome window",
            WIDTH,
            HEIGHT,
            WindowOptions {
                resize: true,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{}", e));
        window.set_target_fps(0); // uncapped framerate

        // the instance is a handle to the GPU
        // BackendBit::PRIMARY -> Vulkan + Metal + DX12 + Browser WebGPU
        log::warn!("wgpu setup");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: Some(Box::new(window.get_display_wrapper())),
        });

        // mini_fb's window type isn't `Send` which is required for wgpu's `WindowHandle` trait
        // so have to use the unsafe variant to create a surface directly from the window handle
        // - the window handles are valid at this point
        // - the window is guranteed to outlive the surface since we're ensuring so in `Application's` Drop impl
        let surface = unsafe {
            instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(&window.inner)
                    .expect("Failed to create surface target."),
            )
        }.expect("Failed to create surface");

        // get the handle to the physical GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(), // HighPerformance, LowPower
                compatible_surface: Some(&surface), // make sure the gpu use our surface
                force_fallback_adapter: false,
            })
            .await.expect("Failed to find an appropriate adapter");
        log::info!("Created wgpu adapter: {:?}", adapter.get_info());

        // get a logical connection to the gpu
        log::warn!("device and queue");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits:  wgpu::Limits::default(),
                    memory_hints: Default::default(),
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .expect("Failed to create device");

        // get what the GPU is capable of (formats, vsync modes, etc.)
        log::warn!("surface");
        let surface_caps = surface.get_capabilities(&adapter);
        // try to get sRGB format for consistent colors
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        // let surface_format = surface_format.remove_srgb_suffix(); // not sure why the tutorial uses this

        // different settings for the surface
        let config = wgpu::SurfaceConfiguration {
            // tell GPU that the primary use for this surface's textures
            // is to be drawn into, like an attachment in a render pass
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, 
            format: surface_format,
            width: window.get_size().0 as u32,
            height: window.get_size().1 as u32,
            // controls VSync; defaults to Fifo (standard VSync)
            // present_mode: surface_caps.present_modes[0],
            // unlocked FPS for benchmarking
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // a bind group describes a set of resources and how they can be accessed by the shader
        // this is separate from the layout because it allows us to swap bind groups
        // on the fly given they share the same layout
        // each texture/sampler has to be added to a bind group
        let texture_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // sampled texture at binding 0
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
                    // sampler at binding 1
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // should match filterable field from above entry
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // normal map texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    // normal map sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            }
        );

        let camera = Camera::new(
            (0.0, 5.0, 10.0),
            cgmath::Deg(-90.0),
            cgmath::Deg(-20.0)
        );
        let projection = Projection::new(
            config.width,
            config.height,
            cgmath::Deg(45.0),
            0.1,
            100.0
        );
        let camera_controller = CameraController::new(4.0, 0.4);
 
        // create camera uniform so we can use our camera data in shaders
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);
            
        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("camera_buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(
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
 
        let camera_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &camera_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }],
                label: Some("camera_bind_group"),
            }
        );
        
        // instancing
        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
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
 
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instance_buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        log::warn!("load model");
        let obj_model =
            resources::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
                .await
                .unwrap();


        // add lighting
        let light_uniform = LightUniform {
            position: [2.0, 2.0, 2.0],
            _padding: 0,
            color: [1.0, 0.96, 0.89],
            _padding2: 0
        };

        // use COPY_DST to update light's position
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("light_buffer"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout = device.create_bind_group_layout(
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
 
        let light_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &light_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.as_entire_binding(),
                }],
                label: Some("light_bind_group"),
            }
        );

        // create our depth texture for correct z rendering
        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        // the pipeline layout defines the interface between the pipeline
        // and the external GPU resources (textures, uniforms, storage buffers)
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render_pipeline_layout"),
                // a bind group is a way to group resources that a shader needs access to
                // ex. a group for scene data, material-specific data
                bind_group_layouts: &[
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
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                Some("render_pipeline"),
            )
        };

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
                Some("light_render_pipeline"),
            )
        };


        let mut application = Application {
            window,
            surface: ManuallyDrop::new(surface),
            surface_format,
            adapter,
            device,
            queue,
            config,
            render_pipeline,
            camera,
            projection,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            instances,
            instance_buffer,
            depth_texture,
            light_uniform,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
            obj_model,
            debug: DebugInfo::new(),
            window_active: false,
        };

        // apply the config to the surface
        // tells GPU to create a swap chain (set of textures to draw to) with the w/h/format
        application.configure_surface();

        application
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn configure_surface(&mut self) {
        // applies the settings stored in the config to the surface itself
        // called at start and on window resize

        // Swap Chain Process:
        // modern rendering doesn't draw directly to the image on screen; instead, it uses two or three buffers:
        // Front buffer - the texture the monitor is currently reading from
        // Back buffer - a hidden texture where the application is drawing the next frame
        // Mailbox buffer - an optional third buffer that stores the last fully completed frame

        // the "swap" in swap chain happens when we want to update the texture the monitor is reading from;
        // so it swaps the pointer to the buffer the monitor will read instead of doing a frame copy
        // (the Back buffer becomes the Front buffer, and vice versa)

        // when using VSync, only the first two buffers are used; this is because it allows the 
        // swap to happen in sync with the refresh rate (VBlank period) to avoid tearing (half rendered frames)
        // otherwise, the GPU keeps churning frames out in the Back buffer, and when a frame is complete
        // it sends it to the Mailbox buffer, and when the monitor refreshes, it reads from the Mailbox buffer
        // which is guarenteed to be a full frame, preventing tearing (but using more VRAM)

        let (width, height) = self.window.get_size();
        // only configure the surface if the dimensions are valid
        if width == 0 || height == 0 {
            return;
        }
        // set the new window sizes
        self.config.width = width as u32;
        self.config.height = height as u32;
        self.projection.resize(width as u32, height as u32);
        // reconfigure the surface
        self.surface.configure(&self.device, &self.config);
        // recreate depth texture with new size
        self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
    }

    fn update(&mut self, dt: Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        let old_position: cgmath::Vector3<_> = self.light_uniform.position.into();
        self.light_uniform.position = (
            cgmath::Quaternion::from_axis_angle(
                (0.0, 1.0, 0.0).into(),
                cgmath::Deg(60.0 * dt.as_secs_f32())
            ) * old_position
        ).into();
        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );
    }

    pub fn draw_frame(&mut self) {
        // draws the current frame

        // reset per-frame debug counters
        self.debug.draw_calls = 0;
        self.debug.rendered_tris = 0;

        let frame = match self.surface.get_current_texture() {
            // request a buffer/texture/frame from the swap chain
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.configure_surface();
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                // skip this frame
                return;
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                // window resized or something else made the swap chain obsolete
                self.configure_surface();
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                // swap chain was lost for a serious reason (like display driver reset)
                log::error!("Swapchain has been lost.");
                self.configure_surface();
                return;
            }
        };

        // we draw to a TextureView instead of directly to the texture
        // a TextureView is like a lens or interpretation of a texture, it describes
        // how we want to look at and use the texture (eg. mipmap levels or array layers to use)
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // create the CommandEncoder, which records a list of all commands to be sent to the GPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        // begins a render pass, a block of drawing operations that target the same set of framebuffers/attachments
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_pass"),
            // specifies where the final colors will be written
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    // attach the view of the frame texture
                    view: &view,
                    resolve_target: None,
                    // tells the GPU what to do with the attachment 
                    // at the beginning (load) and end (store) of the pass
                    ops: wgpu::Operations {
                        // at the start, clear the texture to a background color
                        load: wgpu::LoadOp::Clear(
                            wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
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

        // sets the render pipeline and buffers, and draw our frame
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        
        render_pass.set_pipeline(&self.light_render_pipeline);
        render_pass.draw_light_model(
            &self.obj_model,
            &self.camera_bind_group,
            &self.light_bind_group,
        );
        
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.draw_model_instanced(
            &self.obj_model,
            0..self.instances.len() as u32,
            &self.camera_bind_group,
            &self.light_bind_group,
        );

        self.debug.draw_calls += 1;
        // self.debug.rendered_tris += (self.num_indices / 3) * self.instances.len() as u32;

        drop(render_pass);

        // tell the encoder we are done recording and to package all commands into a command buffer
        let command_buffer = encoder.finish();
        // submit the command buffer with all commands to the Queue
        self.queue.submit(Some(command_buffer));
        // tell the swap chain we're done drawing this frame, ready to be presented to the screen
        frame.present()
    }
}
 
pub fn run() {
    // init the application; since Application::new() is async, 
    // (requesting adaptor/device from OS takes time)
    // pollster lets us call it in our sync fn and wait here until it's done 
    let mut application = pollster::block_on(Application::new());
    let mut last_render_time = Instant::now();

    // main program loop
    loop {
        // processes events like key presses, mouse movements, close button, etc.
        application.window.update();

        // handle special keys
        let keys = application.window.get_keys();
        if keys.contains(&Key::Backspace) { return }
        if (keys.contains(&Key::Escape) && application.window_active) {
            // unlock mouse
            application.window.set_cursor_visibility(true);
            application.window_active = false;
        }

        let mouse_pressed = application.window.get_mouse_down(MouseButton::Left);
        if (mouse_pressed && !application.window_active) {
            // lock mouse
            application.window.set_cursor_visibility(false);
            application.window_active = true;

            // snap immediatly to center to camera doesn't jump
            let (width, height) = application.window.get_size();
            application.window.set_mouse_pos((width / 2) as f32, (height / 2) as f32);
        }

        // handle scroll wheel
        let scroll_wheel_delta = application.window.get_scroll_wheel().unwrap_or((0.0, 0.0)).1;
        
        // handle infinite mouse movement
        let mut mouse_delta = (0.0, 0.0);
        if application.window_active
            && let Some((x, y)) = application.window.get_mouse_pos(MouseMode::Pass) {
                let (width, height) = application.window.get_size();
                let center_x = (width / 2) as f32;
                let center_y = (height / 2) as f32;

                let dx = x - center_x;
                let dy = y - center_y;

                // only update and snap is mouse actually moved
                if dx != 0.0 || dy != 0.0 {
                    mouse_delta = (dx, dy);
                    application.window.set_mouse_pos(center_x, center_y);
                }
        }

        // handle inputs
        application.camera_controller.handle_keys(&keys);
        application.camera_controller.handle_mouse(mouse_delta.0, mouse_delta.1);
        application.camera_controller.handle_mouse_scroll(scroll_wheel_delta);

        // calculate delta time
        let now = Instant::now();
        let dt = now - last_render_time;
        last_render_time = now;
        application.update(dt);

        if !application.window.is_open() {
            return; // exit if window is closed
        }

        application.draw_frame();
        application.debug.update();
        print!(
            "\rFPS: {:.2} | Draw Calls: {:<3} | Tris: {:<4} | Total Frames Drawn: {:<6}",
            application.debug.fps,
            application.debug.draw_calls,
            application.debug.rendered_tris,
            application.debug.total_frames,
        );
        let _ = std::io::stdout().flush();
    }
}