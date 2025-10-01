#![allow(unused)]

use std::{io::Write, mem::ManuallyDrop};
use minifb::KeyRepeat;
use wgpu::util::DeviceExt;
use image::GenericImageView;
pub use minifb::{CursorStyle, Key, MouseMode, WindowOptions};
use cgmath::prelude::*;
use std::time::Instant;

pub mod window;
pub mod texture;
pub mod model;
mod resources;

use window::*;
use texture::*;
use model::*;

// for instancing test
const NUM_INSTANCES_PER_ROW: u32 = 10;
const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(
    NUM_INSTANCES_PER_ROW as f32 * 0.5,
    0.0,
    NUM_INSTANCES_PER_ROW as f32 * 0.5,
);
// starting window size
const WIDTH: usize = 640;
const HEIGHT: usize = 480;


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

    #[allow(dead_code)]
    diffuse_texture: texture::Texture,
    diffuse_bind_group: wgpu::BindGroup,

    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    instances: Vec<Instance>,
    #[allow(dead_code)]
    instance_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,

    obj_model: Model,

    debug: DebugInfo,
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
        // init wgpu handle
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY, // selects best available graphics API
            ..Default::default()
        });

        // create the physical window
        let window = Window::new(
            "awesome window",
            WIDTH,
            HEIGHT,
            WindowOptions {
                resize: true,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| {
            panic!("{}", e);
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
        }
        .expect("Failed to create surface");

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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    // disable some features if WebGL
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    memory_hints: Default::default(),
                    trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device");

        // make all errors forward to the console before panicking so they also show up on the web
        device.on_uncaptured_error(Box::new(|err| {
            log::error!("{err}");
            panic!("{}", err);
        }));

        // get what the GPU is capable of (formats, vsync modes, etc.)
        let surface_caps = surface.get_capabilities(&adapter);

        // try to get sRGB format for consistent colors
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // different settings for the surface
        let config = wgpu::SurfaceConfiguration {
            // tell GPU that the primary use for this surface's textures
            // is to be drawn into, like an attachment in a render pass
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, 
            format: surface_format,
            width: window.get_size().0 as u32,
            height: window.get_size().1 as u32,
            // controls VSync; should be Fifo (standard VSync)
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // grab the bytes from our image file and load them into an image, convert to Vec of RGBA bytes
        let diffuse_bytes = include_bytes!("../assets/images/happy-tree.png");
        let diffuse_texture = Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").unwrap();

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
                ],
                label: Some("texture_bind_group_layout"),
            }
        );

        // a bind group describes a set of resources and how they can be accessed by the shader
        // this is separate from the layout because it allows us to swap bind groups
        // on the fly given they share the same layout
        // each texture/sampler has to be added to a bind group; we'll just make a new one for each
        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    },
                ],
                label: Some("diffuse_bind_group"),
            }
        );

        // camera instancing
        let camera = Camera {
            eye: (0.0, 5.0, 10.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };
        let camera_controller = CameraController::new(0.2);
 
        // create camera uniform so we can use our camera data in shaders
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);
        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // only use in vertex shader
                    visibility: wgpu::ShaderStages::VERTEX,
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

        let obj_model =
            resources::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
                .await
                .unwrap();

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
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // create our depth texture for correct z rendering
        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        // load/check shader at compile time
        let shader = device.create_shader_module(wgpu::include_wgsl!("../assets/shaders/shader.wgsl"));

        // the pipeline layout defines the interface between the pipeline
        // and the external GPU resources (textures, uniforms, storage buffers)
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                // a bind group is a way to group resources that a shader needs access to
                // ex. a group for scene data, material-specific data
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout
                ],
                // a way to send very small amounts of data to shaders very quickly, but limited
                push_constant_ranges: &[],
            });

        // bring the shaders, data layout, and state settings together into a pipeline object
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            // configures the vertex stage of the pipeline
            vertex: wgpu::VertexState {
                module: &shader,
                // start at function `vs_main` in the shader
                entry_point: Some("vs_main"),
                // tells the pipeline how the vertex buffer data is laid out
                buffers: &[
                    model::ModelVertex::desc(),
                    InstanceRaw::desc(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            // configures the vertex stage of the pipeline
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                // start at function `fs_main` in the shader
                entry_point: Some("fs_main"),
                // defines what the fragment shader is writing its output to
                targets: &[
                    // defines a single output buffer (texture)
                    Some(wgpu::ColorTargetState {
                        // must match format of surface (sanity check)
                        format: config.format,
                        // REPLACE: when the shader outputs a color, replace the previous color
                        // ALPHABLENDING: combine new color with old color based on transparency 
                        blend: Some(wgpu::BlendState::REPLACE),
                        // can write to all color channels (rgba)
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            // tells the GPU's rasterizer stage how to interpret vertex data
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                // tells us when pixels are discarded
                // Less: pixels will be drawn front to back
                // other options: Never, Less, Equal, LessEqual,
                // Greater, NotEqual, GreaterEqual, Always
                depth_compare: wgpu::CompareFunction::Less,
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
            multiview: None,
            cache: None,
        });

        // // allocate buffers on the GPU's VRAM and copy our data across
        // let vertex_buffer = device.create_buffer_init(
        //     &wgpu::util::BufferInitDescriptor {
        //         label: Some("Vertex Buffer"),
        //         // our data to be copied; bytemuck safely converts &[Vertex] into &[u8]
        //         contents: bytemuck::cast_slice(VERTICES),
        //         // tell GPU that this buffer is for vertex data during draw calls
        //         // this puts it in a memory region optimized for fast reads by shader cores
        //         // if writing to buffer from CPU often, use COPY_DST
        //         usage: wgpu::BufferUsages::VERTEX,
        //     }
        // );
        // let index_buffer = device.create_buffer_init(
        //     // same as vertex buffer
        //     &wgpu::util::BufferInitDescriptor {
        //         label: Some("Index Buffer"),
        //         contents: bytemuck::cast_slice(INDICES),
        //         usage: wgpu::BufferUsages::INDEX,
        //     }
        // );

        // // store count of all indices to tell GPU how many to draw later
        // let num_indices = INDICES.len() as u32;

        let mut application = Application {
            window,
            surface: ManuallyDrop::new(surface),
            surface_format,
            adapter,
            device,
            queue,
            config,
            render_pipeline,
            diffuse_texture,
            diffuse_bind_group,
            camera,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            instances,
            instance_buffer,
            depth_texture,
            obj_model,
            debug: DebugInfo::new(),
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
        self.camera.aspect = self.config.width as f32 / self.config.height as f32;
        // reconfigure the surface
        self.surface.configure(&self.device, &self.config);
        // recreate depth texture with new size
        self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    pub fn draw_frame(&mut self) {
        // draws the current frame

        // reset per-frame debug counters
        self.debug.draw_calls = 0;
        self.debug.rendered_tris = 0;

        let frame = match self.surface.get_current_texture() {
            // request a buffer/texture/frame from the swap chain
            Ok(surface_texture) => surface_texture,
            Err(err) => match err {
                wgpu::SurfaceError::Timeout => {
                    // took too long to get a new frame; try again next frame
                    log::warn!("Surface texture acquisition timed out.");
                    return;
                }
                wgpu::SurfaceError::Outdated => {
                    // window resized or something else made the swap chain obsolete
                    self.configure_surface();
                    return;
                }
                wgpu::SurfaceError::Lost => {
                    // swap chain was lost for a serious reason (like display driver reset)
                    log::error!("Swapchain has been lost.");
                    self.configure_surface();
                    return;
                }
                wgpu::SurfaceError::OutOfMemory => panic!("Out of memory on surface acquisition"),
                wgpu::SurfaceError::Other => panic!("Other surface error, check log for details"),
            },
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
                label: Some("Main encoder"),
            });

        // begins a render pass, a block of drawing operations that target the same set of framebuffers/attachments
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
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
            occlusion_query_set: None,
        });

        // sets the render pipeline and buffers, and draw our frame
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);

        render_pass.draw_model_instanced(&self.obj_model, 0..self.instances.len() as u32, &self.camera_bind_group);

        self.debug.draw_calls += 1;
        // self.debug.rendered_tris += (self.num_indices / 3) * self.instances.len() as u32;

        drop(render_pass); // manual drop

        // tell the encoder we are done recording and to package all commands into a command buffer
        let command_buffer = encoder.finish();
        // submit the command buffer with all commands to the Queue
        self.queue.submit(Some(command_buffer));
        // tell the swap chain we're done drawing this frame, ready to be presented to the screen
        frame.present()
    }
}

#[rustfmt::skip]
// wgpu's normalized device coordinates have the y-axis/x-axis range -1.0 to +1.0
// and z-axis range 0.0 to +1.0; cgmath uses OpenGL's coordinate system, so
// this matrix scales/translates to account for that
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);
 
struct Camera {
    // position of the camera in 3D world space
    eye: cgmath::Point3<f32>,
    // the point in space the camera is looking at
    // the direction the camera is facing = the vector from eye to target
    target: cgmath::Point3<f32>,
    // defines the "up" direction for the camera so it doesn't roll on its side
    up: cgmath::Vector3<f32>,
    // aspect ratio of the screen (w/h) to prevent stretched/squished image
    aspect: f32,
    // vertical field of view in degrees; basically zoom
    fovy: f32,
    // near/far clipping planes; any geometry outside this range will not be drawn
    znear: f32,
    zfar: f32,
}
 
impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        // world space -> view space
        // moves and rotates the whole world so the camera is at (0,0,0) looking down -Z axis
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        // view space -> ndc space (normalized device coordinates)
        // squashes the viewing frustrum (a pyramid-ish) into a perfect cube (the ndc)
        // warps the scene to account for depth (like far objects look closer to the middle)
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}
 
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // convert cgmath Matrix4 -> 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}
 
impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }
 
    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}
 
struct CameraController {
    speed: f32,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}
 
impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_up_pressed: false,
            is_down_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }
 
    fn handle_keys(&mut self, keys: &[Key]) {
        // handles key inputs
        self.is_up_pressed = keys.contains(&Key::Space);
        self.is_down_pressed = keys.contains(&Key::LeftShift);
        self.is_forward_pressed = keys.contains(&Key::W) | keys.contains(&Key::Up);
        self.is_left_pressed = keys.contains(&Key::A) | keys.contains(&Key::Left);
        self.is_backward_pressed = keys.contains(&Key::S) | keys.contains(&Key::Down);
        self.is_right_pressed = keys.contains(&Key::D) | keys.contains(&Key::Right);
    }

    fn update_camera(&self, camera: &mut Camera) {
        // move the camera's eye based on inputs
        let forward = (camera.target - camera.eye).normalize();
        let right = forward.cross(camera.up);

        if self.is_forward_pressed {
            camera.eye += forward * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward * self.speed;
        }

        if self.is_right_pressed {
            camera.eye += right * self.speed;
        }
        if self.is_left_pressed {
            camera.eye -= right * self.speed;
        }
    }
}

struct Instance {
    position: cgmath::Vector3<f32>,
    // quaternion is used to represent rotation
    rotation: cgmath::Quaternion<f32>,
}
 
impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation))
            .into(),    
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
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
            ],
        }
    }
}
 
pub fn run() {
    env_logger::init();

    // init the application; since Application::new() is async, 
    // because requesting adaptor/device from OS takes time)
    // pollster lets us call it in our sync fn and wait here until it's done 
    let mut application = pollster::block_on(Application::new());

    // main program loop
    loop {
        // processes events like key presses, mouse movements, close button, etc.
        application.window.update();

        let keys = application.window.get_keys();
        if keys.contains(&Key::Escape) {
            return; // exit when esc pressed
        } else {
            application.camera_controller.handle_keys(&keys);
        }
        if !keys.is_empty() {
            application.update();
        }

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