use crate::window::Window;
use std::mem::ManuallyDrop;

pub struct GraphicsContext<'a>  {
    // the minifb window object onscreen
    pub window: Window,

    // the drawable part of the window, like the "canvas"
    pub surface: ManuallyDrop<wgpu::Surface<'a>>,

    // how the pixels on the surface are stored in memory
    pub surface_format: wgpu::TextureFormat,

    // handle to the physical GPU hardware
    pub adapter: wgpu::Adapter,

    // the software interface connection to the GPU
    // to send commands or create resources (buffers/textures/pipelines)
    pub device: wgpu::Device,

    // the channel to submit commands (like drawing instructions) for the GPU to execute
    pub queue: wgpu::Queue,

    // configuration for the surface, like width, height,
    // surface_format, present_mode (vsync) etc.
    // needs to be updated when anything changes (like resizing)
    pub config: wgpu::SurfaceConfiguration,

}

impl<'a> GraphicsContext<'a> {
    pub async fn new(window: Window) -> Self {
        // the instance is a handle to the GPU
        // BackendBit::PRIMARY -> Vulkan + Metal + DX12 + Browser WebGPU
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    // required_limits:  wgpu::Limits::default(),
                    required_limits: adapter.limits(),
                    memory_hints: Default::default(),
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .expect("Failed to create device");

        // get what the GPU is capable of (formats, vsync modes, etc.)
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

        Self {
            window,
            surface: ManuallyDrop::new(surface),
            surface_format,
            adapter,
            device,
            queue,
            config,
        }
    }

    pub fn configure_surface(&mut self, width: u32, height: u32) {
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

        // only configure the surface if the dimensions are valid
        if width == 0 || height == 0 { return; }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    

    }
}


impl<'a> Drop for GraphicsContext<'a> {
    fn drop(&mut self) {
        // the surface object is fundamentally linked to the window;
        // the surface contains pointers to the OS level window.
        // we NEED the surface to be dropped before the window, or crashes occur,
        // so we define a manual drop method here to drop the surface first
        unsafe {
            ManuallyDrop::drop(&mut self.surface);
        }
    }
}