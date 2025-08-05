//! Configurable Headless Rendering Benchmark
//! 
//! INDIVIDUAL FUNCTION PROFILES:
//! 
//! Core Buffer Operations:
//! - buffer_clear: Buffer clearing performance
//! - buffer_merge: Buffer merging (depth testing) 
//! - buffer_to_raw: Buffer to u32 array conversion
//! - scene_render: Full scene rendering pipeline
//! 
//! Triangle & Line Rendering:
//! - draw_triangle: Triangle rasterization (wrapper around render_tri)
//! - render_tri: Direct triangle rendering with full pipeline
//! - draw_line: Line drawing (Bresenham/DDA algorithms)
//! 
//! 3D Projection Pipeline:
//! - project_to_screen_space: Complete 3D to 2D transformation
//! - project_world_to_view: World space to camera/view space
//! - project_view_to_clip: View space to clip space (perspective)
//! - project_clip_to_ndc: Clip space to normalized device coordinates
//! - project_ndc_to_screen: NDC to screen pixel coordinates
//! 
//! Clipping & Culling:
//! - clip_line: 2D line clipping (Cohen-Sutherland algorithm)
//! - clip_line_to_camera_plane: 3D line clipping against camera near plane
//! - compute_region_code: Clipping region code calculation
//! - clip_triangle_against_near_plane: Triangle clipping against camera
//! - is_backface: Backface culling detection
//! - is_fully_outside_clip_space: Clip space frustum culling
//! - is_fully_outside_ndc: NDC space frustum culling
//! 
//! Color Operations:
//! - hue_shift: HSV hue rotation for color effects
//! - random_color: Random color generation
//! - random_color_seeded: Deterministic seeded random colors
//! - rgb_to_hsv: RGB to HSV color space conversion
//! - hsv_to_rgb: HSV to RGB color space conversion

use slime_engine::*;
use std::time::{Instant, Duration};

// ANSI color codes for terminal output
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const GREEN: &str = "\x1b[32m";
const BLUE: &str = "\x1b[34m";
const CYAN: &str = "\x1b[36m";
const YELLOW: &str = "\x1b[33m";
const MAGENTA: &str = "\x1b[35m";
const BRIGHT_GREEN: &str = "\x1b[92m";
const BRIGHT_BLUE: &str = "\x1b[94m";
const BRIGHT_CYAN: &str = "\x1b[96m";
const BRIGHT_YELLOW: &str = "\x1b[93m";
const GRAY: &str = "\x1b[90m";

// =============================================================================
// BENCHMARK CONFIGURATION
// =============================================================================
const WIDTH: usize = 256;
const HEIGHT: usize = WIDTH;

// Toggle which benchmark sections to run
struct BenchmarkConfig {
    // Benchmark sections
    individual_functions: bool,
    scene_complexity: bool,
    buffer_operations: bool,
    rendering_techniques: bool,
    full_pipeline: bool,
    
    // Individual function sub-toggles (only used if individual_functions = true)
    test_buffer_clear: bool,
    test_scene_render: bool,
    test_buffer_merge: bool,
    test_buffer_to_raw: bool,
    test_hue_shift: bool,
    test_project_to_screen_space: bool,
    test_clip_line_to_camera_plane: bool,
    test_draw_line: bool,
    test_clip_line: bool,
    test_compute_region_code: bool,
    test_render_tri: bool,
    test_project_world_to_view: bool,
    test_project_view_to_clip: bool,
    test_project_clip_to_ndc: bool,
    test_project_ndc_to_screen: bool,
    test_is_backface: bool,
    test_clip_triangle_against_near_plane: bool,
    test_is_fully_outside_clip_space: bool,
    test_is_fully_outside_ndc: bool,
    test_random_color: bool,
    test_random_color_seeded: bool,
    test_rgb_to_hsv: bool,
    test_hsv_to_rgb: bool,
    
    // Timing configuration
    test_length_seconds: f64,  // How long to run each benchmark test
    
    // Output format
    simple_output: bool,  // Single line per benchmark vs detailed stats
}

const CONFIG: BenchmarkConfig = BenchmarkConfig {
    // CURRENT CONFIG: Test projection pipeline functions for optimization
    individual_functions: true,
    scene_complexity: false,
    buffer_operations: false,
    rendering_techniques: false,
    full_pipeline: false,
    
    // Individual function tests
    test_buffer_clear:                     true,
    test_scene_render:                     true,
    test_buffer_merge:                     true,
    test_buffer_to_raw:                    true,
    test_hue_shift:                        true,
    test_project_to_screen_space:          true,
    test_clip_line_to_camera_plane:        true,
    test_draw_line:                        true,
    test_clip_line:                        true,
    test_compute_region_code:              true,
    test_render_tri:                       true,
    test_project_world_to_view:            true,
    test_project_view_to_clip:             true,
    test_project_clip_to_ndc:              true,
    test_project_ndc_to_screen:            true,
    test_is_backface:                      true,
    test_clip_triangle_against_near_plane: true,
    test_is_fully_outside_clip_space:      true,
    test_is_fully_outside_ndc:             true,
    test_random_color:                     true,
    test_random_color_seeded:              true,
    test_rgb_to_hsv:                       true,
    test_hsv_to_rgb:                       true,

    test_length_seconds: 5.0,
    simple_output: true, // Single line output for easy comparison
};

#[derive(Clone)]
struct BenchmarkResult {
    name: String,
    iterations: usize,
    total_time: Duration,
    mean: Duration,
    median: Duration,
    min: Duration,
    max: Duration,
    p99: Duration,
    fps: f64,
    throughput: Option<f64>, // operations per second
}

impl BenchmarkResult {
    fn new(name: String, mut times: Vec<Duration>) -> Self {
        times.sort();
        let len = times.len();
        let total: Duration = times.iter().sum();
        let mean = total / len as u32;
        let median = times[len / 2];
        let min = times[0];
        let max = times[len - 1];
        let p99 = times[(len as f32 * 0.99) as usize];
        let fps = 1.0 / mean.as_secs_f64();
        
        BenchmarkResult {
            name,
            iterations: len,
            total_time: total,
            mean,
            median,
            min,
            max,
            p99,
            fps,
            throughput: None,
        }
    }

    fn with_throughput(mut self, ops_per_iteration: f64) -> Self {
        self.throughput = Some(ops_per_iteration / self.mean.as_secs_f64());
        self
    }

    fn print_results(&self, simple: bool) {
        if simple {
            // Color the FPS number based on performance level
            let fps_color = if self.fps >= 1000.0 {
                BRIGHT_GREEN
            } else if self.fps >= 100.0 {
                GREEN
            } else if self.fps >= 10.0 {
                YELLOW
            } else {
                MAGENTA
            };
            
            println!("{CYAN}{}{RESET}: {BOLD}{fps_color}{:.0}{RESET}fps {GRAY}({RESET}{BRIGHT_BLUE}{}{RESET} {GRAY}frames in{RESET} {BRIGHT_CYAN}{:.2}s{RESET}{GRAY}){RESET}", 
                self.name, self.fps, self.iterations, self.total_time.as_secs_f64()
            );
        } else {
            println!("\n{BOLD}{BLUE} === {}{RESET} ==={RESET}", self.name);
            println!("{CYAN}Iterations{RESET}: {BRIGHT_YELLOW}{}{RESET}", self.iterations);
            println!("{CYAN}Total time{RESET}: {BRIGHT_YELLOW}{:.3}s{RESET}", self.total_time.as_secs_f64());
            println!("{CYAN}Mean{RESET}: {BRIGHT_YELLOW}{:.3}ms{RESET}", self.mean.as_secs_f64() * 1000.0);
            println!("{CYAN}Median{RESET}: {BRIGHT_YELLOW}{:.3}ms{RESET}", self.median.as_secs_f64() * 1000.0);
            println!("{CYAN}Min{RESET}: {BRIGHT_GREEN}{:.3}ms{RESET}", self.min.as_secs_f64() * 1000.0);
            println!("{CYAN}Max{RESET}: {MAGENTA}{:.3}ms{RESET}", self.max.as_secs_f64() * 1000.0);
            println!("{CYAN}99th percentile{RESET}: {BRIGHT_YELLOW}{:.3}ms{RESET}", self.p99.as_secs_f64() * 1000.0);
            
            if let Some(throughput) = self.throughput {
                // println!("{}Throughput{}: {}{:.2} ops/sec{}", CYAN, RESET, BRIGHT_CYAN, throughput, RESET);
                println!("{CYAN}Throughput{RESET}: {BRIGHT_CYAN}{throughput:.2} ops/sec{RESET}");
            }
            
            let fps_color = if self.fps >= 1000.0 {
                BRIGHT_GREEN
            } else if self.fps >= 100.0 {
                GREEN
            } else if self.fps >= 10.0 {
                YELLOW
            } else {
                MAGENTA
            };
            
            println!("{CYAN}FPS{RESET}: {BOLD}{fps_color}{:.2}{RESET}", self.fps);
        }
    }
}

fn main() {
    println!("{BOLD}{BRIGHT_CYAN}> Headless Rendering Benchmark <{RESET}");
    println!("{CYAN}Resolution{RESET}: {BRIGHT_YELLOW}{WIDTH}{RESET}x{BRIGHT_YELLOW}{HEIGHT}{RESET}");
    println!("{CYAN}Test duration{RESET}: {BRIGHT_YELLOW}{:.1}s{RESET} per benchmark", CONFIG.test_length_seconds);
    if CONFIG.simple_output {
        println!("{CYAN}Output{RESET}: {GREEN}Simple{RESET} (single line per benchmark)");
    } else {
        println!("{CYAN}Output{RESET}: {GREEN}Detailed{RESET} statistics");
    }
    println!("{BLUE}=========================================={RESET}\n");

    // Create test scenes with different complexities
    let scenes = create_test_scenes();
    let mut all_results = Vec::new();

    // Run enabled benchmark sections
    if CONFIG.individual_functions {
        println!("{BOLD}{BRIGHT_GREEN}> Running Individual Function Benchmarks...{RESET}");
        all_results.extend(benchmark_individual_functions(&scenes));
    }

    if CONFIG.scene_complexity {
        println!("{BOLD}{BRIGHT_GREEN}> Running Scene Complexity Benchmarks...{RESET}");
        all_results.extend(benchmark_scene_complexity());
    }

    if CONFIG.buffer_operations {
        println!("{BOLD}{BRIGHT_GREEN}> Running Buffer Operation Benchmarks...{RESET}");
        all_results.extend(benchmark_buffer_operations());
    }

    if CONFIG.rendering_techniques {
        println!("{BOLD}{BRIGHT_GREEN}> Running Rendering Technique Benchmarks...{RESET}");
        all_results.extend(benchmark_rendering_techniques());
    }

    if CONFIG.full_pipeline {
        println!("{BOLD}{BRIGHT_GREEN}> Running Full Pipeline Benchmarks...{RESET}");
        all_results.extend(benchmark_full_pipeline(&scenes));
    }

    // Print results
    print_benchmark_summary(&all_results);
}

/// Helper function to run a benchmark for a specific time duration
fn run_timed_benchmark<F>(name: String, mut test_fn: F) -> BenchmarkResult
where
    F: FnMut(),
{
    let mut times = Vec::new();
    let test_duration = Duration::from_secs_f64(CONFIG.test_length_seconds);
    let start_time = Instant::now();

    println!("  {CYAN}Benchmarking{RESET}: {BRIGHT_BLUE}{name}{RESET}");
    while start_time.elapsed() < test_duration {
        let iteration_start = Instant::now();
        test_fn();
        times.push(iteration_start.elapsed());
    }
    
    if times.is_empty() {
        // Fallback: run at least one iteration if the function is extremely slow
        let iteration_start = Instant::now();
        test_fn();
        times.push(iteration_start.elapsed());
    }
    
    BenchmarkResult::new(name, times)
}

fn create_test_scenes() -> Vec<(String, Scene)> {
    let mut scenes = Vec::new();

    // Small scene
    let camera = Camera::new(
        Position { x: 0.0, y: 0.0, z: 0.0 },
        Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
        Scale { x: 1.0, y: 1.0, z: 1.0 },
    );

    let cube = RectangularPrism::new_cube(
        Transform {
            position: Position { x: 0.0, y: 0.0, z: -5.0 },
            rotation: Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
            scale: Scale { x: 1.0, y: 1.0, z: 1.0 },
        },
        2.0
    );

    let small_scene = Scene::new(camera, vec![Box::new(cube)]);
    scenes.push(("Small".to_string(), small_scene));

    // Medium scene with floor
    let mut objects: Vec<Box<dyn Object>> = Vec::new();
    
    // Add main cube
    let main_cube = RectangularPrism::new_cube(
        Transform {
            position: Position { x: 0.0, y: 0.0, z: -5.0 },
            rotation: Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
            scale: Scale { x: 1.0, y: 1.0, z: 1.0 },
        },
        2.0
    );
    objects.push(Box::new(main_cube));

    // Add floor grid
    let grid_size = 5;
    let tile_size = 4.0;
    let floor_y = -8.0;
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = (i as f32 - (grid_size as f32) / 2.0 + 0.5) * tile_size;
            let z = (j as f32 - (grid_size as f32) / 2.0 + 0.5) * tile_size;
            let tile = RectangularPrism::new(
                Transform {
                    position: Position { x, y: floor_y, z },
                    rotation: Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
                    scale: Scale { x: 1.0, y: 1.0, z: 1.0 },
                },
                (tile_size, 0.5, tile_size),
            );
            objects.push(Box::new(tile));
        }
    }

    let medium_scene = Scene::new(camera, objects);
    scenes.push(("Medium".to_string(), medium_scene));


    objects = Vec::new();

    // floor grid
    let grid_size = 10;
    let tile_size = 4.0;
    let floor_y = 0.0;
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = (i as f32 - (grid_size as f32) / 2.0 + 0.5) * tile_size;
            let z = (j as f32 - (grid_size as f32) / 2.0 + 0.5) * tile_size;
            let tile = RectangularPrism::new_at_coords(x, floor_y, z, (tile_size, 0.5, tile_size));
            objects.push(Box::new(tile) as Box<dyn Object>);
        }
    }

    // pillar field with varying heights
    let pillar_grid = 7;
    let spacing = 3.0;
    for i in 0..pillar_grid {
        for j in 0..pillar_grid {
            let x = (i as f32 - (pillar_grid as f32) / 2.0) * spacing;
            let z = (j as f32 - (pillar_grid as f32) / 2.0) * spacing;
            // height wobbles so it's not all uniform
            let h = 1.0 + ((i as f32 * 0.5).sin() + (j as f32 * 0.5).cos()).abs() * 3.0;
            let pillar = RectangularPrism::new(
                Transform {
                    position: Position { x, y: floor_y + h / 2.0, z }, // center so bottom rests on floor
                    rotation: Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
                    scale: Scale { x: 1.0, y: h, z: 1.0 },
                },
                (1.0, h, 1.0),
            );
            objects.push(Box::new(pillar) as Box<dyn Object>);
        }
    }

    // ring of scattered cubes with varying rotation/size
    for k in 0..10 {
        let angle = k as f32 / 10.0 * std::f32::consts::TAU;
        let radius = 8.0;
        let x = angle.cos() * radius;
        let z = angle.sin() * radius - 5.0;
        let size = 0.5 + (k as f32 * 0.2);
        let cube = RectangularPrism::new_cube(
            Transform {
                position: Position { x, y: 10.0, z },
                rotation: Rotation { pitch: k as f32 * 12.0, yaw: k as f32 * 20.0, roll: 0.0 },
                scale: Scale::default(),
            },
            size,
        );
        objects.push(Box::new(cube) as Box<dyn Object>);
    }
    let large_scene = Scene::new(camera, objects);
    scenes.push(("Large".to_string(), large_scene));

    // complex scene = 1000 cubes with random positions, rotations, and sizes
    let mut objects: Vec<Box<dyn Object>> = Vec::new();

    for _ in 0..1000 {
        let x = rand::random::<f32>() * 200.0 - 100.0; // random x in [-100, 100]
        let y = rand::random::<f32>() * 200.0; // random y in [0, 200]
        let z = rand::random::<f32>() * 200.0 - 100.0; // random z in [-100, 100]
        let size = rand::random::<f32>() * 50.0 + 0.1; // random size in [0.1, 50.1]
        let pitch = rand::random::<f32>() * 360.0;
        let yaw = rand::random::<f32>() * 360.0;
        let roll = rand::random::<f32>() * 360.0;

        let cube = RectangularPrism::new_cube(
            Transform {
                position: Position { x, y, z },
                rotation: Rotation { pitch, yaw, roll },
                scale: Scale::default(),
            },
            size,
        );

       objects.push(Box::new(cube) as Box<dyn Object>);
   }
    let camera = Camera::new(
        Position { x: 0.0, y: 0.0, z: 0.0 },
        Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
        Scale { x: 1.0, y: 1.0, z: 1.0 },
    );
    scenes.push(("Complex".to_string(), Scene::new(camera, objects)));

    scenes
}

fn benchmark_full_pipeline(scenes: &[(String, Scene)]) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    for (scene_name, scene) in scenes {
        let mut buffer = Buffer::new(WIDTH, HEIGHT);
        let mut frame_counter = 0;
        
        let result = run_timed_benchmark(
            format!("Full Pipeline - {scene_name}"),
            || {
                // Full rendering pipeline
                buffer.clear();
                scene.render(&mut buffer);
                
                // Add gradient overlay
                let gradient = draw_frame_gradient(&buffer, frame_counter);
                buffer.merge(&gradient);
                
                // Convert to raw format
                let _raw = buffer.to_raw();
                
                frame_counter += 1;
            }
        );
        
        results.push(result);
    }

    results
}

fn benchmark_individual_functions(scenes: &[(String, Scene)]) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // Test buffer clear
    if CONFIG.test_buffer_clear {
        let mut buffer = Buffer::new(WIDTH, HEIGHT);
        
        let result = run_timed_benchmark(
            "Buffer Clear".to_string(),
            || {
                buffer.clear();
            }
        ).with_throughput(WIDTH as f64 * HEIGHT as f64); // pixels per second
        
        results.push(result);
    }

    // Test scene rendering
    if CONFIG.test_scene_render {
        for (scene_name, scene) in scenes {
            let mut buffer = Buffer::new(WIDTH, HEIGHT);
            
            let result = run_timed_benchmark(
                format!("Scene Render ({scene_name})"),
                || {
                    buffer.clear(); // Ensure clean state
                    scene.render(&mut buffer);
                }
            );
            
            results.push(result);
        }
    }

    // Test buffer merge
    if CONFIG.test_buffer_merge {
        let mut buffer1 = Buffer::new(WIDTH, HEIGHT);
        let buffer2 = draw_test_pattern(WIDTH, HEIGHT);
        
        let result = run_timed_benchmark(
            "Buffer Merge".to_string(),
            || {
                buffer1.merge(&buffer2);
            }
        ).with_throughput(WIDTH as f64 * HEIGHT as f64);
        
        results.push(result);
    }

    // Test buffer to_raw conversion
    if CONFIG.test_buffer_to_raw {
        let buffer = draw_test_pattern(WIDTH, HEIGHT);
        
        let result = run_timed_benchmark(
            "Buffer to Raw".to_string(),
            || {
                let _raw = buffer.to_raw();
            }
        ).with_throughput(WIDTH as f64 * HEIGHT as f64);
        
        results.push(result);
    }

    // Test hue shifting
    if CONFIG.test_hue_shift {
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Hue Shift".to_string(),
            || {
                let _result = hue_shift(128, 64, 192, counter as f32);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test project_to_screen_space
    if CONFIG.test_project_to_screen_space {
        let buffer = Buffer::new(WIDTH, HEIGHT);
        let camera = Camera::new(
            Position { x: 0.0, y: 0.0, z: 0.0 },
            Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
            Scale { x: 1.0, y: 1.0, z: 1.0 },
        );
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Project to Screen Space".to_string(),
            || {
                let vertex = (
                    (counter as f32 % 20.0) - 10.0,  // x: -10 to 10
                    ((counter / 20) as f32 % 20.0) - 10.0,  // y: -10 to 10
                    -5.0 - (counter as f32 % 10.0)  // z: -5 to -15
                );
                
                let _screen_pos = project_to_screen_space(&vertex, &camera, &buffer);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test clip_line_to_camera_plane
    if CONFIG.test_clip_line_to_camera_plane {
        let camera = Camera::new(
            Position { x: 0.0, y: 0.0, z: 0.0 },
            Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
            Scale { x: 1.0, y: 1.0, z: 1.0 },
        );
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Clip Line to Camera Plane".to_string(),
            || {
                let v1 = (
                    (counter as f32 % 10.0) - 5.0,
                    ((counter / 10) as f32 % 10.0) - 5.0,
                    -1.0 - (counter as f32 % 20.0)
                );
                let v2 = (
                    ((counter + 1) as f32 % 10.0) - 5.0,
                    (((counter + 1) / 10) as f32 % 10.0) - 5.0,
                    -1.0 - ((counter + 1) as f32 % 20.0)
                );
                
                let _clipped = clip_line_to_camera_plane(v1, v2, &camera);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test draw_line
    if CONFIG.test_draw_line {
        let mut buffer = Buffer::new(WIDTH, HEIGHT);
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Draw Line".to_string(),
            || {
                let p1 = ((counter as f32 % 100.0), ((counter / 100) as f32 % 100.0));
                let p2 = ((counter as f32 % 100.0) + 200.0, ((counter / 100) as f32 % 100.0) + 200.0);
                let color = (255, 255, 255, 255);
                
                draw_line(&mut buffer, p1, p2, color);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test clip_line
    if CONFIG.test_clip_line {
        let bounds = (0.0, WIDTH as f32, 0.0, HEIGHT as f32);
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Clip Line".to_string(),
            || {
                let p1 = (
                    (counter as f32 % 200.0) - 50.0,  // Can be outside bounds
                    ((counter / 200) as f32 % 200.0) - 50.0
                );
                let p2 = (
                    ((counter + 100) as f32 % 200.0) - 50.0,
                    (((counter + 100) / 200) as f32 % 200.0) - 50.0
                );
                
                let _clipped = clip_line(p1, p2, bounds);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test compute_region_code
    if CONFIG.test_compute_region_code {
        let bounds = (0.0, WIDTH as f32, 0.0, HEIGHT as f32);
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Compute Region Code".to_string(),
            || {
                let x = (counter as f32 % 200.0) - 50.0;  // Can be outside bounds
                let y = ((counter / 200) as f32 % 200.0) - 50.0;
                
                let _code = compute_region_code(x, y, bounds.0, bounds.1, bounds.2, bounds.3);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test render_tri (direct triangle rendering)
    if CONFIG.test_render_tri {
        let mut buffer = Buffer::new(WIDTH, HEIGHT);
        let camera = Camera::new(
            Position { x: 0.0, y: 0.0, z: 0.0 },
            Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
            Scale { x: 1.0, y: 1.0, z: 1.0 },
        );
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Render Triangle".to_string(),
            || {
                let triangle: Triangle = [
                    ((counter as f32 % 10.0) - 5.0, ((counter / 10) as f32 % 10.0) - 5.0, -5.0),
                    ((counter as f32 % 10.0) - 3.0, ((counter / 10) as f32 % 10.0) - 3.0, -5.0),
                    ((counter as f32 % 10.0) - 4.0, ((counter / 10) as f32 % 10.0) - 2.0, -5.0),
                ];
                let colors = [(255, 128, 64, 255), (64, 255, 128, 255), (128, 64, 255, 255)];
                
                render_tri(&mut buffer, &camera, &triangle, colors);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    use nalgebra::Vector4;

    // Test project_world_to_view
    if CONFIG.test_project_world_to_view {
        let camera = Camera::new(
            Position { x: 0.0, y: 0.0, z: 0.0 },
            Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
            Scale { x: 1.0, y: 1.0, z: 1.0 },
        );
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Project World to View".to_string(),
            || {
                let point = Vector4::new(
                    (counter as f32 % 20.0) - 10.0,
                    ((counter / 20) as f32 % 20.0) - 10.0,
                    -5.0 - (counter as f32 % 10.0),
                    1.0
                );
                
                let _view_point = project_world_to_view(&point, &camera);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test project_view_to_clip
    if CONFIG.test_project_view_to_clip {
        let buffer = Buffer::new(WIDTH, HEIGHT);
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Project View to Clip".to_string(),
            || {
                let point = Vector4::new(
                    (counter as f32 % 20.0) - 10.0,
                    ((counter / 20) as f32 % 20.0) - 10.0,
                    -5.0 - (counter as f32 % 10.0),
                    1.0
                );
                
                let _clip_point = project_view_to_clip(&point, &buffer);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test project_clip_to_ndc
    if CONFIG.test_project_clip_to_ndc {
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Project Clip to NDC".to_string(),
            || {
                let point = Vector4::new(
                    (counter as f32 % 20.0) - 10.0,
                    ((counter / 20) as f32 % 20.0) - 10.0,
                    -5.0 - (counter as f32 % 10.0),
                    2.0 + (counter as f32 % 3.0) // Non-zero w for perspective division
                );
                
                let _ndc_point = project_clip_to_ndc(&point);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test project_ndc_to_screen
    if CONFIG.test_project_ndc_to_screen {
        let buffer = Buffer::new(WIDTH, HEIGHT);
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Project NDC to Screen".to_string(),
            || {
                let point = Vector4::new(
                    ((counter as f32 % 100.0) / 50.0) - 1.0, // NDC range [-1, 1]
                    (((counter / 100) as f32 % 100.0) / 50.0) - 1.0,
                    (counter as f32 % 100.0) / 100.0, // depth [0, 1]
                    1.0
                );
                
                let _screen_point = project_ndc_to_screen(&point, &buffer);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test is_backface
    if CONFIG.test_is_backface {
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Backface Culling".to_string(),
            || {
                let triangle = [
                    Vector4::new((counter as f32 % 10.0) - 5.0, ((counter / 10) as f32 % 10.0) - 5.0, -5.0, 1.0),
                    Vector4::new((counter as f32 % 10.0) - 3.0, ((counter / 10) as f32 % 10.0) - 3.0, -5.0, 1.0),
                    Vector4::new((counter as f32 % 10.0) - 4.0, ((counter / 10) as f32 % 10.0) - 2.0, -5.0, 1.0),
                ];
                
                let _is_back = is_backface(&triangle);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test clip_triangle_against_near_plane
    if CONFIG.test_clip_triangle_against_near_plane {
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Triangle Near Plane Clipping".to_string(),
            || {
                let triangle = [
                    Vector4::new((counter as f32 % 10.0) - 5.0, ((counter / 10) as f32 % 10.0) - 5.0, -0.1 - (counter as f32 % 5.0), 1.0),
                    Vector4::new((counter as f32 % 10.0) - 3.0, ((counter / 10) as f32 % 10.0) - 3.0, -0.1 - (counter as f32 % 5.0), 1.0),
                    Vector4::new((counter as f32 % 10.0) - 4.0, ((counter / 10) as f32 % 10.0) - 2.0, -0.1 - (counter as f32 % 5.0), 1.0),
                ];
                
                let _clipped = clip_triangle_against_near_plane(&triangle);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test is_fully_outside_clip_space
    if CONFIG.test_is_fully_outside_clip_space {
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Clip Space Frustum Culling".to_string(),
            || {
                let triangle = [
                    Vector4::new((counter as f32 % 20.0) - 10.0, ((counter / 20) as f32 % 20.0) - 10.0, -5.0, 2.0),
                    Vector4::new((counter as f32 % 20.0) - 8.0, ((counter / 20) as f32 % 20.0) - 8.0, -5.0, 2.0),
                    Vector4::new((counter as f32 % 20.0) - 9.0, ((counter / 20) as f32 % 20.0) - 7.0, -5.0, 2.0),
                ];
                
                let _outside = is_fully_outside_clip_space(triangle);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test is_fully_outside_ndc
    if CONFIG.test_is_fully_outside_ndc {
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "NDC Space Frustum Culling".to_string(),
            || {
                let triangle = [
                    Vector4::new(((counter as f32 % 100.0) / 25.0) - 2.0, (((counter / 100) as f32 % 100.0) / 25.0) - 2.0, 0.5, 1.0),
                    Vector4::new(((counter as f32 % 100.0) / 25.0) - 1.8, (((counter / 100) as f32 % 100.0) / 25.0) - 1.8, 0.5, 1.0),
                    Vector4::new(((counter as f32 % 100.0) / 25.0) - 1.9, (((counter / 100) as f32 % 100.0) / 25.0) - 1.7, 0.5, 1.0),
                ];
                
                let _outside = is_fully_outside_ndc(&triangle);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test random_color
    if CONFIG.test_random_color {
        let result = run_timed_benchmark(
            "Random Color Generation".to_string(),
            || {
                let _color = random_color();
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test random_color_seeded
    if CONFIG.test_random_color_seeded {
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Seeded Random Color Generation".to_string(),
            || {
                let _color = random_color_seeded(counter as u64);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test rgb_to_hsv
    if CONFIG.test_rgb_to_hsv {
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "RGB to HSV Conversion".to_string(),
            || {
                let r = (counter % 256) as u8;
                let g = ((counter / 256) % 256) as u8;
                let b = ((counter / 65536) % 256) as u8;
                
                let _hsv = rgb_to_hsv(r, g, b);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test hsv_to_rgb
    if CONFIG.test_hsv_to_rgb {
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "HSV to RGB Conversion".to_string(),
            || {
                let h = counter as f32 % 360.0;
                let s = ((counter % 100) as f32) / 100.0;
                let v = (((counter / 100) % 100) as f32) / 100.0;
                
                let _rgb = hsv_to_rgb(h, s, v);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    results
}

fn benchmark_scene_complexity() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    let object_counts = [1, 5, 10, 20, 50];

    println!("  {BRIGHT_BLUE}Testing scene complexity scaling...{RESET}");

    for &count in &object_counts {
        let scene = create_scene_with_objects(count);
        let mut buffer = Buffer::new(WIDTH, HEIGHT);

        let result = run_timed_benchmark(
            format!("Scene with {count} Objects"),
            || {
                buffer.clear();
                scene.render(&mut buffer);
            }
        ).with_throughput(count as f64); // objects per second

        results.push(result);
    }

    results
}

fn benchmark_buffer_operations() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // Test pixel drawing performance
    {
        let mut buffer = Buffer::new(WIDTH, HEIGHT);
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Pixel Drawing".to_string(),
            || {
                let x = counter % WIDTH;
                let y = (counter / WIDTH) % HEIGHT;
                buffer.draw_pixel(x, y, (255, 128, 64, 255), 1.0);
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    // Test line drawing performance
    {
        let mut buffer = Buffer::new(WIDTH, HEIGHT);
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Line Drawing".to_string(),
            || {
                draw_line(&mut buffer, 
                    (0.0, counter as f32 % HEIGHT as f32), 
                    (WIDTH as f32, (counter * 2) as f32 % HEIGHT as f32), 
                    (255, 255, 255, 255)
                );
                counter += 1;
            }
        ).with_throughput(1.0);
        
        results.push(result);
    }

    results
}

fn benchmark_rendering_techniques() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // Test gradient generation
    {
        let mut counter = 0;
        
        let result = run_timed_benchmark(
            "Gradient Generation".to_string(),
            || {
                let buffer = Buffer::new(WIDTH, HEIGHT);
                let _gradient = draw_frame_gradient(&buffer, counter);
                counter += 1;
            }
        ).with_throughput(WIDTH as f64 * HEIGHT as f64);
        
        results.push(result);
    }

    // Test animated rendering
    {
        let mut scene = create_scene_with_objects(5);
        let mut buffer = Buffer::new(WIDTH, HEIGHT);
        let mut frame = 0;
        
        let result = run_timed_benchmark(
            "Animated Rendering".to_string(),
            || {
                // Apply animation
                let frequency = 0.02;
                let amplitude = 1.5;
                let offset1 = (frame as f32 * frequency).sin() * amplitude;
                let offset2 = (frame as f32 * frequency * 1.5).cos() * amplitude;
                let offset3 = (frame as f32 * frequency * 2.0).sin() * amplitude;

                scene.objects[0].set_position(Position {
                    x: offset1,
                    y: offset2,
                    z: -5.0 + offset3,
                });

                buffer.clear();
                scene.render(&mut buffer);
                frame += 1;
            }
        );
        
        results.push(result);
    }

    results
}

fn create_scene_with_objects(count: usize) -> Scene {
    let camera = Camera::new(
        Position { x: 0.0, y: 0.0, z: 0.0 },
        Rotation { pitch: 0.0, yaw: 0.0, roll: 0.0 },
        Scale { x: 1.0, y: 1.0, z: 1.0 },
    );

    let mut objects: Vec<Box<dyn Object>> = Vec::new();
    for i in 0..count {
        let cube = RectangularPrism::new_cube(
            Transform {
                position: Position { 
                    x: (i as f32 - count as f32 / 2.0) * 2.5, 
                    y: ((i % 3) as f32 - 1.0) * 2.0,
                    z: -5.0 - (i / 3) as f32 * 1.5
                },
                rotation: Rotation { 
                    pitch: i as f32 * 0.1, 
                    yaw: i as f32 * 0.2, 
                    roll: 0.0 
                },
                scale: Scale { x: 1.0, y: 1.0, z: 1.0 },
            },
            1.0
        );
        objects.push(Box::new(cube));
    }

    Scene::new(camera, objects)
}

fn draw_test_pattern(width: usize, height: usize) -> Buffer {
    let mut buffer = Buffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = ((x + y) * 128 / (width + height)) as u8;
            buffer.draw_pixel(x, y, (r, g, b, 255), 1.0);
        }
    }
    buffer
}

fn print_benchmark_summary(results: &[BenchmarkResult]) {
    if CONFIG.simple_output {
        println!("\n{BOLD}{BRIGHT_CYAN}> BENCHMARK RESULTS{RESET}");
        println!("{BLUE}=================={RESET}");
        for result in results {
            result.print_results(true);
        }
    } else {
        println!("\n{BOLD}{BRIGHT_CYAN}> BENCHMARK SUMMARY{RESET}");
        println!("{BLUE}==============================================={RESET}");

        for result in results {
            result.print_results(false);
        }
        
        // Performance insights
        let scene_results: Vec<_> = results.iter()
            .filter(|r| r.name.contains("Scene Render"))
            .collect();
        
        if scene_results.len() >= 2 {
            let small = scene_results.iter().find(|r| r.name.contains("Small"));
            let complex = scene_results.iter().find(|r| r.name.contains("Complex"));
            
            if let (Some(small), Some(complex)) = (small, complex) {
                let overhead = complex.mean.as_secs_f64() / small.mean.as_secs_f64();
                println!("\n{BOLD}{BRIGHT_YELLOW}> Performance Insight{RESET}: Complex scenes are {BRIGHT_YELLOW}{:.1}x{RESET} slower than small scenes",
                    overhead);
            }
        }
    }

    println!("\n{BOLD}{BRIGHT_GREEN}> Benchmark Complete!{RESET}");
}

fn draw_frame_gradient(buffer: &Buffer, frame: usize) -> Buffer {
    // Same gradient function as in your test.rs
    let mut data = Buffer::new(buffer.width, buffer.height);
    for y in 0..buffer.height {
        for x in 0..buffer.width {
            let r = ((x * 255) / buffer.width) as u8;
            let g = ((y * 255) / buffer.height) as u8;
            let b = 128;
            let frame_shift = ((frame * 2) % 360) as f32;
            let (r, g, b) = hue_shift(r/2, g/2, b/2, frame_shift);
            data.draw_pixel(x, y, (r, g, b, 255), 100.0);
        }
    }
    data
}
