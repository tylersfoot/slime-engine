#![allow(unused)]

use cgmath::{Quaternion, Rotation3, Rad, Euler, Deg};
use slime_engine::{
    App, Engine, WindowOptions, core::GraphicsContext, env_logger, input::Key, model::{Material, MaterialTextures, MaterialUniforms, Model, ModelVertex}, node::Node3D, pollster::block_on, primitives::Primitive, scene::{CameraId, ModelId, NodeId}, transform::Transform3D, window::Window,
};
use std::time::Duration;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, Seedable};
use rand::{Rng, SeedableRng, rngs::StdRng, RngExt};

const CHUNK_SIZE_XZ: usize = 16; // x/z width of a chunk (must be divisible by 4)
const CHUNK_SIZE_Y: usize = 128; // y height of a chunk (must be divisible by 8)
const RENDER_DISTANCE: usize = 8; // render distance (square side length)
const RENDER_HEIGHT: usize = 1; // how many chunks high to generate
const SEED: u32 = 42;

// helper functions
fn rand_color() -> [f32; 4] {
    [rand::random(), rand::random(), rand::random(), 1.0]
}
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}
fn fade(t: f64) -> f64 {
    t*t*t*(t*(t*6.0 - 15.0) + 10.0)
}

struct PerlinNoiseGenerator {
    // holds random shuffled ints
    table: [i32; 512],
    // random offsets added to every coordinate
    offset_x: f64,
    offset_y: f64,
    offset_z: f64,
}

impl PerlinNoiseGenerator {
    // https://github.com/Spottedleaf/OldGenerator/blob/master/src/main/java/ca/spottedleaf/oldgenerator/generator/b173/noise/NoiseGeneratorPerlin173.java
    // keeping some variable numbers for now for ease of porting (var1 -> v1)
    fn new(rng: &mut StdRng) -> Self {
        let mut table = [0; 512];
        // random offsets
        let offset_x = rng.random::<f64>() * 256.0_f64;
        let offset_y = rng.random::<f64>() * 256.0_f64;
        let offset_z = rng.random::<f64>() * 256.0_f64;

        // fill first 256 with 0-255
        for (i, val) in (0..256).enumerate() {
            table[i] = val;
        }

        // fill first 256 with 0-255
        let mut v2 = 0;
        while v2 < 256 {
            // pick random idx >= v2
            let v3 = rng.random_range(0..(256 - v2)) + v2;
            let v4 = table[v2];
            table[v3] = v4;
            table[v2 + 256] = table[v2]; // duplicate into upper half
            v2 += 1;
        }
        
        Self { table, offset_x, offset_y, offset_z }
    }

    fn gradient3(hash: i32, x: f64, y: f64, z: f64) -> f64 {
        // computes gradient dot product
        let h: i32 = hash & 15;
        let v9: f64 =  if (h < 8) {x} else {y};
        let v11: f64 =  if (h < 4) {y} else {
            if (h != 12 && h != 14) {z} else {x}
        };
        let a: f64 = if (h & 1) == 0 {v9} else {-v9};
        let b: f64 = if (h & 2) == 0 {v11} else {-v11};
        a + b  
    }
}

#[derive(Clone, Debug, Copy, PartialEq)]
enum BlockType {
    Air,
    Grass,
    Dirt,
    Stone,
    Water,
    Ice,
}

#[derive(Clone, Debug, Copy)]
struct Block {
    block_type: BlockType,
}

impl Block {
    fn new() -> Self {
        Self {
            block_type: BlockType::Air,
        }
    }
}


#[derive(Clone, Debug, Copy)]
struct Chunk {
    position: [usize; 3], // chunk position so *CHUNK_SIZE
    blocks: [[[Block; CHUNK_SIZE_XZ]; CHUNK_SIZE_Y]; CHUNK_SIZE_XZ],
    is_empty: bool, // if all blocks are air
}

impl Chunk {
    fn new(position: [usize; 3]) -> Self {
        let blocks = [[[Block::new(); CHUNK_SIZE_XZ]; CHUNK_SIZE_Y]; CHUNK_SIZE_XZ];

        let mut chunk = Self {
            position,
            blocks,
            is_empty: true,
        };
        chunk.generate_terrain();
        chunk
    }

    fn generate_terrain(&mut self) {
        let random = StdRng::seed_from_u64(SEED as u64);

        let noise_min_limit   = Fbm::<Perlin>::new(SEED+1).set_octaves(16); // 'low' density bound
        let noise_max_limit    = Fbm::<Perlin>::new(SEED+2).set_octaves(16); // 'high' density bound
        let noise_selector    = Fbm::<Perlin>::new(SEED+3).set_octaves(8); // lerp weight between low/high
        let noise_beach       = Fbm::<Perlin>::new(SEED+4).set_octaves(4); // sand/gravel patches
        let noise_surface_depth = Fbm::<Perlin>::new(SEED+5).set_octaves(4); // how deep dirt/sand layer is
        let noise_scale       = Fbm::<Perlin>::new(SEED+6).set_octaves(10); // horizontal 'stretch' of terrain
        let noise_depth       = Fbm::<Perlin>::new(SEED+7).set_octaves(16); // base elevation/depth
        let noise_tree_count   = Fbm::<Perlin>::new(SEED+8).set_octaves(8); // tree density in populate

        // how many density CELLS per chunk
        const DENSITY_CELLS_XZ: usize = 4;
        const DENSITY_CELLS_Y: usize = 8;
        // how many density POINTS per chunk
        const DENSITY_SAMPLES_XZ: usize = CHUNK_SIZE_XZ / DENSITY_CELLS_XZ + 1;
        const DENSITY_SAMPLES_Y: usize = CHUNK_SIZE_Y / DENSITY_CELLS_Y + 1;
        let mut density_grid = [[[0.0; DENSITY_SAMPLES_XZ]; DENSITY_SAMPLES_Y]; DENSITY_SAMPLES_XZ];
        // const NOISE_SCALE_XZ: f64 = 684.412;
        // const NOISE_SCALE_Y: f64 = 684.412;
        const NOISE_SCALE_XZ: f64 = 0.5;
        const NOISE_SCALE_Y:  f64 = 0.5;
        const SEA_LEVEL: f64 = 64.0;

        // calculate density grid values
        for (x, density_x) in density_grid.iter_mut().enumerate() {
            for (y, density_y) in density_x.iter_mut().enumerate() {
                for (z, density_value) in density_y.iter_mut().enumerate() {
                    // global coordinates
                    let gx = (x as f64 * DENSITY_CELLS_XZ as f64) + (self.position[0] * CHUNK_SIZE_XZ) as f64;
                    let gy = (y as f64 * DENSITY_CELLS_Y as f64) + (self.position[1] * CHUNK_SIZE_Y) as f64;
                    let gz = (z as f64 * DENSITY_CELLS_XZ as f64) + (self.position[2] * CHUNK_SIZE_XZ) as f64;

                    // 2d fields (per column) - control terrain shape
                    let n_scale = noise_scale.get([
                        gx * 1.121_f64,
                        10.0_f64,
                        gz * 1.121_f64,
                    ]);
                    let n_depth = noise_depth.get([
                        gx * 200.0_f64,
                        10.0_f64,
                        gz * 200.0_f64,
                    ]);

                    // 3d fields (the density) - min/max's frequencies overflow at farlands here
                    let n_selector  = noise_selector.get([
                        gx * NOISE_SCALE_XZ * 0.5,
                        gy * NOISE_SCALE_Y * 0.5,
                        gz * NOISE_SCALE_XZ * 0.5,
                        // gx * (NOISE_SCALE_XZ / 80.0_f64),
                        // gy * (NOISE_SCALE_Y / 160.0_f64),
                        // gz * (NOISE_SCALE_XZ / 80.0_f64),
                    ]);
                    let n_min_limit = noise_min_limit.get([
                        gx * NOISE_SCALE_XZ,
                        gy * NOISE_SCALE_Y,
                        gz * NOISE_SCALE_XZ,
                    ]);
                    let n_max_limit = noise_max_limit.get([
                        gx * NOISE_SCALE_XZ,
                        gy * NOISE_SCALE_Y,
                        gz * NOISE_SCALE_XZ,
                    ]);


                    // ---- PER-COLUMN NOISE ----

                    // TODO humidity/temp

                    // horizontal stretch - compresses vertical density gradient
                    let mut horizontal_stretch = (n_scale + 256.0_f64) / 512.0_f64;
                    horizontal_stretch = horizontal_stretch.min(1.0_f64);

                    // elevation/depth
                    let mut depth = n_depth / 1.0_f64; // 8000.0_f64
                    if depth < 0.0_f64 { depth = -depth * 0.3_f64 }
                    depth = depth * 3.0_f64 - 2.0_f64;
                    if depth < 0.0_f64 {
                        depth /= 2.0_f64;
                        depth = depth.max(-1.0_f64);
                        depth /= 1.4_f64;
                        depth /= 2.0_f64;
                        horizontal_stretch = 0.0_f64; // deep oceans get flattened
                    } else {
                        depth = depth.min(1.0_f64);
                        depth /= 8.0_f64;
                    }
                    horizontal_stretch = horizontal_stretch.max(0.0_f64);
                    horizontal_stretch += 0.5_f64;

                    depth = depth * (DENSITY_SAMPLES_Y as f64) / 16.0_f64;
                    // the Y (in samples) where density crosses zero
                    let center_height = (DENSITY_SAMPLES_Y as f64) / 2.0_f64 + depth * 4.0_f64;

                    // ---- PER-POINT NOISE ----

                    // distance of this sample from center_height, scaled by stretch
                    // below center is multiplied by 4 to fill underground
                    let mut vertical_falloff = (y as f64 - center_height) * 12.0_f64 / horizontal_stretch;
                    if vertical_falloff < 0.0_f64 { vertical_falloff *= 4.0_f64 }

                    let min_density = n_min_limit / 1.0_f64; // 512.0_f64
                    let max_density = n_max_limit / 1.0_f64; // 512.0_f64
                    let selector = (n_selector / 10.0_f64 + 1.0_f64) / 2.0_f64;
                    let mut density = lerp(
                        min_density,
                        max_density,
                        selector.clamp(0.0_f64, 1.0_f64)
                    ) - vertical_falloff;

                    // force top 3 sample layers toward -10 so nothing generates at world height
                    if y > DENSITY_SAMPLES_Y - 4 {
                        let t = ((y - (DENSITY_SAMPLES_Y - 4)) as f32 / 3.0_f32) as f64;
                        density = density * (1.0_f64 - t) + (-10.0_f64) * t;
                    }

                    *density_value = density;
                }
            }
        }


        // carve base terrain

        let mut solid_blocks = 0;
        for i in 0..(CHUNK_SIZE_XZ * CHUNK_SIZE_Y * CHUNK_SIZE_XZ) {
            let x = i % CHUNK_SIZE_XZ;
            let y = (i / CHUNK_SIZE_XZ) % CHUNK_SIZE_Y;
            let z = i / (CHUNK_SIZE_XZ * CHUNK_SIZE_Y);

            // global coordinates
            let gx = x as f64 + (self.position[0] * CHUNK_SIZE_XZ) as f64;
            let gy = y as f64 + (self.position[1] * CHUNK_SIZE_Y) as f64;
            let gz = z as f64 + (self.position[2] * CHUNK_SIZE_XZ) as f64;

            // which density grid cell
            let cx = x / DENSITY_CELLS_XZ;
            let cy = y / DENSITY_CELLS_Y;
            let cz = z / DENSITY_CELLS_XZ;

            // position inside density grid cell
            let dx = (x % DENSITY_CELLS_XZ) as f64 / DENSITY_CELLS_XZ as f64;
            let dy = (y % DENSITY_CELLS_Y) as f64 / DENSITY_CELLS_Y as f64;
            let dz = (z % DENSITY_CELLS_XZ) as f64 / DENSITY_CELLS_XZ as f64;

            // grab density grid corners
            let c000 = density_grid[cx  ][cy  ][cz  ];
            let c001 = density_grid[cx  ][cy  ][cz+1];
            let c010 = density_grid[cx  ][cy+1][cz  ];
            let c011 = density_grid[cx  ][cy+1][cz+1];
            let c100 = density_grid[cx+1][cy  ][cz  ];
            let c101 = density_grid[cx+1][cy  ][cz+1];
            let c110 = density_grid[cx+1][cy+1][cz  ];
            let c111 = density_grid[cx+1][cy+1][cz+1];

            // trilinear interpolation
            let x00 = lerp(c000, c100, dx);
            let x10 = lerp(c010, c110, dx);
            let x01 = lerp(c001, c101, dx);
            let x11 = lerp(c011, c111, dx);
            let y0 = lerp(x00, x10, dy);
            let y1 = lerp(x01, x11, dy);
            let density = lerp(y0, y1, dz);

            if density > 0.0 {
                solid_blocks += 1;
                self.blocks[x][y][z] = Block {
                    block_type: BlockType::Stone,
                }
            }
        }

        // safeguard against empty chunks
        self.is_empty = solid_blocks < 1;

    }

    fn generate_model(&self, gfx: &GraphicsContext, layout: &wgpu::BindGroupLayout) -> Model {
        // generates cube data for each block and compiles into one Model for the chunk
        let device = &gfx.device;
        let queue = &gfx.queue;
        let color = [0.4, 0.4, 0.4];

        let mut vertices = Vec::new();
        let mut indices: Vec<i32> = Vec::new();

        // local cube data
        let face_data = [
            // (normal, [corner; 4])
            ([ 0.0,  0.0,  1.0], [[-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]]), // front  (+Z)
            ([ 0.0,  0.0, -1.0], [[ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5]]), // back   (-Z)
            ([ 1.0,  0.0,  0.0], [[ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5]]), // right  (+X)
            ([-1.0,  0.0,  0.0], [[-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5]]), // left   (-X)
            ([ 0.0,  1.0,  0.0], [[-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5]]), // top    (+Y)
            ([ 0.0, -1.0,  0.0], [[-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5]]), // bottom (-Y)
        ];
        let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

        for x in 0..CHUNK_SIZE_XZ {
            for y in 0..CHUNK_SIZE_Y {
                for z in 0..CHUNK_SIZE_XZ {
                    // dont render air
                    if self.blocks[x][y][z].block_type == BlockType::Air {
                        continue;
                    }

                    let (fx, fy, fz) = (x as f32, y as f32, z as f32);
                    for (normal, corners) in face_data {
                        // index of first vertex of THIS face
                        let base = vertices.len() as i32;
                        
                        for i in 0..4 {
                            vertices.push(ModelVertex {
                                // position: corners[i],
                                position: [
                                    corners[i][0] + fx,
                                    corners[i][1] + fy,
                                    corners[i][2] + fz,
                                ],
                                tex_coords: uvs[i],
                                normal,
                            });
                        }

                        indices.extend_from_slice(&[
                            base, base + 1, base + 2,
                            base, base + 2, base + 3,
                        ]);
                    }
                }
            }
        }

        let textures = MaterialTextures::default(device, queue).unwrap();
        let uniforms = MaterialUniforms {
            diffuse_color: color,
            ..Default::default()
        };
        let material = Material::new(device, "cube_material", textures, layout, uniforms);
        Model::from_raw(device, "cube", &vertices, &indices, material)

        // for (z, layer_z) in self.blocks.iter().enumerate() {
        //     for (y, row_y) in layer_z.iter().enumerate() {
        //         for (x, block) in row_y.iter().enumerate() {
        //             let block_color = match block.block_type {
        //                 BlockType::Dirt => [0.5, 0.3, 0.1, 1.0],
        //                 BlockType::Grass => [0.3, 0.7, 0.3, 1.0],
        //                 BlockType::Stone => [0.4, 0.4, 0.4, 1.0],
        //                 BlockType::Air => [0.0, 0.0, 0.0, 0.0],
        //             };
                    
        //             if block.block_type != BlockType::Air {
        //                 engine.scene.spawn_node(
        //                     Node3D::new(Some(cube_model)).with_transform(
        //                         Transform3D::new()
        //                             .with_position([
        //                                 (x + self.position[0] * CHUNK_SIZE) as f32,
        //                                 (y + self.position[1] * CHUNK_SIZE) as f32,
        //                                 (z + self.position[2] * CHUNK_SIZE) as f32,
        //                                 ])
        //                             .with_scale([1.0, 1.0, 1.0])
        //                     ).with_color(block_color)
        //                 );
        //             }
        //         }
        //     }
        // }
    }
}

fn generate_bare_terrain() {
    
}

struct ExampleScene {
    time_passed: f32,
    camera: Option<CameraId>,
    cube_model: Option<ModelId>,
}

impl App for ExampleScene {
    fn start(&mut self, engine: &mut Engine) {
        let gfx = &engine.gfx;
        let layout = &engine.renderer.texture_bind_group_layout;

        let camera = engine.scene.spawn_camera(
            [-10.0, 100.0, -10.0],
            45.0,
            -20.0
        );
        self.camera = Some(camera);

        let cube_model = engine.scene.load_primitive(Primitive::Cube, &engine.gfx, &engine.renderer);
        self.cube_model = Some(cube_model);

        // generate a bunch of chunks
        for x in 0..RENDER_DISTANCE {
            for y in 0..RENDER_HEIGHT {
                for z in 0..RENDER_DISTANCE {
                    let mut chunk = Chunk::new([x, y, z]);
                    if chunk.is_empty {
                        continue;
                    }
                    let chunk_model = chunk.generate_model(gfx, layout);
                    let chunk_model_id = engine.scene.load_model(chunk_model, &engine.gfx);
                    engine.scene.spawn_node(
                        Node3D::new(Some(chunk_model_id)).with_transform(
                            Transform3D::new()
                                .with_position([(x * CHUNK_SIZE_XZ) as f32, (y * CHUNK_SIZE_Y) as f32, (z * CHUNK_SIZE_XZ) as f32])
                        )
                        // .with_color(rand_color())
                    );
                }
            }
        }

        // let mut chunk00 = Chunk::new([0, 0, 0]);
        // let chunk00_model = chunk00.generate_model(gfx, layout);
        // let chunk00_model_id = engine.scene.load_model(chunk00_model, &engine.gfx);
        // engine.scene.spawn_node(
        //     Node3D::new(Some(chunk00_model_id)).with_transform(
        //         Transform3D::new()
        //             .with_position([0.0, 0.0, 0.0])
        //     )
        // );

        // let mut chunk01 = Chunk::new([0, 0, 1]);
        // let chunk01_model_id = engine.scene.load_model(chunk01.generate_model(gfx, layout), &engine.gfx);
        // let mut chunk10 = Chunk::new([1, 0, 0]);
        // let chunk10_model_id = engine.scene.load_model(chunk10.generate_model(gfx, layout), &engine.gfx);
        // let mut chunk11 = Chunk::new([1, 0, 1]);
        // let chunk11_model_id = engine.scene.load_model(chunk11.generate_model(gfx, layout), &engine.gfx);

    }

    fn update(&mut self, engine: &mut Engine, dt: Duration) {
        self.time_passed += dt.as_secs_f32();
        let delta = (dt.as_secs_f64() as f32).max(1e-6);

    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut window = Window::new(
        "voxel-engine",
        1000,
        800,
        WindowOptions {
            resize: true,
            ..Default::default()
        },
    ).unwrap_or_else(|e| panic!("{}", e));
    window.set_target_fps(0);
    let engine = block_on(Engine::new(window));

    let program = ExampleScene {
        time_passed: 0.0,
        camera: None,
        cube_model: None,
    };

    engine.run(program);
}
