#![allow(unused)]

use cgmath::{Quaternion, Rotation3, Rad, Euler, Deg};
use slime_engine::{
    App, Engine, WindowOptions, core::GraphicsContext, env_logger, input::Key, model::{Material, MaterialTextures, MaterialUniforms, Model, ModelVertex}, node::Node3D, pollster::block_on, primitives::Primitive, scene::{CameraId, ModelId, NodeId}, transform::Transform3D, window::Window,
};
use std::time::Duration;
use noise::{NoiseFn, Perlin, Seedable};

const CHUNK_SIZE: usize = 32; // x/z width of a chunk
const RENDER_DISTANCE: usize = 8; // render distance (square side length)
const RENDER_HEIGHT: usize = 4; // how many chunks high to generate

fn rand_color() -> [f32; 4] {
    [rand::random(), rand::random(), rand::random(), 1.0]
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
    blocks: [[[Block; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    is_empty: bool, // if all blocks are air
}

impl Chunk {
    fn new(position: [usize; 3]) -> Self {
        let blocks = [[[Block::new(); CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];

        let mut chunk = Self {
            position,
            blocks,
            is_empty: true,
        };
        chunk.generate_terrain();
        chunk
    }

    fn generate_terrain(&mut self) {
        let perlin = Perlin::new(42);

        // density grid (cells of 4 horizontally, 8 vertically)
        const DENSITY_WIDTH: usize = CHUNK_SIZE / 4 + 1;
        const DENSITY_HEIGHT: usize = CHUNK_SIZE / 8 + 1;
        let mut density_grid = [[[0.0; DENSITY_WIDTH]; DENSITY_HEIGHT]; DENSITY_WIDTH];

        // calculate density grid values
        for (x, density_x) in density_grid.iter_mut().enumerate() {
            for (y, density_y) in density_x.iter_mut().enumerate() {
                for (z, density) in density_y.iter_mut().enumerate() {
                    // global coordinates
                    let gx = (x as f64 * 4.0) + (self.position[0] * CHUNK_SIZE) as f64;
                    let gy = (y as f64 * 8.0) + (self.position[1] * CHUNK_SIZE) as f64;
                    let gz = (z as f64 * 4.0) + (self.position[2] * CHUNK_SIZE) as f64;

                    let frequency = 0.03;
                    let value = perlin.get([
                        gx * frequency,
                        gy * frequency,
                        gz * frequency,
                    ]);

                    let sea_level = 63.0;
                    let ground_level = 66.0;
                    let k = 0.05;
                    *density = value - (gy - ground_level) * k;
                }
            }
        }

        let mut solid_blocks = 0;
        for i in 0..(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) {
            let x = i % CHUNK_SIZE;
            let y = (i / CHUNK_SIZE) % CHUNK_SIZE;
            let z = i / (CHUNK_SIZE * CHUNK_SIZE);

            // global coordinates
            let gx = x as f64 + (self.position[0] * CHUNK_SIZE) as f64;
            let gy = y as f64 + (self.position[1] * CHUNK_SIZE) as f64;
            let gz = z as f64 + (self.position[2] * CHUNK_SIZE) as f64;

            // which density grid cell
            let cx = x / 4;
            let cy = y / 8;
            let cz = z / 4;

            // position inside density grid cell
            let dx = (x % 4) as f64 / 4.0;
            let dy = (y % 8) as f64 / 8.0;
            let dz = (z % 4) as f64 / 4.0;

            // grab density grid corners
            let c000 = density_grid[cx  ][cy  ][cz  ];
            let c001 = density_grid[cx  ][cy  ][cz+1];
            let c010 = density_grid[cx  ][cy+1][cz  ];
            let c011 = density_grid[cx  ][cy+1][cz+1];
            let c100 = density_grid[cx+1][cy  ][cz  ];
            let c101 = density_grid[cx+1][cy  ][cz+1];
            let c110 = density_grid[cx+1][cy+1][cz  ];
            let c111 = density_grid[cx+1][cy+1][cz+1];

            // liner interpolation
            fn lerp(a: f64, b: f64, t: f64) -> f64 {
                a + (b - a) * t
            }

            // trilinear interpolation
            let x00 = lerp(c000, c100, dx);
            let x10 = lerp(c010, c110, dx);
            let x01 = lerp(c001, c101, dx);
            let x11 = lerp(c011, c111, dx);
            let z0 = lerp(x00, x10, dy);
            let z1 = lerp(x01, x11, dy);
            let density = lerp(z0, z1, dz);

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

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
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
            [-10.0, 20.0, -10.0],
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
                                .with_position([(x * CHUNK_SIZE) as f32, (y * CHUNK_SIZE) as f32, (z * CHUNK_SIZE) as f32])
                        ).with_color(rand_color())
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
