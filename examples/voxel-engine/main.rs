#![allow(unused)]

use cgmath::{Quaternion, Rotation3, Rad, Euler, Deg};
use slime_engine::{
    App, Engine, WindowOptions, core::GraphicsContext, env_logger, input::Key, model::{Material, MaterialTextures, MaterialUniforms, Model, ModelVertex}, node::Node3D, pollster::block_on, primitives::Primitive, scene::{CameraId, ModelId, NodeId}, transform::Transform3D, window::Window,
};
use std::time::Duration;

fn rand_color() -> [f32; 4] {
    [rand::random(), rand::random(), rand::random(), 1.0]
}

#[derive(Clone, Debug, Copy, PartialEq)]
enum BlockType {
    Air,
    Grass,
    Dirt,
    Stone,
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

const CHUNK_SIZE: usize = 16;
#[derive(Clone, Debug, Copy)]
struct Chunk {
    position: [usize; 3], // chunk position so *CHUNK_SIZE
    blocks: [[[Block; 16]; 16]; 16],
}

impl Chunk {
    fn new(position: [usize; 3]) -> Self {
        let blocks = [[[Block::new(); 16]; 16]; 16];

        let mut chunk = Self {
            position,
            blocks,
        };
        chunk.generate_terrain();
        chunk
    }

    fn generate_terrain(&mut self) {
        let mut solid_blocks = 0;
        for i in 0..(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) {
            let x = i % CHUNK_SIZE;
            let y = (i / CHUNK_SIZE) % CHUNK_SIZE;
            let z = i / (CHUNK_SIZE * CHUNK_SIZE);

            let fx = x as f32;
            let fy = y as f32;
            let fz = z as f32;

            // Global coordinates
            let global_fx = fx + (self.position[0] * CHUNK_SIZE) as f32;
            let global_fy = fy + (self.position[1] * CHUNK_SIZE) as f32;
            let global_fz = fz + (self.position[2] * CHUNK_SIZE) as f32;

            // ==========================================
            // 1. GLOBAL TERRAIN HEIGHT
            // ==========================================
            let mut height_offset = 0.0;
            height_offset += (global_fx * 0.04).sin() * 4.0;
            height_offset += (global_fz * 0.05).cos() * 4.0;
            height_offset += ((global_fx + global_fz) * 0.12).sin() * 1.5;
            height_offset -= ((global_fx * 0.25).cos() + (global_fz * 0.25).sin()).abs() * 2.0;
            height_offset += (global_fx * 0.7).sin() * (global_fz * 0.7).cos() * 0.5;

            // Set a global base height (e.g., ground level is at Y=32 across the whole world)
            let base_height = 32.0; 
            
            // NO MORE CLAMPING! We want the true global height.
            // Using isize because terrain could technically dip below Y=0
            let global_surface_y = (base_height + height_offset).round() as isize; 
            let current_global_y = global_fy.round() as isize;

            // ==========================================
            // 2. CAVE SYSTEM
            // ==========================================
            let warp_x = global_fx + (global_fy * 0.2).sin() * 2.0;
            let warp_z = global_fz + (global_fy * 0.2).cos() * 2.0;

            let cave_density = (warp_x * 0.15).sin() 
                             * (global_fy * 0.25).cos() 
                             * (warp_z * 0.15).sin();
            
            let is_cave = cave_density > 0.15;

            // ==========================================
            // 3. BLOCK PLACEMENT (Comparing Global to Global)
            // ==========================================
            
            // We now check if THIS block's absolute world position is below the world's surface
            if current_global_y <= global_surface_y && !is_cave {
                
                let dirt_depth = 1.0 + ((global_fx * 0.2).sin() * 2.0).max(0.0);
                let dirt_limit = global_surface_y - (dirt_depth.round() as isize);

                let block_type = if current_global_y == global_surface_y {
                    BlockType::Grass
                } else if current_global_y >= dirt_limit {
                    BlockType::Dirt
                } else {
                    BlockType::Stone
                };

                // We still use local x, y, z to place it in the chunk's 16x16x16 array!
                solid_blocks += 1;
                self.blocks[x][y][z] = Block {
                    block_type,
                }
            }
        }

        // temporary safeguard against empty chunks
        if solid_blocks < 1 {
            self.blocks[0][0][0] = Block {
                block_type: BlockType::Stone,
            }
        }

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

        // let mut chunks: Vec<Chunk> = vec![];
        const CHUNK_AMOUNT: usize = 8; // generate a square of chunks
        const CHUNK_HEIGHT: usize = 3; // how tall to make chunks

        // generate a bunch of chunks
        for x in 0..CHUNK_AMOUNT {
            for y in 0..CHUNK_HEIGHT {
                for z in 0..CHUNK_AMOUNT {
                    let mut chunk = Chunk::new([x, y, z]);
                    let chunk_model = chunk.generate_model(gfx, layout);
                    let chunk_model_id = engine.scene.load_model(chunk_model, &engine.gfx);
                    engine.scene.spawn_node(
                        Node3D::new(Some(chunk_model_id)).with_transform(
                            Transform3D::new()
                                .with_position([(x * CHUNK_SIZE) as f32, (y * CHUNK_SIZE) as f32, (z * CHUNK_SIZE) as f32])
                        )
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
