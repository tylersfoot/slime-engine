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
const RENDER_DISTANCE: usize = 32; // render distance (square side length)
const RENDER_HEIGHT: usize = 1; // how many chunks high to generate
const SEED: u32 = 42;

// helper functions
fn rand_color(rng: &mut StdRng) -> [f32; 4] {
    [rng.random(), rng.random(), rng.random(), 1.0]
}
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[derive(Clone, Debug)]
struct NoiseGeneratorOctaves {
    octaves: Vec<NoiseGeneratorPerlin>,
    total_octaves: usize,
}

impl NoiseGeneratorOctaves {
    fn new(rng: &mut StdRng, count: usize) -> Self {
        let mut generator = Self {
            octaves: vec![],
            total_octaves: count,
        };

        for i in 0..count {
            generator.octaves.insert(i, NoiseGeneratorPerlin::new(rng));
        }

        generator
    }

    fn generate_noise_for_coordinate(&self, var1: f64, var3: f64) -> f64 {
        let mut var5: f64 = 0.0_f64;
        let mut var7: f64 = 1.0_f64;

        for var9 in 0..self.total_octaves {
            var5 += self.octaves[var9].sample_point(var1 * var7, var3 * var7, 0.0) / var7;
            var7 /= 2.0_f64;
        }

        var5
    }

    // public double[] generateNoise(double[] var1, double var2, double var4, double var6, int var8, int var9, int var10, double var11, double var13, double var15) {
    //     if (var1 == null) {
    //         var1 = new double[var8 * var9 * var10];
    //     } else {
    //         for(int var17 = 0; var17 < var1.length; ++var17) {
    //             var1[var17] = 0.0D;
    //         }
    //     }

    //     double var20 = 1.0D;

    //     for(int var19 = 0; var19 < this.totalNoiseGenerators; ++var19) {
    //         this.noiseGenerators[var19].a(var1, var2, var4, var6, var8, var9, var10, var11 * var20, var13 * var20, var15 * var20, var20);
    //         var20 /= 2.0D;
    //     }

    //     return var1;
    // }

    fn generate_noise(&self, out: Vec<f64>, 
        var2: f64, var4: f64, var6: f64,
        var8: i32, var9: i32, var10: i32,
        var11: f64, var13: f64, var15: f64) -> Vec<f64> {

        let mut out = vec![0.0_f64; (var8 * var9 * var10) as usize];
        let mut scale: f64 = 1.0_f64;

        for octave in 0..self.total_octaves {
            out = self.octaves[octave].sample_grid(
                out,
                var2, var4, var6,
                var8, var9, var10,
                var11 * scale, var13 * scale, var15 * scale,
                scale
            );
            scale /= 2.0_f64;
        }

        out
    }

    fn generate_noise2(&self, out: Vec<f64>, 
        var2: i32, var3: i32, var4: i32, var5: i32,
        var6: f64, var8: f64, var10: f64) -> Vec<f64> {
        self.generate_noise(
            out,
            var2 as f64,
            10.0,
            var3 as f64,
            var4,
            1,
            var5,
            var6,
            1.0,
            var8
        )
    }

    fn sample_point_fbm(&self, x: f64, y: f64, z: f64) -> f64 {
        let mut total = 0.0_f64;
        let mut scale = 1.0_f64;
        for octave in &self.octaves {
            // mc weighting: freq *= scale (halves), result /= scale (amplitude grows)
            total += octave.sample_point(x * scale, y * scale, z * scale) / scale;
            scale /= 2.0_f64;
        }
        total
    }
}

#[derive(Clone, Debug, Copy)]
struct NoiseGeneratorPerlin {
    // permutation table; holds random shuffled ints
    table: [i32; 512],
    // random offsets added to every coordinate
    offset_x: f64,
    offset_y: f64,
    offset_z: f64,
}

impl NoiseGeneratorPerlin {
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
            table.swap(v2, v3);
            table[v2 + 256] = table[v2]; // duplicate into upper half
            v2 += 1;
        }
        
        Self { table, offset_x, offset_y, offset_z }
    }

    fn gradient3(&self, hash: i32, x: f64, y: f64, z: f64) -> f64 {
        // computes 3D gradient dot product
        let h: i32 = hash & 15;
        let v1: f64 =  if (h < 8) {x} else {y};
        let v2: f64 =  if (h < 4) {y} else {
            if (h != 12 && h != 14) {z} else {x}
        };
        let v3: f64 = if (h & 1) == 0 {v1} else {-v1};
        let v4: f64 = if (h & 2) == 0 {v2} else {-v2};
        v3 + v4  
    }

    fn gradient2(&self, hash: i32, x: f64, y: f64) -> f64 {
        // computes 2D gradient dot product
        let h: i32 = hash & 15;
        let v1: f64 = (1 - ((h & 8) >> 3)) as f64 * x;
        let v2: f64 =  if (h < 4) {0.0_f64} else {
            if (h != 12 && h != 14) {y} else {x}
        };
        let v3: f64 = if ((h & 1) == 0) {v1} else {-v1};
        let v4: f64 = if ((h & 2) == 0) {v2} else {-v2};
        v3 + v4
    }

    fn sample_point(&self, mut x: f64, mut y: f64, mut z: f64) -> f64 {
        // // apply generator's random origin offset
        // x += self.offset_x;
        // y += self.offset_y;
        // z += self.offset_z;
        // // cast to ints (this is what causes the farlands)
        // let mut xi: i32 = x as i32;
        // let mut yi: i32 = y as i32;
        // let mut zi: i32 = z as i32;
        // if (x < xi as f64) {
        //     xi -= 1;
        // }
        // if (y < yi as f64) {
        //     yi -= 1;
        // }
        // if (z < zi as f64) {
        //     zi -= 1;
        // }
        // let var16: i32 = xi & 255;
        // let var17: i32 = yi & 255;
        // let var18: i32 = zi & 255;
        // x -= xi as f64;
        // y -= yi as f64;
        // z -= zi as f64;
        // let var19: f64 = x * x * x * (x * (x * 6.0_f64 - 15.0_f64) + 10.0_f64);
        // let var21: f64 = y * y * y * (y * (y * 6.0_f64 - 15.0_f64) + 10.0_f64);
        // let var23: f64 = z * z * z * (z * (z * 6.0_f64 - 15.0_f64) + 10.0_f64);
        // let var25: i32 = self.table[var16 as usize] + var17;
        // let var26: i32 = self.table[var25 as usize] + var18;
        // let var27: i32 = self.table[var25 as usize + 1] + var18;
        // let var28: i32 = self.table[var16 as usize + 1] + var17;
        // let var29: i32 = self.table[var28 as usize] + var18;
        // let var30: i32 = self.table[var28 as usize + 1] + var18;
        // return lerp(
        //     lerp(
        //         lerp(
                    
        //             self.gradient3(
        //                 self.table[var26 as usize],
        //                 x,
        //                 y,
        //                 z
        //             ),
        //             self.gradient3(
        //                 self.table[var29 as usize],
        //                 x - 1.0_f64,
        //                 y,
        //                 z
        //             ),
        //             var19,
        //         ),
        //         lerp(
        //             self.gradient3(
        //                 self.table[var27 as usize],
        //                 x,
        //                 y - 1.0_f64,
        //                 z
        //             ),
        //             self.gradient3(
        //                 self.table[var30 as usize],
        //                 x - 1.0_f64,
        //                 y - 1.0_f64,
        //                 z
        //             ),
        //             var19,
        //         ),
        //         var21,
        //     ),
        //     lerp(
        //         lerp(
        //             self.gradient3(
        //                 self.table[var26 as usize + 1],
        //                 x,
        //                 y,
        //                 z - 1.0_f64
        //             ),
        //             self.gradient3(
        //                 self.table[var29 as usize + 1],
        //                 x - 1.0_f64,
        //                 y,
        //                 z - 1.0_f64
        //             ),
        //             var19,
        //         ),
        //         lerp(
        //             self.gradient3(
        //                 self.table[var27 as usize + 1],
        //                 x,
        //                 y - 1.0_f64,
        //                 z - 1.0_f64
        //             ),
        //             self.gradient3(
        //                 self.table[var30 as usize + 1],
        //                 x - 1.0_f64,
        //                 y - 1.0_f64,
        //                 z - 1.0_f64
        //             ),
        //             var19,
        //         ),
        //         var21,
        //     ),
        //     var23,
        // );

        let mut var7: f64 = x + self.offset_x;
        let mut var9: f64 = y + self.offset_y;
        let mut var11: f64 = z + self.offset_z;
        let mut var13: i32 = var7 as i32;
        let mut var14: i32 = var9 as i32;
        let mut var15: i32 = var11 as i32;
        if (var7 < var13 as f64) {
            var13 -= 1;
        }
        if (var9 < var14 as f64) {
            var14 -= 1;
        }
        if (var11 < var15 as f64) {
            var15 -= 1;
        }
        let var16: i32 = var13 & 255;
        let var17: i32 = var14 & 255;
        let var18: i32 = var15 & 255;
        var7 -= var13 as f64;
        var9 -= var14 as f64;
        var11 -= var15 as f64;
        let var19: f64 = var7 * var7 * var7 * (var7 * (var7 * 6.0_f64 - 15.0_f64) + 10.0_f64);
        let var21: f64 = var9 * var9 * var9 * (var9 * (var9 * 6.0_f64 - 15.0_f64) + 10.0_f64);
        let var23: f64 = var11 * var11 * var11 * (var11 * (var11 * 6.0_f64 - 15.0_f64) + 10.0_f64);
        let var25: i32 = self.table[var16 as usize] + var17;
        let var26: i32 = self.table[var25 as usize] + var18;
        let var27: i32 = self.table[var25 as usize + 1] + var18;
        let var28: i32 = self.table[var16 as usize + 1] + var17;
        let var29: i32 = self.table[var28 as usize] + var18;
        let var30: i32 = self.table[var28 as usize + 1] + var18;
        lerp(
            lerp(
                lerp(
                    
                    self.gradient3(
                        self.table[var26 as usize],
                        var7,
                        var9,
                        var11
                    ),
                    self.gradient3(
                        self.table[var29 as usize],
                        var7 - 1.0_f64,
                        var9,
                        var11
                    ),
                    var19,
                ),
                lerp(
                    self.gradient3(
                        self.table[var27 as usize],
                        var7,
                        var9 - 1.0_f64,
                        var11
                    ),
                    self.gradient3(
                        self.table[var30 as usize],
                        var7 - 1.0_f64,
                        var9 - 1.0_f64,
                        var11
                    ),
                    var19,
                ),
                var21,
            ),
            lerp(
                lerp(
                    self.gradient3(
                        self.table[var26 as usize + 1],
                        var7,
                        var9,
                        var11 - 1.0_f64
                    ),
                    self.gradient3(
                        self.table[var29 as usize + 1],
                        var7 - 1.0_f64,
                        var9,
                        var11 - 1.0_f64
                    ),
                    var19,
                ),
                lerp(
                    self.gradient3(
                        self.table[var27 as usize + 1],
                        var7,
                        var9 - 1.0_f64,
                        var11 - 1.0_f64
                    ),
                    self.gradient3(
                        self.table[var30 as usize + 1],
                        var7 - 1.0_f64,
                        var9 - 1.0_f64,
                        var11 - 1.0_f64
                    ),
                    var19,
                ),
                var21,
            ),
            var23,
        )
    }

    fn sample_grid(&self, out: Vec<f64>,
        var2: f64, var4: f64, var6: f64, 
        var8: i32, var9: i32, var10: i32, 
        var11: f64, var13: f64, var15: f64,
        var17: f64) -> Vec<f64> {
        // we return new vec instead of editing given vec
        let mut out = out.clone();
        let mut var19: i32;
        let mut var20: i32;
        let mut var21: f64;
        let mut var23: f64;
        let mut var25: f64;
        let mut var27: i32;
        let mut var28: f64;
        let mut var30: i32;
        let mut var31: i32;
        let mut var32: i32;
        let mut var33: i32;
        let mut var36: bool;
        let mut var37: bool;
        let mut var42: f64;
        let mut var46: i32;
        if (var9 == 1) {
            let var34: bool = false;
            let var35: bool = false;
            var36 = false;
            var37 = false;
            let mut var38: f64 = 0.0_f64;
            let mut var40: f64 = 0.0_f64;
            var33 = 0;
            var42 = 1.0_f64 / var17;

            for var44 in 0..var8 {
                var21 = (var2 + var44 as f64) * var11 + self.offset_x;
                let mut var45: i32 = var21 as i32;
                if (var21 < var45 as f64) {
                    var45 -= 1;
                }

                var46 = var45 & 255;
                var21 -= var45 as f64;
                var23 = var21 * var21 * var21 * (var21 * (var21 * 6.0_f64 - 15.0_f64) + 10.0_f64);

                for var27 in 0..var10 {
                    var25 = (var6 + var27 as f64) * var15 + self.offset_z;
                    var30 = var25 as i32;
                    if (var25 < var30 as f64) {
                        var30 -= 1;
                    }

                    var31 = var30 & 255;
                    var25 -= var30 as f64;
                    var28 = var25 * var25 * var25 * (var25 * (var25 * 6.0_f64 - 15.0_f64) + 10.0_f64);
                    var19 = self.table[var46 as usize]; // + 0
                    let var47: i32 = self.table[var19 as usize] + var31;
                    let var48: i32 = self.table[var46 as usize + 1]; // + 0
                    var20 = self.table[var48 as usize] + var31;
                    var38 = lerp(
                        self.gradient2(self.table[var47 as usize], var21, var25),
                        self.gradient3(self.table[var20 as usize], var21 - 1.0_f64, 0.0_f64, var25),
                        var23,
                    );
                    var40 = lerp(
                        self.gradient3(self.table[var47 as usize + 1], var21, 0.0_f64, var25 - 1.0_f64),
                        self.gradient3(self.table[var20 as usize + 1], var21 - 1.0_f64, 0.0_f64, var25 - 1.0_f64),
                        var23,
                    );
                    let var49: f64 = lerp(var38, var40, var28);
                    var32 = var33;
                    var33 += 1;
                    out[var32 as usize] += var49 * var42;
                }
            }
        } else {
            var19 = 0;
            let var66: f64 = 1.0_f64 / var17;
            var20 = -1;
            var36 = false;
            var37 = false;
            let var67: bool = false;
            let var39: bool = false;
            let var68: bool = false;
            let var41: bool = false;
            var42 = 0.0_f64;
            var21 = 0.0_f64;
            let mut var69: f64 = 0.0_f64;
            var23 = 0.0_f64;

            for var27 in 0..var8 {
                var25 = (var2 + var27 as f64) * var11 + self.offset_x;
                var30 = var25 as i32;
                if (var25 < var30 as f64) {
                    var30 -= 1;
                }

                var31 = var30 & 255;
                var25 -= var30 as f64;
                var28 = var25 * var25 * var25 * (var25 * (var25 * 6.0_f64 - 15.0_f64) + 10.0_f64);

                for var46 in 0..var10 {
                    let mut var70: f64 = (var6 + var46 as f64) * var15 + self.offset_z;
                    let mut var71: i32 = var70 as i32;
                    if (var70 < var71 as f64) {
                        var71 -= 1;
                    }

                    let var50: i32 = var71 & 255;
                    var70 -= var71 as f64;
                    let var51: f64 = var70 * var70 * var70 * (var70 * (var70 * 6.0_f64 - 15.0_f64) + 10.0_f64);

                    for var53 in 0..var9 {
                        let mut var54: f64 = (var4 + var53 as f64) * var13 + self.offset_y;
                        let mut var56: i32 = var54 as i32;
                        if (var54 < var56 as f64) {
                            var56 -= 1;
                        }

                        let var57: i32 = var56 & 255;
                        var54 -= var56 as f64;
                        let var58: f64 = var54 * var54 * var54 * (var54 * (var54 * 6.0_f64 - 15.0_f64) + 10.0_f64);
                        if (var53 == 0 || var57 != var20) {
                            var20 = var57;
                            let var60: i32 = self.table[var31 as usize] + var57;
                            let var61: i32 = self.table[var60 as usize] + var50;
                            let var62: i32 = self.table[var60 as usize + 1] + var50;
                            let var63: i32 = self.table[var31 as usize + 1] + var57;
                            var33 = self.table[var63 as usize] + var50;
                            let var64: i32 = self.table[var63 as usize + 1] + var50;
                            var42 = lerp(
                                
                                self.gradient3(self.table[var61 as usize], var25, var54, var70),
                                self.gradient3(self.table[var33 as usize], var25 - 1.0_f64, var54, var70),
                                var28,
                            );
                            var21 = lerp(
                                self.gradient3(self.table[var62 as usize], var25, var54 - 1.0_f64, var70),
                                self.gradient3(self.table[var64 as usize], var25 - 1.0_f64, var54 - 1.0_f64, var70),
                                var28,
                            );
                            var69 = lerp(
                                self.gradient3(self.table[var61 as usize + 1], var25, var54, var70 - 1.0_f64),
                                self.gradient3(self.table[var33 as usize + 1], var25 - 1.0_f64, var54, var70 - 1.0_f64),
                                var28,
                            );
                            var23 = lerp(
                                self.gradient3(self.table[var62 as usize + 1], var25, var54 - 1.0_f64, var70 - 1.0_f64),
                                self.gradient3(self.table[var64 as usize + 1], var25 - 1.0_f64, var54 - 1.0_f64, var70 - 1.0_f64),
                                var28,
                            );
                        }

                        let var72: f64 = lerp(var42, var21, var58);
                        let var73: f64 = lerp(var69, var23, var58);
                        let var74: f64 = lerp(var72, var73, var51);
                        var32 = var19;
                        var19 += 1;
                        out[var32 as usize] += var74 * var66;
                    }
                }
            }
        }

        out
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
        let mut rng = StdRng::seed_from_u64(SEED as u64);

        let noise_min_limit = NoiseGeneratorOctaves::new(&mut rng, 16); // 'low' density bound
        let noise_max_limit  = NoiseGeneratorOctaves::new(&mut rng, 16); // 'high' density bound
        let noise_selector = NoiseGeneratorOctaves::new(&mut rng, 8); // lerp weight between low/high
        let noise_beach = NoiseGeneratorOctaves::new(&mut rng, 4); // sand/gravel patches
        let noise_surface_depth  = NoiseGeneratorOctaves::new(&mut rng, 4); // how deep dirt/sand layer is
        let noise_scale = NoiseGeneratorOctaves::new(&mut rng, 10); // horizontal 'stretch' of terrain
        let noise_depth = NoiseGeneratorOctaves::new(&mut rng, 16); // base elevation/depth
        let noise_tree_count = NoiseGeneratorOctaves::new(&mut rng, 8); // tree density in populate

        // let noise_min_limit = Fbm::<Perlin>::new(SEED+1).set_octaves(16); // 'low' density bound
        // let noise_max_limit = Fbm::<Perlin>::new(SEED+2).set_octaves(16); // 'high' density bound
        // let noise_selector = Fbm::<Perlin>::new(SEED+3).set_octaves(8); // lerp weight between low/high
        // let noise_beach = Fbm::<Perlin>::new(SEED+4).set_octaves(4); // sand/gravel patches
        // let noise_surface_depth = Fbm::<Perlin>::new(SEED+5).set_octaves(4); // how deep dirt/sand layer is
        // let noise_scale = Fbm::<Perlin>::new(SEED+6).set_octaves(10); // horizontal 'stretch' of terrain
        // let noise_depth = Fbm::<Perlin>::new(SEED+7).set_octaves(16); // base elevation/depth
        // let noise_tree_count = Fbm::<Perlin>::new(SEED+8).set_octaves(8); // tree density in populate

        // how many density CELLS per chunk
        const DENSITY_CELLS_XZ: usize = 4;
        const DENSITY_CELLS_Y: usize = 8;
        // how many density POINTS per chunk
        const DENSITY_SAMPLES_XZ: usize = CHUNK_SIZE_XZ / DENSITY_CELLS_XZ + 1;
        const DENSITY_SAMPLES_Y: usize = CHUNK_SIZE_Y / DENSITY_CELLS_Y + 1;
        let mut density_grid = [[[0.0; DENSITY_SAMPLES_XZ]; DENSITY_SAMPLES_Y]; DENSITY_SAMPLES_XZ];
        const NOISE_SCALE_XZ: f64 = 684.412;
        const NOISE_SCALE_Y: f64 = 684.412;
        const SEA_LEVEL: f64 = 64.0;

        // calculate density grid values
        for (x, density_x) in density_grid.iter_mut().enumerate() {
            for (y, density_y) in density_x.iter_mut().enumerate() {
                for (z, density_value) in density_y.iter_mut().enumerate() {
                    // global coordinates
                    // let gx = (x as f64 * DENSITY_CELLS_XZ as f64) + (self.position[0] * CHUNK_SIZE_XZ) as f64;
                    // let gy = (y as f64 * DENSITY_CELLS_Y as f64) + (self.position[1] * CHUNK_SIZE_Y) as f64;
                    // let gz = (z as f64 * DENSITY_CELLS_XZ as f64) + (self.position[2] * CHUNK_SIZE_XZ) as f64;
                    let gx = (x as f64) + (self.position[0] * DENSITY_CELLS_XZ) as f64;
                    let gy = (y as f64) + (self.position[1] * DENSITY_CELLS_Y) as f64;
                    let gz = (z as f64) + (self.position[2] * DENSITY_CELLS_XZ) as f64;

                    // 2d fields (per column) - control terrain shape
                    let n_scale = noise_scale.sample_point_fbm(
                        gx * 1.121_f64,
                        10.0_f64,
                        gz * 1.121_f64
                    );
                    let n_depth = noise_depth.sample_point_fbm(
                        gx * 200.0_f64,
                        10.0_f64,
                        gz * 200.0_f64,
                    );

                    // 3d fields (the density) - min/max's frequencies overflow at farlands here
                    let n_selector = noise_selector.sample_point_fbm(
                        gx * (NOISE_SCALE_XZ / 80.0_f64),
                        gy * (NOISE_SCALE_Y / 160.0_f64),
                        gz * (NOISE_SCALE_XZ / 80.0_f64),
                    );
                    let n_min_limit = noise_min_limit.sample_point_fbm(
                        gx * NOISE_SCALE_XZ,
                        gy * NOISE_SCALE_Y,
                        gz * NOISE_SCALE_XZ,
                    );
                    let n_max_limit = noise_max_limit.sample_point_fbm(
                        gx * NOISE_SCALE_XZ,
                        gy * NOISE_SCALE_Y,
                        gz * NOISE_SCALE_XZ,
                    );

                    // ---- PER-COLUMN NOISE ----

                    // TODO humidity/temp

                    // horizontal stretch - compresses vertical density gradient
                    let mut horizontal_stretch = (n_scale + 256.0_f64) / 512.0_f64;
                    horizontal_stretch = horizontal_stretch.min(1.0_f64);

                    // elevation/depth
                    let mut depth = n_depth / 8000.0_f64; 
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

                    let min_density = n_min_limit / 512.0_f64;
                    let max_density = n_max_limit / 512.0_f64;
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
        let mut indices: Vec<u32> = Vec::new();

        // local cube data
        let face_data = [
            // (normal, neighbor_offset, [corner; 4])
            ([ 0.0,  0.0,  1.0], ( 0,  0,  1), [[-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]]), // front  (+Z)
            ([ 0.0,  0.0, -1.0], ( 0,  0, -1), [[ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5]]), // back   (-Z)
            ([ 1.0,  0.0,  0.0], ( 1,  0,  0), [[ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5]]), // right  (+X)
            ([-1.0,  0.0,  0.0], (-1,  0,  0), [[-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5]]), // left   (-X)
            ([ 0.0,  1.0,  0.0], ( 0,  1,  0), [[-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5]]), // top    (+Y)
            ([ 0.0, -1.0,  0.0], ( 0, -1,  0), [[-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5]]), // bottom (-Y)
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
                    for (normal, (ox, oy, oz), corners) in face_data {
                        // dont render face if against another solid block
                        let nx = x as i32 + ox;
                        let ny = y as i32 + oy;
                        let nz = z as i32 + oz;
                        if self.is_solid(nx, ny, nz) {
                            continue;
                        }
                        
                        // index of first vertex of THIS face
                        let base = vertices.len() as u32;
                        
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

    fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        // out of bounds = treat as air
        if x < 0 || y < 0 || z < 0
            || x >= CHUNK_SIZE_XZ as i32
            || y >= CHUNK_SIZE_Y as i32
            || z >= CHUNK_SIZE_XZ as i32 {
            return false;
        }
        self.blocks[x as usize][y as usize][z as usize].block_type != BlockType::Air
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
        let mut color_rng = StdRng::seed_from_u64(SEED as u64);

        // tweak to change camera + chunks root position (for precision errors)
        // make it divisible by CHUNK_SIZE_XZ (16) im pretty sure
        // also probably keep y=0
        let offset: [f64; 3] = [12_550_780.0, 0.0, 0.0];

        let camera = engine.scene.spawn_camera(
            [
                // (-10.0 + offset[0]) as f32,
                // (100.0 + offset[1]) as f32,
                // (-10.0 + offset[2]) as f32,
                -10.0, 100.0, -10.0
            ],
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
                    // with offset applied
                    let coords = [
                        x + (offset[0] as usize / CHUNK_SIZE_XZ),
                        y + (offset[1] as usize / CHUNK_SIZE_Y),
                        z + (offset[2] as usize / CHUNK_SIZE_XZ)
                    ];
                    let mut chunk = Chunk::new(coords);

                    // skip making model if the chunk is empty (all air)
                    if chunk.is_empty {
                        continue;
                    }
                    let chunk_model = chunk.generate_model(gfx, layout);
                    let chunk_model_id = engine.scene.load_model(chunk_model, &engine.gfx);
                    engine.scene.spawn_node(
                        Node3D::new(Some(chunk_model_id)).with_transform(
                            Transform3D::new()
                                .with_position([
                                    // (coords[0] * CHUNK_SIZE_XZ) as f32,
                                    // (coords[1] * CHUNK_SIZE_Y) as f32,
                                    // (coords[2] * CHUNK_SIZE_XZ) as f32
                                    (x * CHUNK_SIZE_XZ) as f32,
                                    (y * CHUNK_SIZE_Y) as f32,
                                    (z * CHUNK_SIZE_XZ) as f32,
                                    ])
                        )
                        .with_color(rand_color(&mut color_rng))
                    );
                }
            }
        }
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
