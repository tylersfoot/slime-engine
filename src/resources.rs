use std::io::{BufReader, Cursor};
use std::path::Path;
use wgpu::util::DeviceExt;
use anyhow::{Context, Result};

use crate::{model, texture};

// helper to parse unknown mtl colors/vec3
fn parse_vec3(value: Option<&String>, default: [f32; 3]) -> [f32; 3] {
    value.and_then(|s| {
        let mut parts = s.split_whitespace();
        Some([
            parts.next()?.parse().ok()?,
            parts.next()?.parse().ok()?,
            parts.next()?.parse().ok()?,
        ])
    })
    .unwrap_or(default)
}

// helper to parse unknown mtl floats
fn parse_f32(value: Option<&String>, default: f32) -> f32 {
    value.and_then(|s| s.trim().parse().ok()).unwrap_or(default)
}

// helper to clamp colors/vec3 to [0,1]
fn clamp_color(c: [f32; 3]) -> [f32; 3] {
    c.map(|x| x.clamp(0.0, 1.0))
}

pub fn load_string(file_name: &str) -> Result<String> {
    let path = Path::new(env!("OUT_DIR"))
        .join("res")
        .join(file_name);
    let txt = std::fs::read_to_string(path)?;
    Ok(txt)
}

pub fn load_binary(file_name: &str) -> Result<Vec<u8>> {
    let path = Path::new(env!("OUT_DIR"))
        .join("res")
        .join(file_name);
    let data = std::fs::read(path)?;
    Ok(data)
}

pub fn load_texture(
    file_name: &str,
    is_normal_map: bool,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<texture::Texture> {
    let data = load_binary(file_name)?;
    texture::Texture::from_bytes(device, queue, &data, file_name, is_normal_map)
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> Result<model::Model> {
    let obj_text = load_string(file_name)?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    // folder the .obj file is in
    let parent_dir = Path::new(file_name).parent()
        .unwrap_or(Path::new(""));

    let (models, obj_materials) = tobj::load_obj_buf(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| {
            let mtl_path = parent_dir.join(p);

            match load_string(&mtl_path.to_string_lossy()) {
                Ok(mat_text) => {
                    tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
                },
                Err(e) => {
                    log::warn!("Missing or invalid .mtl file at {:?}: {}. Using default materials", mtl_path, e);
                    tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(String::new())))
                }
            }
        },
    )?;

    let mut materials = Vec::new();

    // fallback to empty vector if materials failed to parse
    let obj_materials = obj_materials.unwrap_or_else(|e| {
        log::warn!("Failed to parse materials: {}", e);
        Vec::new()
    });

    for m in obj_materials {
        // helper closure to try loading texture, or use fallback
        let mut get_texture = |
            texture: Option<&String>,
            is_normal: bool,
            fallback_color: [u8; 4],
            fallback_name: &str
        | -> Result<texture::Texture> {
            if let Some(t) = texture {
                let path = parent_dir.join(t);
                let path = path.to_string_lossy();
                match load_texture(&path, is_normal, device, queue) {
                    Ok(tex) => return Ok(tex),
                    Err(e) => log::warn!("Failed to load texture {}: {}. Using fallback", path, e),
                }
            }

            // if texture is None or failed loading, return fallback
            texture::Texture::from_color(device, queue, fallback_color, fallback_name, is_normal)
        };

        // diffuse (map_Kd)
        let diffuse_texture = get_texture(m.diffuse_texture.as_ref(),
            false, [255, 255, 255, 255], "default_diffuse")?;
        
        // parse keywords
        let ambient_color = clamp_color(m.ambient.unwrap_or([1.0, 1.0, 1.]));
        let diffuse_color = clamp_color(m.diffuse.unwrap_or([1.0, 1.0, 1.0]));
        let specular_color = clamp_color(m.specular.unwrap_or([0.0, 0.0, 0.0]));
        let emissive_color = clamp_color(parse_vec3(m.unknown_param.get("Ke"), [0.0, 0.0, 0.0]));
        // d first, then Tr
        let dissolve = m.dissolve.or_else(|| {
            m.unknown_param.get("Tr")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .map(|tr| 1.0 - tr)
        }).unwrap_or(1.0).clamp(0.0, 1.0);
        // Ns first, then Pr
        let specular_exponent = m.shininess.or_else(|| {
            m.unknown_param.get("Pr")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .map(|pr| {
                    (2.0 / (pr.max(0.0001) * pr.max(0.0001))) - 2.0
                })
            }).unwrap_or(10.0).clamp(0.0, 1000.0);

        materials.push(model::Material::new(
            device,
            &m.name,
            model::MaterialTextures::new(
                diffuse_texture,
            ),
            layout,
            model::MaterialUniforms::new(
                ambient_color,
                diffuse_color,
                specular_color,
                emissive_color,
                dissolve,
                specular_exponent,
            )
        ));
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let has_texcoords = !m.mesh.texcoords.is_empty();
            let has_normals = !m.mesh.normals.is_empty();

            let mut vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: if has_texcoords {
                        [
                            m.mesh.texcoords[i * 2],
                            1.0 - m.mesh.texcoords[i * 2 + 1]
                        ]
                    } else {
                        // fallback UV coordinate
                        [0.0, 0.0]
                    },
                    normal: if has_normals {
                        [
                            m.mesh.normals[i * 3],
                            m.mesh.normals[i * 3 + 1],
                            m.mesh.normals[i * 3 + 2],
                        ]
                    } else {
                        // fallback normal pointing up
                        // TODO: calculate normals?
                        [0.0, 1.0, 0.0]
                    },
                })
                .collect::<Vec<_>>();

            let indices = &m.mesh.indices;

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{file_name:?}_vertex_buffer")),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{file_name:?}_index_buffer")),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            log::info!("Loaded model: {}->{} [v:{} i:{}]",
                Path::new(file_name).file_stem().and_then(|s| s.to_str()).unwrap_or(file_name),
                m.name,
                vertices.len(),
                indices.len()
            );

            model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes, materials })
}

