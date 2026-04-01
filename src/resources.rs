use std::io::{BufReader, Cursor};
use wgpu::util::DeviceExt;

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

pub fn load_string(file_name: &str) -> anyhow::Result<String> {
    let path = std::path::Path::new(env!("OUT_DIR"))
        .join("res")
        .join(file_name);
    let txt = std::fs::read_to_string(path)?;
    Ok(txt)
}

pub fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    let path = std::path::Path::new(env!("OUT_DIR"))
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
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name)?;
    texture::Texture::from_bytes(device, queue, &data, file_name, is_normal_map)
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let obj_text = load_string(file_name)?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    // folder the .obj file is in
    let parent_dir = std::path::Path::new(file_name).parent()
        .unwrap_or(std::path::Path::new(""));

    let (models, obj_materials) = tobj::load_obj_buf(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| {
            let mat_text = load_string(parent_dir.join(p).to_str().unwrap()).unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        // load textures, and use fallbacks if missing
        // diffuse (map_Kd)
        let diffuse_texture = if let Some(t) = &m.diffuse_texture {
            load_texture(parent_dir.join(t).to_str().unwrap(), false, device, queue)?
        } else {
            texture::Texture::from_color(
                device, queue, [255, 255, 255, 255], "default_diffuse", false)?
        };

        // normal (norm)
        let normal_texture = if let Some(t) = &m.normal_texture {
            load_texture(parent_dir.join(t).to_str().unwrap(),true, device, queue)?
        } else {
            texture::Texture::from_color(
                device, queue, [128, 128, 255, 255], "default_normal", false)?
        };

        // specular color (map_Ks)
        let specular_texture = if let Some(t) = &m.specular_texture {
            load_texture(parent_dir.join(t).to_str().unwrap(),false, device, queue)?
        } else {
            texture::Texture::from_color(
                device, queue, [255, 255, 255, 255], "default_specular", false)?
        };

        // dissolve/opacity (map_d)
        let dissolve_texture = if let Some(t) = &m.dissolve_texture {
            load_texture(parent_dir.join(t).to_str().unwrap(),false, device, queue)?
        } else {
            texture::Texture::from_color(
                device, queue, [255, 255, 255, 255], "default_dissolve", false)?
        };

        // ambient occlusion (map_Ka)
        let ambient_texture = if let Some(t) = &m.ambient_texture {
            load_texture(parent_dir.join(t).to_str().unwrap(),false, device, queue)?
        } else {
            texture::Texture::from_color(
                device, queue, [255, 255, 255, 255], "default_ambient", false)?
        };

        // roughness (map_Pr)
        let roughness_texture = if let Some(t) = &m.unknown_param.get("map_Pr") {
            load_texture(parent_dir.join(t).to_str().unwrap(),false, device, queue)?
        } else {
            texture::Texture::from_color(
                device, queue, [0, 0, 0, 255], "default_roughness", false)?
        };

        // metallic (map_Pm)
        let metal_texture = if let Some(t) = m.unknown_param.get("map_Pm") {
            load_texture(parent_dir.join(t).to_str().unwrap(), false, device, queue)?
        } else {
            texture::Texture::from_color(
                device, queue, [0, 0, 0, 255], "default_metal", false)?
        };

        // parse keywords
        let ambient_color = clamp_color(m.ambient.unwrap_or([1.0, 1.0, 1.]));
        let diffuse_color = clamp_color(m.diffuse.unwrap_or([1.0, 1.0, 1.0]));
        let specular_color = clamp_color(m.specular.unwrap_or([0.0, 0.0, 0.0]));
        let emissive_color = clamp_color(parse_vec3(m.unknown_param.get("Ke"), [0.0, 0.0, 0.0]));
        let transmission_filter = clamp_color(parse_vec3(m.unknown_param.get("Tf"), [1.0, 1.0, 1.0]));
        // d first, then Tr
        let dissolve = m.dissolve.or_else(|| {
            m.unknown_param.get("Tr")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .map(|tr| 1.0 - tr)
        }).unwrap_or(1.0).clamp(0.0, 1.0);
        let reflection_sharpness = parse_f32(m.unknown_param.get("sharpness"), 60.0).clamp(0.0, 1000.0);
        let optical_density = m.optical_density.unwrap_or(1.0).clamp(0.001, 10.0);
        // Ns first, then Pr
        let specular_exponent = m.shininess.or_else(|| {
            m.unknown_param.get("Pr")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .map(|pr| {
                    (2.0 / (pr.max(0.0001) * pr.max(0.0001))) - 2.0
                })
            }).unwrap_or(10.0).clamp(0.0, 1000.0);
        let metallic = parse_f32(m.unknown_param.get("Pm"), 0.0).clamp(0.0, 1.0);
        let sheen = parse_f32(m.unknown_param.get("Ps"), 0.0).clamp(0.0, 1.0);
        let clearcoat_thickness = parse_f32(m.unknown_param.get("Pc"), 0.0).clamp(0.0, 1.0);
        let clearcoat_roughness = parse_f32(m.unknown_param.get("Pcr"), 0.0).clamp(0.0, 1.0);
        let anisotropy = parse_f32(m.unknown_param.get("aniso"), 0.0).clamp(0.0, 1.0);
        let anisotropy_rotation = parse_f32(m.unknown_param.get("anisor"), 0.0).clamp(0.0, 1.0);
        
        let illumination_model = u32::from(m.illumination_model.unwrap_or(2)).clamp(0, 10); 



        materials.push(model::Material::new(
            device,
            &m.name,
            model::MaterialTextures::new(
                diffuse_texture,
                normal_texture,
                specular_texture,
                dissolve_texture,
                ambient_texture,
                roughness_texture,
                metal_texture,
            ),
            layout,
            model::MaterialUniforms::new(
                ambient_color,
                diffuse_color,
                specular_color,
                emissive_color,
                transmission_filter,
                dissolve,
                specular_exponent,
                optical_density,
                reflection_sharpness,
                metallic,
                sheen,
                clearcoat_thickness,
                clearcoat_roughness,
                anisotropy,
                anisotropy_rotation,
                illumination_model,
            )
        ));
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let mut vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [
                        m.mesh.texcoords[i * 2],
                        1.0 - m.mesh.texcoords[i * 2 + 1]
                    ],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                })
                .collect::<Vec<_>>();

            let indices = &m.mesh.indices;
            let mut triangles_included = vec![0; vertices.len()];

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{file_name:?} Vertex Buffer")),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{file_name:?} Index Buffer")),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            log::info!("Loaded model: {}", m.name);
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

