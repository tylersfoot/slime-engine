#![allow(unused)]
use std::{collections::HashMap, path::Path};
use crate::*;

#[derive(Debug, Default)]
pub struct RawObjData {
    // holds the raw data from the .obj file
    pub vertices: Vec<[f32; 3]>,
    pub texture_coords: Vec<[f32; 2]>,
    pub normals: Vec<[f32; 3]>,
    // list of mesh parts found in the file (o tag)
    pub object_parts: Vec<ObjectPart>,
}

#[derive(Debug, Clone)]
pub struct ObjectPart {
    // represents a single part of the .obj file
    pub name: String,
    pub material_name: Option<String>,
    pub faces: Vec<Face>,
}

#[derive(Debug, Clone)]
pub struct Material {
    pub name: String, // netmtl
    pub ambient_color: [f32; 3], // Ka
    pub diffuse_color: [f32; 3], // Kd
    pub emission_color: [f32; 3], // Ke
    pub roughness: f32, // Pr
    pub clearcoat: f32, // Pl
    pub sheen: f32, // Pds
    pub transmission_filter: [f32; 3], // Tf
    pub transparency: f32, // Tr
    pub illumination_model: u32, // illum
}

impl Material {
    pub fn new(name: String) -> Self {
        Self {
            name,
            ambient_color: [0.0, 0.0, 0.0],
            diffuse_color: [0.0, 0.0, 0.0],
            emission_color: [0.0, 0.0, 0.0],
            roughness: 0.0,
            clearcoat: 0.0,
            sheen: 0.0,
            transmission_filter: [1.0, 1.0, 1.0],
            transparency: 1.0,
            illumination_model: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Face {
    pub vertex_indices: Vec<u32>, // indices of vertices in the face
    pub texture_indices: Vec<u32>, // indices of texture coordinates in the face
    pub normal_indices: Vec<u32>, // indices of normals in the face
}

pub fn load_object_into_scene(
    scene: &mut Scene,
    obj_path: &str,
    mtl_path: &str,
    name: &str,
    base_transform: Transform, // applies to whole imported model
) -> Result<NodeId, String> {
    // loads an .obj file and populates the given scene with its contents

    // parse the material and object data from files
    let materials = parse_mtl(Path::new(mtl_path))?;

    // add materials to the scene's arena and create a map of name -> MaterialId
    let mut material_map: HashMap<String, MaterialId> = HashMap::new();
    for (name, material) in materials {
        let id = scene.materials.insert(material);
        material_map.insert(name, id);
    }

    let raw_data = parse_obj(Path::new(obj_path))?;

    // create a single root node for the entire .obj file
    // makes it easy to move/rotate/scale the whole model
    let root_node_id = scene.add_node(
        name,
        base_transform,
        None,
        None,
    );

    // process each part of the object file
    for part in raw_data.object_parts {
        // find the MaterialId for this part using the name from the file
        let material_id = part.material_name.and_then(|name| material_map.get(&name).copied());
        
        // create the mesh data component
        let mesh_component = MeshComponent {
            vertices: raw_data.vertices.iter().map(|&v| (v[0], v[1], v[2])).collect(),
            faces: part.faces,
            material_id,
        };

        // add the mesh component to the scene's arena
        let mesh_id = scene.meshes.insert(mesh_component);

        // Create a new node for this specific part, parented to the root node.
        scene.add_node(
            &part.name,
            Transform::default(), // It starts at the parent's origin
            Some(ObjectKind::Mesh(mesh_id)),
            Some(root_node_id),
        );
    }

    Ok(root_node_id)
}


fn parse_obj(obj_path: &Path) -> Result<RawObjData, String> {
    // parse the OBJ file and return a vector of objects
    let content = std::fs::read_to_string(obj_path)
        .map_err(|e| format!("Failed to read OBJ file: {e}"))?;

    let mut data = RawObjData::default();
    let mut current_part: Option<ObjectPart> = None;

    // pass 1:  read all geometric data
    for line in content.lines() {
        // split line into tokens/words/numbers
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() || tokens[0].starts_with('#') {
            // skip empty lines and comments
            continue;
        }

        match tokens[0] {
            "v" => {
                // vertex position
                let mut vertex = [0.0; 3];
                for i in 0..3 {
                    vertex[i] = tokens[i + 1].parse::<f32>().map_err(|e| format!("Failed to parse vertex: {e}"))?;
                }
                data.vertices.push(vertex);
            },
            "vt" => {
                // texture coordinate
                let mut tex_coord = [0.0; 2];
                for i in 0..2 {
                    tex_coord[i] = tokens[i + 1].parse::<f32>().map_err(|e| format!("Failed to parse texture coordinate: {e}"))?;
                }
                data.texture_coords.push(tex_coord);
            }
            "vn" => {
                // vertex normal
                let mut normal = [0.0; 3];
                for i in 0..3 {
                    normal[i] = tokens[i + 1].parse::<f32>().map_err(|e| format!("Failed to parse vertex normal: {e}"))?;
                }
                data.normals.push(normal);
            }
            _ => (),
        }
    }

    // pass 2: read object parts and faces
    for line in content.lines() {
        // split line into tokens/words/numbers
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() || tokens[0].starts_with('#') {
            // skip empty lines and comments
            continue;
        }

        match tokens[0] {
            "o" => {
                if let Some(part) = current_part.take() {
                    data.object_parts.push(part);
                }
                current_part = Some(ObjectPart {
                    name: tokens.get(1).unwrap_or(&"unnamed").to_string(),
                    material_name: None,
                    faces: Vec::new(),
                });
            }
            "usemtl" => {
                if let Some(part) = current_part.as_mut() {
                    part.material_name = Some(tokens[1..].join(" "));
                }
            }
            "f" => {
                if let Some(part) = current_part.as_mut() {
                    // figure out the type of face
                    // type:1 [vertex] - f v1 v2 v3
                    // type:2 [vertex/texture] - f v1/vt1 v2/vt2 v3/vt3
                    // type:3 [vertex/texture/normal] - f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                    // type:4 [vertex/normal] - f v1//vn1 v2//vn2 v3//vn3
                    let mut face_type = 1;
                    if tokens[1].contains("//") {
                        face_type = 4; // vertex/normal
                    } else if tokens[1].contains("/") {
                        if tokens[1].matches("/").count() == 1 {
                            face_type = 2; // vertex/texture
                        } else if tokens[1].matches("/").count() == 2 {
                            face_type = 3; // vertex/texture/normal
                        }
                    } else {
                        face_type = 1; // vertex only
                    };

                    let mut face = Face {
                        vertex_indices: Vec::new(),
                        texture_indices: Vec::new(),
                        normal_indices: Vec::new(),
                    };
                    match face_type {
                        // NOTE: OBJ indices are 1-based, convert to 0-based
                        1 => {
                            // vertex only
                            for token in tokens.iter().skip(1) {
                                let index = token.parse::<u32>().map_err(|e| format!("Failed to parse vertex index: {e}"))?;
                                face.vertex_indices.push(index - 1);
                            }
                        }
                        2 => {
                            // vertex/texture
                            for token in tokens.iter().skip(1) {
                                let parts: Vec<&str> = token.split('/').collect();
                                if parts.len() < 2 {
                                    return Err("Invalid face format for vertex/texture".to_string());
                                }
                                let index = parts[0].parse::<u32>().map_err(|e| format!("Failed to parse vertex index: {e}"))?;
                                face.vertex_indices.push(index - 1);
                                let tex_index = parts[1].parse::<u32>().map_err(|e| format!("Failed to parse texture index: {e}"))?;
                                face.texture_indices.push(tex_index - 1);
                            }
                        }
                        3 => {
                            // vertex/texture/normal
                            for token in tokens.iter().skip(1) {
                                let parts: Vec<&str> = token.split('/').collect();
                                if parts.len() < 3 {
                                    return Err("Invalid face format for vertex/texture/normal".to_string());
                                }
                                let index = parts[0].parse::<u32>().map_err(|e| format!("Failed to parse vertex index: {e}"))?;
                                face.vertex_indices.push(index - 1);
                                let tex_index = parts[1].parse::<u32>().map_err(|e| format!("Failed to parse texture index: {e}"))?;
                                face.texture_indices.push(tex_index - 1);
                                let norm_index = parts[2].parse::<u32>().map_err(|e| format!("Failed to parse normal index: {e}"))?;
                                face.normal_indices.push(norm_index - 1);
                            }
                        }
                        4 => {
                            // vertex/normal
                            for token in tokens.iter().skip(1) {
                                let parts: Vec<&str> = token.split("//").collect();
                                if parts.len() < 2 {
                                    return Err("Invalid face format for vertex/normal".to_string());
                                }
                                let index = parts[0].parse::<u32>().map_err(|e| format!("Failed to parse vertex index: {e}"))?;
                                face.vertex_indices.push(index - 1);
                                let norm_index = parts[1].parse::<u32>().map_err(|e| format!("Failed to parse normal index: {e}"))?;
                                face.normal_indices.push(norm_index - 1);
                            }
                        }
                        _ => return Err("Unknown face type".to_string()),
                    }

                    part.faces.push(face);
                }
            }
            _ => (),
        }
    }

    // add the last part
    if let Some(part) = current_part.take() {
        data.object_parts.push(part);
    }

    Ok(data)
}


fn parse_mtl(mtl_path: &Path) -> Result<HashMap<String, Material>, String> {
    // parse the MTL file and return a vector of materials
    fn parse_color(tokens: &[&str]) -> Result<[f32; 3], String> {
        if tokens.len() < 3 {
            return Err("Not enough values for color".to_string());
        }
        let r = tokens[0].parse::<f32>().map_err(|e| format!("Failed to parse color value: {e}"))?;
        let g = tokens[1].parse::<f32>().map_err(|e| format!("Failed to parse color value: {e}"))?;
        let b = tokens[2].parse::<f32>().map_err(|e| format!("Failed to parse color value: {e}"))?;
        Ok([r, g, b])
    }

    let content = std::fs::read_to_string(mtl_path)
        .map_err(|e| format!("Failed to read MTL file: {e}"))?;

    let mut materials: Vec<Material> = Vec::new();
    let mut current_material: Option<Material> = None;
    for line in content.lines() {
        // split line into tokens/words/numbers
        let tokens = line.split_whitespace().collect::<Vec<&str>>();
        if tokens.is_empty() || tokens[0].starts_with('#') {
            // skip empty lines and comments
            continue;
        }
        // first token = tag, rest are values
        match tokens[0] {
            "newmtl" => {
                // material name
                let name = tokens[1..].join(" ");
                if let Some(material) = current_material.take() {
                    materials.push(material);
                }
                current_material = Some(Material::new(name));
            }
            "Ka" => {
                // ambient color
                if let Some(material) = current_material.as_mut() {
                    material.ambient_color = parse_color(&tokens[1..])?;
                }
            }
            "Kd" => {
                // diffuse color
                if let Some(material) = current_material.as_mut() {
                    material.diffuse_color = parse_color(&tokens[1..])?;
                }

            }
            "Ke" => {
                // emission color
                if let Some(material) = current_material.as_mut() {
                    material.emission_color = parse_color(&tokens[1..])?;
                }

            }
            "Pr" => {
                // roughness
                if let Some(material) = current_material.as_mut() {
                    material.roughness = tokens[1].parse::<f32>().map_err(|e| format!("Failed to parse roughness: {e}"))?;
                }
            }
            "Pl" => {
                // clearcoat
                if let Some(material) = current_material.as_mut() {
                    material.clearcoat = tokens[1].parse::<f32>().map_err(|e| format!("Failed to parse clearcoat: {e}"))?;
                }
            }
            "Pds" => {
                // sheen
                if let Some(material) = current_material.as_mut() {
                    material.sheen = tokens[1].parse::<f32>().map_err(|e| format!("Failed to parse sheen: {e}"))?;
                }
            }
            "Tf" => {
                // transmission filter
                if let Some(material) = current_material.as_mut() {
                    material.transmission_filter = parse_color(&tokens[1..])?;
                }
            }
            "Tr" => {
                // transparency
                if let Some(material) = current_material.as_mut() {
                    material.transparency = tokens[1].parse::<f32>().map_err(|e| format!("Failed to parse transparency: {e}"))?;
                }
            }
            "illum" => {
                // illumination model
                if let Some(material) = current_material.as_mut() {
                    material.illumination_model = tokens[1].parse::<u32>().map_err(|e| format!("Failed to parse illumination model: {e}"))?;
                }
            }
            _ => {
                let mut print = "".to_string();
                for token in tokens.iter() {
                    print = print + " {" + token + "}";
                }
                println!("Unknown material property: {print}");
            }
        }
    }

    let materials_map = materials.into_iter().map(|m| (m.name.clone(), m)).collect();
    Ok(materials_map)
}
