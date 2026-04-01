// vertex shader

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: Camera;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}
@group(2) @binding(0)
var<uniform> light: Light;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
};
struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,

    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    // vector goes on right, and matrix goes on left in order of importance
    let world_position: vec4<f32> = model_matrix * vec4<f32>(model.position, 1.0);

    var out: VertexOutput;
    // we apply the camera's view projection to everything in world view (the model)

    out.tex_coords = model.tex_coords;
    out.world_normal = normal_matrix * model.normal;
    out.world_position = world_position.xyz;
    out.clip_position = camera.view_proj * world_position;

    return out;
}

// fragment shader

@group(0) @binding(0)
var diffuse_texture: texture_2d<f32>;
@group(0) @binding(1)
var diffuse_sampler: sampler;
@group(0) @binding(2)
var normal_texture: texture_2d<f32>;
@group(0) @binding(3)
var normal_sampler: sampler;
@group(0) @binding(4)
var specular_texture: texture_2d<f32>;
@group(0) @binding(5)
var specular_sampler: sampler;
@group(0) @binding(6)
var dissolve_texture: texture_2d<f32>;
@group(0) @binding(7)
var dissolve_sampler: sampler;
@group(0) @binding(8)
var ambient_texture: texture_2d<f32>;
@group(0) @binding(9)
var ambient_sampler: sampler;
@group(0) @binding(10)
var roughness_texture: texture_2d<f32>;
@group(0) @binding(11)
var roughness_sampler: sampler;
@group(0) @binding(12)
var metal_texture: texture_2d<f32>;
@group(0) @binding(13)
var metal_sampler: sampler;

struct MaterialUniforms {
    ambient_color: vec3<f32>,
    dissolve: f32,
    diffuse_color: vec3<f32>,
    specular_exponent: f32,
    specular_color: vec3<f32>,
    optical_density: f32,
    emissive_color: vec3<f32>,
    reflection_sharpness: f32,
    transmission_filter: vec3<f32>,
    metallic: f32,
    sheen: f32,
    clearcoat_thickness: f32,
    clearcoat_roughness: f32,
    anisotropy: f32,
    anisotropy_rotation: f32,
    illumination_model: u32,
};

@group(0) @binding(14)
var<uniform> material: MaterialUniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    
    var object_color: vec4<f32>;
    let normal_map: vec4<f32> = textureSample(normal_texture, normal_sampler, in.tex_coords);
    let specular_map = textureSample(specular_texture, specular_sampler, in.tex_coords);
    let metal_map = textureSample(metal_texture, metal_sampler, in.tex_coords);
    let ambient_map = textureSample(ambient_texture, ambient_sampler, in.tex_coords);
    let roughness_map = textureSample(roughness_texture, roughness_sampler, in.tex_coords);

    // checkered ground
    let grid = floor(in.world_position.xz);
    let checker = (i32(grid.x) + i32(grid.y)) % 2;
    if (in.world_position.y < -0.395) {
        if (checker == 0) {
            object_color = vec4<f32>(0.2, 0.2, 0.2, 1.0);
        } else {
            object_color = vec4<f32>(0.5, 0.5, 0.5, 1.0);
        }
    } else {
        let texture_color = textureSample(diffuse_texture, diffuse_sampler, in.tex_coords);
        object_color = texture_color * vec4<f32>(material.diffuse_color, 1.0);
    }

    // normalized vectors (world space)
    let light_vector: vec3<f32> = normalize(light.position - in.world_position);
    let view_vector: vec3<f32> = normalize(camera.view_pos.xyz - in.world_position);
    let reflected_light_vector: vec3<f32> = reflect(-light_vector, in.world_normal);

    // metalness: removes diffuse color and tints reflections
    let metalness = material.metallic * metal_map.r;
    let diffuse_albedo = object_color.xyz * (1.0-metalness);

    // ambient lighting: baseline room lighting, multiplied by ao map
    let ambient_scene_light = light.color * 0.1;
    let ambient_occlusion = ambient_map.r;
    let ambient_color = ambient_scene_light * ambient_occlusion * material.ambient_color;

    // diffuse lighting: direct lighting from the light source
    let diffuse_strength = max(dot(light_vector, in.world_normal), 0.0);
    let diffuse_color = light.color * diffuse_strength * diffuse_albedo;

    // specular lighting: the light source reflecting off objects into camera
    let smoothness = 1.0 - roughness_map.r;
    let specular_focus = clamp((smoothness * smoothness) * material.specular_exponent, 1.0, 1000.0);
    // F0 (4% reflectivity for non-metals)
    let dielectric_base = vec3<f32>(0.04, 0.04, 0.04);
    let specular_base = dielectric_base * material.specular_color * specular_map.rgb;
    // tints based on metalness (mixes between 4% grey and the base color of the metal)
    let specular_metal_tint = mix(specular_base, object_color.xyz, metalness);
    let specular_strength = pow(max(dot(reflected_light_vector, view_vector), 0.0), specular_focus);
    // energy conservation - multiply final color by "smoothness"
    let specular_color = specular_strength * specular_metal_tint * light.color * smoothness;


    // let result = (ambient_color + diffuse_color + specular_color) * object_color.xyz;
    // let result: vec3<f32> = specular_color;
    let result: vec3<f32> = (ambient_color * diffuse_albedo) + diffuse_color + specular_color;

    return vec4<f32>(result, object_color.a);
}
