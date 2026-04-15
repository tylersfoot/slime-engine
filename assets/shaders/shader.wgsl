// vertex shader

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
};
@group(2) @binding(0)
var<uniform> camera: Camera;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
    direction: vec3<f32>,
    inner_cutoff: f32,
    outer_cutoff: f32,
    light_view_proj: mat4x4<f32>,
}
@group(3) @binding(0)
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

    @location(12) instance_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) instance_color: vec4<f32>,
    @location(4) light_space_position: vec4<f32>,
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
    out.instance_color = instance.instance_color;
    out.light_space_position = light.light_view_proj * world_position;

    return out;
}

// fragment shader

@group(0) @binding(0)
var shadow_map: texture_depth_2d;
@group(0) @binding(1)
var shadow_sampler: sampler_comparison;

@group(1) @binding(0)
var diffuse_texture: texture_2d<f32>;
@group(1) @binding(1)
var diffuse_sampler: sampler;

struct MaterialUniforms {
    ambient_color: vec3<f32>,
    dissolve: f32,
    diffuse_color: vec3<f32>,
    specular_exponent: f32,
    specular_color: vec3<f32>,
    emissive_color: vec3<f32>,
};

@group(1) @binding(2)
var<uniform> material: MaterialUniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let texture_color = textureSample(diffuse_texture, diffuse_sampler, in.tex_coords);
    let object_color: vec4<f32> = texture_color * vec4<f32>(material.diffuse_color, 1.0) * in.instance_color;

    // normalized vectors (world space)
    let light_vector: vec3<f32> = normalize(light.position - in.world_position);
    let view_vector: vec3<f32> = normalize(camera.view_pos.xyz - in.world_position);
    let reflected_light_vector: vec3<f32> = reflect(-light_vector, normal);

    // shadow logic:
    // perspective divide
    let light_space_xyz = in.light_space_position.xyz / in.light_space_position.w;
    var shadow = 1.0;
    let biased_z = light_space_xyz.z;
    if light_space_xyz.z <= 1.0 && abs(light_space_xyz.x) <= 1.0 && abs(light_space_xyz.y) <= 1.0 {
        // convert to uv coordinates
        let shadow_uv = vec2<f32>(light_space_xyz.x * 0.5 + 0.5, -light_space_xyz.y * 0.5 + 0.5);
        shadow = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv, light_space_xyz.z);
    }

    // spotlight logic:
    // cosine of angle between light's aim and pixel
    let theta = dot(normalize(-light.direction), light_vector);
    let epsilon = light.inner_cutoff - light.outer_cutoff;
    // clamp intensity between 0.0-1.0 (outside-inside angles)
    let light_intensity = clamp((theta - light.outer_cutoff) / epsilon, 0.0, 1.0);

    // ambient lighting: baseline room lighting
    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength * material.ambient_color;

    // diffuse lighting: direct lighting from the light source
    let diffuse_strength = max(dot(light_vector, normal), 0.0);
    let diffuse_color = diffuse_strength 
                        * light.color * light_intensity * shadow;

    // specular lighting: the light source reflecting off objects into camera
    let specular_focus = clamp(material.specular_exponent, 1.0, 1000.0);
    let specular_strength = pow(max(dot(reflected_light_vector, view_vector), 0.0), specular_focus);
    let specular_color = specular_strength * material.specular_color 
                        * light.color * light_intensity * shadow;

    // let result: vec3<f32> = (ambient_color + diffuse_color) * object_color.rgb + specular_color;
    let result: vec3<f32> = (ambient_color + diffuse_color + specular_color) * object_color.rgb;
    return vec4<f32>(result, object_color.a);
}
