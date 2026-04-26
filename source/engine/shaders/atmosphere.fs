#version 330 core

in vec2 frag_uv;
out vec4 FragColor;

uniform vec3  cam_pos;
uniform mat4  inv_view;
uniform mat4  inv_proj;
uniform mat4  view;
uniform mat4  proj;

uniform vec3  sphere_center;
uniform float sphere_radius;
uniform vec3  sphere_color;
uniform float transparency;

uniform vec3 light_pos;

uniform sampler2D scene_color_tex;
uniform sampler2D depth_tex;
uniform vec2      screen_size;

uniform float near_plane;
uniform float far_plane;

uniform float min_depth;
uniform float max_depth;

vec2 intersect_sphere(vec3 ro, vec3 rd, vec3 center, float radius)
{
    vec3  oc = ro - center;
    float b  = dot(oc, rd);
    float c  = dot(oc, oc) - radius * radius;
    float h  = b * b - c;
    if (h < 0.0) return vec2(1.0, -1.0);
    h = sqrt(h);
    return vec2(-b - h, -b + h);
}

float linearise_depth(float d)
{
    float ndc = d * 2.0 - 1.0;
    return (2.0 * near_plane * far_plane) /
           (far_plane + near_plane - ndc * (far_plane - near_plane));
}

void main()
{
    vec2  screen_uv   = gl_FragCoord.xy / screen_size;
    vec4  scene_color  = texture(scene_color_tex, screen_uv);
    float raw_depth    = texture(depth_tex, screen_uv).r;

    // Build world-space ray
    vec4 clip_ray = vec4(frag_uv, -1.0, 1.0);
    vec4 eye_ray  = inv_proj * clip_ray;
    eye_ray       = vec4(eye_ray.xy, -1.0, 0.0);
    vec3 ray_dir  = normalize((inv_view * eye_ray).xyz);
    vec3 ray_origin = cam_pos;

    // Linearise depth
    float linear_eye_depth = linearise_depth(raw_depth);
    vec4 eye_unnorm = inv_proj * clip_ray;
    eye_unnorm      = vec4(eye_unnorm.xy, -1.0, 0.0);
    vec3 view_vec   = (inv_view * eye_unnorm).xyz;
    float scene_dist = linear_eye_depth * length(view_vec);

    // Ray-sphere intersection
    vec2 atmo_hit = intersect_sphere(ray_origin, ray_dir, sphere_center, sphere_radius);
    float entry_t = atmo_hit.x;
    float exit_t  = atmo_hit.y;

    // No hit
    if (entry_t > exit_t || exit_t < 0.0)
    {
        FragColor = scene_color;
        return;
    }
    if (entry_t < 0.0) entry_t = 0.0;  // camera inside atmosphere

    // The ray travels through the atmosphere shell.
    // Clamp exit by scene geometry so we only count the visible portion.
    float effective_exit = min(exit_t, scene_dist);
    float path_length = effective_exit - entry_t;
    path_length = max(path_length, 0.0);

    // Normalise path length: short paths (looking straight down) -> transparent,
    // long paths (looking at horizon) -> coloured.
    float depth_t = clamp((path_length - min_depth) / (max_depth - min_depth), 0.0, 1.0);
    // Use a power curve so it stays transparent for most angles and only
    // shows colour near the horizon / limb.
    float atmo_factor = pow(depth_t, 3.0);

    // Lighting: simple diffuse on the entry point normal
    vec3 entry_hit = ray_origin + ray_dir * entry_t;
    vec3 n     = normalize(entry_hit - sphere_center);
    vec3 l     = normalize(light_pos - entry_hit);
    float diff = max(dot(n, l), 0.0);
    float light_val = diff;

    vec3 atmo_color = sphere_color * light_val;

    float final_alpha = atmo_factor * (1.0 - transparency);

    vec3 blended = mix(scene_color.rgb, atmo_color, final_alpha);
    FragColor = vec4(blended, 1.0);
}

