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

// Scene color and depth textures from the opaque render pass (FBO)
uniform sampler2D scene_color_tex;
uniform sampler2D depth_tex;
uniform vec2      screen_size;

// Near and far clip planes — needed to linearise the depth buffer
uniform float near_plane;
uniform float far_plane;

// How deep (world-space units) before the water is fully opaque dark-blue
uniform float max_depth;

// ---------------------------------------------------------------------------
// Analytical ray-sphere intersection.
// Returns vec2(t_near, t_far).  If no hit, t_near > t_far.
// ---------------------------------------------------------------------------
vec2 intersect_sphere(vec3 ro, vec3 rd, vec3 center, float radius)
{
    vec3  oc = ro - center;
    float b  = dot(oc, rd);
    float c  = dot(oc, oc) - radius * radius;
    float h  = b * b - c;
    if (h < 0.0) return vec2(1.0, -1.0);   // no intersection
    h = sqrt(h);
    return vec2(-b - h, -b + h);
}

// ---------------------------------------------------------------------------
// Convert a non-linear depth-buffer value [0,1] to a linear eye-space depth.
// This is the GLSL equivalent of Unity's LinearEyeDepth().
// For a standard perspective projection:
//   z_buffer = (f+n)/(f-n)/2 + 0.5 + f*n/(f-n) / z_eye  (roughly)
// The exact inversion for OpenGL's [0,1] mapped depth is:
//   linearDepth = (2 * near * far) / (far + near - (2*d - 1) * (far - near))
// ---------------------------------------------------------------------------
float linearise_depth(float d)
{
    float ndc = d * 2.0 - 1.0;                     // [0,1] -> [-1,1]
    return (2.0 * near_plane * far_plane) /
           (far_plane + near_plane - ndc * (far_plane - near_plane));
}

void main()
{
    // ---- sample the scene colour and depth from the FBO ----------------
    vec2  screen_uv   = gl_FragCoord.xy / screen_size;
    vec4  scene_color  = texture(scene_color_tex, screen_uv);
    float raw_depth    = texture(depth_tex, screen_uv).r;

    // ---- build the world-space ray ------------------------------------
    vec4 clip_ray = vec4(frag_uv, -1.0, 1.0);
    vec4 eye_ray  = inv_proj * clip_ray;
    eye_ray       = vec4(eye_ray.xy, -1.0, 0.0);
    vec3 ray_dir  = normalize((inv_view * eye_ray).xyz);
    vec3 ray_origin = cam_pos;

    // ---- linearise the depth buffer (Sebastian Lague's approach) ------
    // 1. Convert non-linear depth -> linear eye depth (perpendicular dist)
    float linear_eye_depth = linearise_depth(raw_depth);

    // 2. The eye_ray before normalisation gives us the view-space direction
    //    whose z = -1.  Its length tells us how much longer the actual ray
    //    is compared to the perpendicular depth.  Multiplying gives the
    //    true world-space distance along the ray to the scene surface.
    //    viewVector = (inv_view * vec4(eye_unnorm.xy, -1, 0)).xyz
    //    length(viewVector) is the correction factor.
    vec4 eye_unnorm = inv_proj * clip_ray;
    eye_unnorm      = vec4(eye_unnorm.xy, -1.0, 0.0);
    vec3 view_vec   = (inv_view * eye_unnorm).xyz;
    float scene_dist = linear_eye_depth * length(view_vec);

    // ---- analytical ray-sphere intersection (ocean surface) ----------
    vec2 ocean_hit = intersect_sphere(ray_origin, ray_dir, sphere_center, sphere_radius);
    float entry_t = ocean_hit.x;
    float exit_t  = ocean_hit.y;

    // No ocean hit — just output the scene colour
    if (entry_t > exit_t || exit_t < 0.0)
    {
        FragColor = scene_color;
        return;
    }
    if (entry_t < 0.0) entry_t = 0.0;               // camera inside sphere

    // If the scene geometry is in front of the ocean entry, skip water
    if (scene_dist < entry_t)
    {
        FragColor = scene_color;
        return;
    }

    vec3 entry_hit = ray_origin + ray_dir * entry_t;

    // ---- water depth (world-space, simple subtraction) ----------------
    // How far into the ocean does the terrain sit?
    // scene_dist  = world-space ray distance to terrain surface
    // entry_t     = world-space ray distance to ocean entry
    // exit_t      = world-space ray distance to ocean exit
    // Water depth = distance from ocean entry to the terrain (or ocean exit
    //               if the terrain is behind the entire ocean shell).
    float water_depth = min(scene_dist, exit_t) - entry_t;
    water_depth = max(water_depth, 0.0);

    // ---- lighting on the entry surface normal ------------------------
    vec3 n     = normalize(entry_hit - sphere_center);
    vec3 v     = normalize(cam_pos - entry_hit);
    vec3 l     = normalize(light_pos - entry_hit);
    float diff = max(dot(n, l), 0.0);
    float ambient = 0.0;
    float light_val = ambient + diff;

    // ---- colour blending based on depth ------------------------------
    vec3 deep_color    = vec3(0.01, 0.05, 0.35);
    vec3 shallow_color = vec3(0.15, 0.55, 0.80);

    float depth_t      = clamp(water_depth / max_depth, 0.0, 1.0);
    float depth_curved = pow(depth_t, 0.35);

    vec3 water_color = mix(shallow_color, deep_color, depth_curved) * light_val;

    // specular highlight
    vec3  h    = normalize(l + v);
    float spec = pow(max(dot(n, h), 0.0), 256.0) * 0.6;
    water_color += vec3(spec);

    // ---- alpha / compositing -----------------------------------------
    float min_alpha   = 0.0; //experimental
    float max_alpha   = 1.0 - transparency;
    float final_alpha = mix(min_alpha, max_alpha, depth_curved);

    // Manually blend water over the scene colour
    vec3 blended = mix(scene_color.rgb, water_color, final_alpha);
    FragColor = vec4(blended, 1.0);
}
