#version 330 core

in vec2 frag_uv;
in vec2 ndc_pos;
out vec4 FragColor;

uniform sampler2D scene_color;
uniform sampler2D scene_depth;

// Camera / projection uniforms for ray construction
uniform vec3 cam_pos;
uniform mat4 inv_view;
uniform mat4 inv_proj;

// Near / far planes for depth linearisation
uniform float near_plane;
uniform float far_plane;

// Simple SDF sphere parameters
uniform vec3  sdf_center;
uniform float sdf_radius;

// Screen dimensions (for depth texture sampling)
uniform vec2 screen_size;

// --- helpers ---------------------------------------------------------------
float linearise_depth(float d)
{
    float ndc = d * 2.0 - 1.0;
    return (2.0 * near_plane * far_plane) /
           (far_plane + near_plane - ndc * (far_plane - near_plane));
}

float sdf_sphere(vec3 p, vec3 center, float radius)
{
    return length(p - center) - radius;
}

// ---------------------------------------------------------------------------
void main()
{


    vec4 base_color = texture(scene_color, frag_uv);

    // ---- build world-space ray ----------------------------------------
    vec4 clip_ray = vec4(ndc_pos, -1.0, 1.0);
    vec4 eye_ray  = inv_proj * clip_ray;
    eye_ray       = vec4(eye_ray.xy, -1.0, 0.0);
    vec3 ray_dir  = normalize((inv_view * eye_ray).xyz);

    // ---- linearise scene depth ----------------------------------------
//     float raw_depth       = texture(scene_depth, frag_uv).r;
    float raw_depth       = gl_FragCoord.z;
    //float linear_eye_dep  = linearise_depth(raw_depth);
    float linear_eye_dep  = gl_FragCoord.z;
    vec3  view_vec        = (inv_view * vec4((inv_proj * clip_ray).xy, -1.0, 0.0)).xyz;
    float scene_dist      = linear_eye_dep * length(view_vec);

    // ---- ray-march the SDF sphere -------------------------------------
    const int   MAX_STEPS = 64;
    const float MAX_DIST  = 50000.0;
    const float SURF_DIST = 0.01;

    float t = 0.0;
    bool  hit = false;

    for (int i = 0; i < MAX_STEPS; ++i)
    {
        vec3  p = cam_pos + ray_dir * t;
        float d = sdf_sphere(p, sdf_center, sdf_radius);
        if (d < SURF_DIST)
        {
            hit = true;
            break;
        }
        if (t > MAX_DIST) break;
        t += d;
    }

    if (hit && t < scene_dist)
    {
        vec4 soft_pink = vec4(1.0, 0.75, 0.8, 1.0);
        FragColor = soft_pink;
        //FragColor = vec4(0.0, 1.0, 1.0, 1.0);   // simple white sphere
    }
    else
    {
        FragColor = base_color;
    }
//     FragColor = base_color;
}
