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

uniform vec3 light_pos;

float sdf_sphere(vec3 p, vec3 center, float radius)
{
    return length(p - center) - radius;
}

vec3 estimate_normal(vec3 p, vec3 center, float radius)
{
    const float eps = 0.001;
    return normalize(vec3(
        sdf_sphere(p + vec3(eps, 0, 0), center, radius) - sdf_sphere(p - vec3(eps, 0, 0), center, radius),
        sdf_sphere(p + vec3(0, eps, 0), center, radius) - sdf_sphere(p - vec3(0, eps, 0), center, radius),
        sdf_sphere(p + vec3(0, 0, eps), center, radius) - sdf_sphere(p - vec3(0, 0, eps), center, radius)
    ));
}

const int   MAX_STEPS = 128;
const float MAX_DIST  = 50000.0;
const float SURF_DIST = 0.01;

void main()
{
    gl_FragDepth = 1.0;

    vec4 clip = vec4(frag_uv, -1.0, 1.0);
    vec4 eye  = inv_proj * clip;
    eye = vec4(eye.xy, -1.0, 0.0);
    vec3 ray_dir = normalize((inv_view * eye).xyz);
    vec3 ray_origin = cam_pos;

    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; ++i)
    {
        vec3 p = ray_origin + ray_dir * t;
        float d = sdf_sphere(p, sphere_center, sphere_radius);
        if (d < SURF_DIST)
        {
            vec3 hit = p;
            vec3 n   = estimate_normal(hit, sphere_center, sphere_radius);
            vec3 l   = normalize(light_pos - hit);
            float diff = max(dot(n, l), 0.0);
            float ambient = 0.12;
            vec3 color = sphere_color * (ambient + diff);
            FragColor = vec4(color, 1.0);

            // Write correct depth so SDF sphere is depth-tested against scene geometry
            vec4 clip_pos = proj * view * vec4(hit, 1.0);
            float ndc_depth = clip_pos.z / clip_pos.w;
            gl_FragDepth = (ndc_depth + 1.0) * 0.5;

            return;
        }
        if (t > MAX_DIST) break;
        t += d;
    }

    discard;
}
