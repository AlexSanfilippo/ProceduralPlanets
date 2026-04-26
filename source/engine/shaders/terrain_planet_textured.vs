#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texcoord;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

/* fbm_noise parameters (passed as uniforms) */
uniform int octaves;
uniform float lacunarity;
uniform float gain;
uniform float amplitude;
uniform float frequency;
uniform float seed;

out vec3 normal;
out vec3 frag_pos;
out vec2 tex_coords;

/* cheap 3D hash -> [0,1) */
float hash31(vec3 p)
{
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123);
}

/* smooth value noise in 3D returning [-1, 1] */
float noise3(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f); // smoothstep interpolation

    float n000 = hash31(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash31(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash31(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash31(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash31(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash31(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash31(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash31(i + vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);

    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);

    float nxyz = mix(nxy0, nxy1, u.z);

    return nxyz * 2.0 - 1.0; // map to [-1,1]
}

/* fbm_noise similar to the Python version: accumulates octaves */
float fbm_noise(vec3 p, int octs, float lac, float g, float amp, float freq, float sd)
{
    float value = 0.0;
    for (int i = 0; i < octs; ++i)
    {
        vec3 samplePos = p * freq / (float(i)/10 + 1.0f) + vec3(sd);
        float n = noise3(samplePos);
        value += amp * n / (float(i)/10 + 1.0f);
        freq *= lac;
        amp *= g;
        sd += 1;
    }
    return value;
}

void get_normal_triangle_points(vec3 point, float radius, float delta,
                                 out vec3 p1, out vec3 p2, out vec3 p3)
{
    vec3 n = normalize(point);
    vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 u = normalize(cross(up, n));
    vec3 v = cross(n, u);

    float halfChord = clamp(delta * 0.5, 0.0, 1.0);
    float theta = 2.0 * asin(halfChord);

    float ct = cos(theta);
    float st = sin(theta);

    const float TWO_PI_OVER_3 = 2.0943951023931954923;
    const float FOUR_PI_OVER_3 = 4.1887902047863909846;

    vec3 dir1 = cos(0.0) * u + sin(0.0) * v;
    vec3 dir2 = cos(TWO_PI_OVER_3) * u + sin(TWO_PI_OVER_3) * v;
    vec3 dir3 = cos(FOUR_PI_OVER_3) * u + sin(FOUR_PI_OVER_3) * v;

    p1 = radius * (ct * n + st * dir1);
    p2 = radius * (ct * n + st * dir2);
    p3 = radius * (ct * n + st * dir3);
}

vec3 triangle_normal(vec3 p1, vec3 p2, vec3 p3)
{
    vec3 e1 = p2 - p1;
    vec3 e2 = p3 - p1;

    vec3 raw = cross(e1, e2);
    float len = length(raw);

    if (len < 1e-6)
        return vec3(0.0, 1.0, 0.0);

    vec3 n = raw / len;

    if (dot(n, p1) < 0.0)
        n = -n;

    return n;
}

void main()
{
    float radius = 160.f;
    vec3 worldPos = vec3(model * vec4(a_position, 1.0));
    vec3 worldNormal = normalize(mat3(transpose(inverse(model))) * a_normal);

    float n = fbm_noise(worldPos, octaves, lacunarity, gain, amplitude, frequency, seed);

    // Blend between sigmoid (smooth valleys) and raw n (sharp mountains)
    float sig = 1.0 / (1.0 + exp(-n * 5.0));
    float t = clamp(n, 0.0, 1.0);
    n = mix(sig, n, t);

    vec3 displaced = worldPos + worldNormal * n;
    normal = worldNormal;

    vec3 p1, p2, p3;
    get_normal_triangle_points(worldPos, radius, 0.005, p1, p2, p3);
    p1 = p1 + worldNormal * fbm_noise(p1, octaves, lacunarity, gain, amplitude, frequency, seed);
    p2 = p2 + worldNormal * fbm_noise(p2, octaves, lacunarity, gain, amplitude, frequency, seed);
    p3 = p3 + worldNormal * fbm_noise(p3, octaves, lacunarity, gain, amplitude, frequency, seed);
    normal = triangle_normal(p1, p2, p3);

    frag_pos = displaced;
    tex_coords = a_texcoord * 1.f;
    gl_Position = projection * view * vec4(displaced, 1.0);
}

