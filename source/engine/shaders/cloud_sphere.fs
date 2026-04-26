#version 330 core
in vec2 tex_coords;
in vec3 frag_pos;
in vec3 normal;
out vec4 frag_color;
uniform sampler2D cloud_texture;
uniform float cloud_strength;
uniform vec3 light_pos;
void main()
{
    vec4 tex_sample = texture(cloud_texture, tex_coords);
    float density = dot(tex_sample.rgb, vec3(0.299, 0.587, 0.114));
    vec3 norm = normalize(normal);
    vec3 light_dir = normalize(light_pos - frag_pos);
    float diff = max(dot(norm, light_dir), 0.0);
    float light_val = 0.05 + diff;
    vec3 cloud_color = vec3(1.0) * light_val;
    float alpha = density * cloud_strength;
    frag_color = vec4(cloud_color, alpha);
}
