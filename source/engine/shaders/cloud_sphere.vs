#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texcoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 tex_coords;
out vec3 frag_pos;
out vec3 normal;

void main()
{
    vec4 world_pos = model * vec4(a_position, 1.0);
    frag_pos = world_pos.xyz;
    normal = normalize(mat3(transpose(inverse(model))) * a_normal);
    tex_coords = a_texcoord;
    gl_Position = projection * view * world_pos;
}

