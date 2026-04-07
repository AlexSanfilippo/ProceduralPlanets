#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 v_color;

void main() {
    vec4 world_pos = model * vec4(a_position, 1.0);
    v_color = a_color;
    gl_Position = projection * view * world_pos;
}
