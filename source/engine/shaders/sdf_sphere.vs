#version 330 core

layout(location = 0) in vec3 a_position;

out vec2 frag_uv;

void main()
{
    // a_position is a fullscreen quad in NDC (-1..1)
    frag_uv = a_position.xy;
    gl_Position = vec4(a_position, 1.0);
}

