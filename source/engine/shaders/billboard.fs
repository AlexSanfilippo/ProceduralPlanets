#version 330 core

in vec2 v_tex_coord;

out vec4 frag_color;

uniform sampler2D u_texture;

void main()
{
    frag_color = texture(u_texture, v_tex_coord);
}

