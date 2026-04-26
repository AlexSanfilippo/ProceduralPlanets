#version 330 core

layout(location = 0) in vec3 a_position;

out vec2 frag_uv;
out vec2 ndc_pos;   // NDC [-1,1] for ray construction

void main()
{
    frag_uv  = a_position.xy * 0.5 + 0.5;  // map from [-1,1] to [0,1]
    ndc_pos  = a_position.xy;               // keep NDC for ray-marching
    gl_Position = vec4(a_position, 1.0);
}

