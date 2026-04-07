#version 330 core
out vec4 frag_color;

in vec3 v_color;

uniform float alpha = 1.0;

void main() {
    frag_color = vec4(v_color, alpha);
}
