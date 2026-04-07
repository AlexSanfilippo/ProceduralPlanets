# version 330
in vec3 normal;
in vec3 frag_pos;
out vec4 frag_color;
void main()
{
    vec3 norm = normalize(normal);
    frag_color = vec4(abs(norm), 1.0);
}