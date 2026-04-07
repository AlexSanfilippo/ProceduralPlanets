#version 330 core
in vec3 normal;
in vec3 frag_pos;
out vec4 frag_color;

struct PointLight {
    vec3 position;
    float constant;
    float linear;
    float quadratic;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform PointLight point_light;
uniform vec3 view_pos;
uniform vec3 object_color;
uniform float shininess;
uniform float sphere_radius;


void main()
{

    vec3 norm = normalize(normal);
    vec3 light_dir = normalize(point_light.position - frag_pos);

    //hack for rainbow normal-vector planet
    vec3 object_color = norm;


    // Ambient
    vec3 ambient = point_light.ambient * object_color;

    // Diffuse
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = point_light.diffuse * diff * object_color;

    // Specular (Phong)
    vec3 view_dir = normalize(view_pos - frag_pos);
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    vec3 specular = point_light.specular * spec;

    // Attenuation
    float distance = length(point_light.position - frag_pos);
    float attenuation = 1.0 / (point_light.constant + point_light.linear * distance + point_light.quadratic * (distance * distance));

    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;

    vec3 result = ambient + diffuse + specular;
    frag_color = vec4(result, 1.0);
}