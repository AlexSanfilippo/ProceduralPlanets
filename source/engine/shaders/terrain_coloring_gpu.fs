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
uniform int planet_type; // 0: heightmap, 1: mars-like, 2: earth-like


void main()
{

    vec3 norm = normalize(normal);
    vec3 light_dir = normalize(point_light.position - frag_pos);

    //get distance between point and radius of sphere
    float height = length(frag_pos) - sphere_radius;

    //map height to 0-1 range for coloring
    //float amplitude_guess = 2.0;
    float amplitude_guess = 1.;
    //float height_normalized = clamp((height) / amplitude_guess, 0.0, 1.0);
    float height_offset = -0.0; //adjust based on terrain
    //float height_normalized = (height / amplitude_guess) - height_offset;
    float height_normalized = clamp((height) / amplitude_guess, 0.0, 1.0);


    float shininess;
    shininess = 0.25;
    //apply color gradient based on height
    vec3 object_color;
    if (height_normalized < 0.3) {
        object_color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), height_normalized / 0.3); // Blue to Green
        shininess = 10.0;
    } else if (height_normalized < 0.89) {
        object_color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (height_normalized - 0.3) / 0.3); // Green to Yellow
    } else {
        object_color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (height_normalized - 0.6) / 0.4); // Yellow to Red
    }


    if (planet_type == 0) {
        float h = height_normalized;
        //color like venus
        if (h < 0.15) {
            object_color = mix(vec3(0.80), vec3(0.88), h / 0.15); // very light -> slightly darker
        } else if (h < 0.35) {
            object_color = mix(vec3(0.88), vec3(0.92), (h - 0.15) / 0.20);
        } else if (h < 0.55) {
            object_color = mix(vec3(0.92), vec3(0.96), (h - 0.35) / 0.20);
        } else if (h < 0.75) {
            object_color = mix(vec3(0.96), vec3(0.98), (h - 0.55) / 0.20);
        } else {
            object_color = mix(vec3(0.98), vec3(1.00), (h - 0.75) / 0.25); // lightest
        }
    }

    //apply color gradient based on height with mars-like colors
    if (planet_type == 1){
        if (height_normalized < 0.5) {
            object_color = mix(vec3(0.5, 0.25, 0.1), vec3(0.8, 0.4, 0.2), height_normalized / 0.5);
        } else {
            object_color = mix(vec3(0.8, 0.4, 0.2), vec3(1.0, 0.8, 0.6), (height_normalized - 0.5) / 0.5);
        }
    }

    //Earth-like colors
    if (planet_type == 2){
        if (height_normalized < 0.3) {
            object_color = mix(vec3(0.0, 0.0, 0.5), vec3(0.0, 0.5, 0.0), height_normalized / 0.3); // Deep Blue to Green
        } else if (height_normalized < 0.6) {
            object_color = mix(vec3(0.0, 0.5, 0.0), vec3(0.5, 0.25, 0.1), (height_normalized - 0.3) / 0.3); // Green to Brown
        } else {
            object_color = mix(vec3(0.5, 0.25, 0.1), vec3(1.0, 1.0, 1.0), (height_normalized - 0.6) / 0.4); // Brown to White
        }
    }




    //hack for rainbow normal-vector planet
    //vec3 object_color = norm;



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

    //testing normal
//     frag_color = vec4(normal, 1.0);
}