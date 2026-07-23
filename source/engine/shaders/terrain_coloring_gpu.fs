#version 330 core
in vec3 normal;
in vec3 frag_pos;
in float tilt;
in float rainfall;
in float latitude;
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
uniform float sdf_radius;   // radius of the SDF ocean sphere; fragments below this are discarded
uniform int planet_type; // 0: heightmap, 1: mars-like, 2: earth-like
uniform int draw_hydrosphere; // 0: off, 1: on
uniform int view_mode;        // 0: terrain (deprecated), 1: normals, 2: heat map, 3: rainfall, 4: biomes, 5: biome true, 6: heightmap
uniform float global_temperature; // offset to heat value
uniform float global_rainfall_reduction; // planetwide rainfall reduction [0,1]
uniform sampler2D biome_map;       // X=temperature (cold->hot), Y=rainfall (dry->wet)
uniform sampler2D biome_true_map;  // same UV layout, photo-realistic colors


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

    // Apply planetwide rainfall reduction
    float rainfall_reduced = clamp(rainfall - global_rainfall_reduction, 0.0, 1.0);

    // Calculate base heat (without rainfall noise) and clamp rainfall to never exceed it
    float elevation_cooling = height_normalized * 0.80; //was 0.4
    float base_heat = clamp((1.0 - latitude) + global_temperature - elevation_cooling, 0.0, 1.0);
    rainfall_reduced = min(rainfall_reduced, base_heat);


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
        //color like the moon
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

        // Blend dark grey patches using rainfall noise
        if (draw_hydrosphere == 1) {
            vec3 dark_grey = vec3(0.30, 0.30, 0.32);
            float patch = smoothstep(0.35, 0.65, rainfall_reduced) * 0.25;
            object_color = mix(object_color, dark_grey, patch);
        }
    }

    //apply color gradient based on height with mars-like colors
    if (planet_type == 1){
        if (height_normalized < 0.5) {
            object_color = mix(vec3(0.5, 0.25, 0.1), vec3(0.8, 0.4, 0.2), (height_normalized - 0.1) / 0.4);
        } else {
            object_color = mix(vec3(0.8, 0.4, 0.2), vec3(1.0, 0.8, 0.6), (height_normalized - 0.5) / 0.5);
        }
        // Desaturate by 15%
        float lum = dot(object_color, vec3(0.299, 0.587, 0.114));
        object_color = mix(object_color, vec3(lum), 0.15);
    }

    //Earth-like colors
    if (planet_type == 2){
        // Elevation gradient: forest green (low) -> light green -> tan -> desert tan (high)
        vec3 elev_color;
        if (height_normalized < 0.3) {
            elev_color = mix(vec3(0.10, 0.35, 0.08), vec3(0.25, 0.55, 0.15), height_normalized / 0.3);
        } else if (height_normalized < 0.6) {
            elev_color = mix(vec3(0.25, 0.55, 0.15), vec3(0.65, 0.58, 0.35), (height_normalized - 0.3) / 0.3);
        } else {
            elev_color = mix(vec3(0.65, 0.58, 0.35), vec3(0.82, 0.72, 0.48), (height_normalized - 0.6) / 0.4);
        }

        // Steep slopes (high tilt) become grey rock regardless of elevation
        vec3 rock_color = vec3(0.50, 0.48, 0.45); // grey rock
        float slope_blend = smoothstep(0.35, 0.65, tilt);
        object_color = mix(elev_color, rock_color, slope_blend);
    }

    // Polar ice caps — blend to icy white near poles (for Earth and Mars)
    // Use heat value (accounts for global_temperature, elevation cooling, and noise) to drive ice
    if (planet_type == 1 || planet_type == 2) {
        float noise_offset = (rainfall_reduced - 0.5) * 0.15;
        float heat = clamp(base_heat + noise_offset, 0.0, 1.0);
        vec3 ice_color = vec3(0.92, 0.95, 0.98);
        float ice_blend = 1.0 - smoothstep(0.15, 0.30, heat);
        object_color = mix(object_color, ice_color, ice_blend);
    }




    // ---- View mode overrides ----
    if (view_mode == 1) {
        // Normals visualization: remap [-1,1] to [0,1]
        frag_color = vec4(norm * 0.5 + 0.5, 1.0);
        return;
    }
    if (view_mode == 2) {
        // Heat map: latitude 0 (equator) = hot, latitude 1 (pole) = cold
        // Elevation also cools: higher terrain is colder
        float noise_offset = (rainfall_reduced - 0.5) * 0.15; // perturb heat bands with noise
        float heat = clamp(base_heat + noise_offset, 0.0, 1.0);
        vec3 heat_color;
        if (heat < 0.25) {
            heat_color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), heat / 0.25);
        } else if (heat < 0.5) {
            heat_color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (heat - 0.25) / 0.25);
        } else if (heat < 0.75) {
            heat_color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (heat - 0.5) / 0.25);
        } else {
            heat_color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), (heat - 0.75) / 0.25);
        }
        frag_color = vec4(heat_color, 1.0);
        return;
    }
    if (view_mode == 3) {
        // Rainfall map: high rainfall = deep blue, mid = light blue, low-mid = green, low (dry) = white
        float r = rainfall_reduced; // already in [0,1]; 1=wet, 0=dry
        vec3 rain_color;
        if (r > 0.75) {
            rain_color = mix(vec3(0.10, 0.30, 0.80), vec3(0.02, 0.10, 0.50), (r - 0.75) / 0.25); // light blue -> deep blue
        } else if (r > 0.50) {
            rain_color = mix(vec3(0.40, 0.75, 0.95), vec3(0.10, 0.30, 0.80), (r - 0.50) / 0.25); // sky blue -> light blue
        } else if (r > 0.25) {
            rain_color = mix(vec3(0.20, 0.65, 0.20), vec3(0.40, 0.75, 0.95), (r - 0.25) / 0.25); // green -> sky blue
        } else {
            rain_color = mix(vec3(0.97, 0.97, 0.97), vec3(0.20, 0.65, 0.20), r / 0.25); // white -> green
        }
        frag_color = vec4(rain_color, 1.0);
        return;
    }
    if (view_mode == 4) {
        // Biomes: sample a biome map texture using temperature (U) and rainfall (V)
        // U=0 cold, U=1 hot; V=0 dry, V=1 wet
        float noise_offset = (rainfall_reduced - 0.5) * 0.15;
        float temp = clamp(base_heat + noise_offset, 0.0, 0.99);
        float rain = clamp(rainfall_reduced, 0.0, 0.99);
        rain = min(rain, temp);  // rainfall cannot exceed temperature
        vec2 biome_uv = vec2(temp, rain);
        vec3 biome_color = texture(biome_map, biome_uv).rgb;
        frag_color = vec4(biome_color, 1.0);
        return;
    }
    if (view_mode == 5) {
        // Biome True: same UV lookup but using photo-realistic color texture.
        // Does NOT return early — falls through to Phong lighting below.
        float noise_offset = (rainfall_reduced - 0.5) * 0.15;
        float temp = clamp(base_heat + noise_offset, 0.0, 0.99);
        float rain = clamp(rainfall_reduced, 0.0, 0.99);
        rain = min(rain, temp);  // rainfall cannot exceed temperature
            vec2 biome_uv = vec2(temp, rain);
        object_color = texture(biome_true_map, biome_uv).rgb;
        shininess = 4.0;
    }
    if (view_mode == 6) {
        // Heightmap: use the pre-computed object_color based on planet_type
        // Falls through to Phong lighting below
        shininess = 0.25;
    }

    // ...
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