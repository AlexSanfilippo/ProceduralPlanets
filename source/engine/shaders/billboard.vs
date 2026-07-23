#version 330 core

// Per-vertex attributes
layout(location = 0) in vec2 local_pos;     // Local quad position (-1..1)
layout(location = 1) in vec2 tex_coord;     // Texture coordinates (0..1)

// Per-instance attributes
layout(location = 2) in vec3 instance_pos;  // World-space position of this billboard

// Uniforms
uniform mat4 view;
uniform mat4 projection;
uniform float scale;

// Output to fragment shader
out vec2 v_tex_coord;

void main()
{
    // Extract camera's right and up vectors from the view matrix
    // view matrix column 0 is the right vector (negated)
    // view matrix column 1 is the up vector
    vec3 camera_right = vec3(view[0][0], view[1][0], view[2][0]);
    vec3 camera_up = vec3(view[0][1], view[1][1], view[2][1]);

    // Build the quad vertex position in world space
    // The quad is constructed from the instance position +/- scaled camera vectors
    vec3 world_pos = instance_pos
                   + camera_right * local_pos.x * scale
                   + camera_up * local_pos.y * scale;

    // Transform to clip space
    gl_Position = projection * view * vec4(world_pos, 1.0);

    // Pass texture coordinates to fragment shader
    v_tex_coord = tex_coord;
}

