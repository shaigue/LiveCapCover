#version 330 core

const int MAX_JOINTS = 100;  // max number of joints allowed in a skeleton // TODO Izo: define this in config.py!
const int MAX_WEIGHTS = 3;  // max number of joints that can affect a vertex

in vec3 position;           // vertex
in vec2 texture_coords;
in vec3 normal;
in ivec3 joint_indices;
in vec3 weights; 

out vec2 pass_texture_coords;

uniform mat4 transformation_matrix;
uniform mat4 projection_matrix;
uniform mat4 view_matrix;               // camera

uniform mat4 joint_transforms[MAX_JOINTS];

void main(void) {
    
    vec4 total_local_pos = vec4(0.0);
    vec4 total_normal = vec4(0.0);
    for(int i=0; i<MAX_WEIGHTS; i++) {
        mat4 joint_transform = joint_transforms[joint_indices[i]];
        
        vec4 pose_position = joint_transform * vec4(position, 1.0);
        total_local_pos += pose_position * weights[i];

        vec4 world_normal = joint_transform * vec4(normal, 0.0);
        total_normal += world_normal * weights[i];
    }
    
    gl_Position = projection_matrix * view_matrix * transformation_matrix * total_local_pos;
    // gl_Position = projection_matrix * view_matrix * transformation_matrix * vec4(position, 1.0);
    pass_texture_coords = texture_coords;
}