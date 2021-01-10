#version 420

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color_in;
uniform mat4 MVP;
uniform mat4 MV;
uniform vec3 cam_pos;
out float color_out;

void main()
{
	gl_Position = MVP * vec4(position, 1.0);
	vec4 pp;
	pp = MV * vec4(position, 1.0);
	color_out = pp.z;
}