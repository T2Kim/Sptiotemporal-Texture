#version 420

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color_in;
uniform mat4 MVP;
uniform mat4 MV;
uniform mat4 P;
uniform vec3 cam_pos;
uniform vec3 mode;
out vec3 color_out;

void main()
{

	// view_dir = position;
	gl_Position = MVP * vec4(position, 1.0);
	vec4 pp;
	pp = MV * vec4(position, 1.0);
	color_out = color_in;
	// color_out.x = pp.z;
	// color_out.y = pp.z;
	// color_out.z = pp.z;
}