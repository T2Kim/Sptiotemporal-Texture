#version 420

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal_in;
uniform mat4 MVP;
uniform mat4 MV;
uniform mat4 P;
uniform vec3 cam_pos;
uniform vec3 mode;
out vec4 color_out;

void main()
{

	// view_dir = position;
	gl_Position = MVP * vec4(position, 1.0);
	vec4 pp;
	pp = MV * vec4(normal_in, 0.0);
	if (pp.z < 0.7 && pp.z > -0.7)
		color_out = vec4(pp.xyz, 1.0);
	else
		color_out = vec4(0.0, 0.0, 0.0, 0.0);
	//color_out = vec4(pp.xyz, 1.0);
	// color_out.x = pp.z;
	// color_out.y = pp.z;
	// color_out.z = pp.z;
}