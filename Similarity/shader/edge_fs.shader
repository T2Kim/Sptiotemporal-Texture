#version 420

layout(location = 0) out vec4 color;
// layout(location = 0) out int color;
in vec4 color_out;

void main()

{
	color = color_out;
	// color = vec4(1.0, 1.0, 1.0, 1.0);
    //depth
	// color = int(1000.0 * color_out.x);

    // color.x = gl_FragCoord.z;
    // color.y = gl_FragCoord.z;
    // color.z = gl_FragCoord.z;
    // color.w = 1.0;
}