#version 420

layout(location = 0) out int color;
in float color_out;

void main()

{
    //depth
	color = int(1000.0 * color_out);
}