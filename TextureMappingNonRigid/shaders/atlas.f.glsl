varying vec3 varyingImg_coord;
varying vec2 varyingAtlas_coord;
uniform sampler2D tempImg;
uniform sampler2D bufImg;

void main(){
	vec2 img_coord = varyingImg_coord.xy;
	//gl_FragColor = vec4(1.0,1.0,1.0,1.0);
	//gl_FragColor = vec4(texture2D(tempImg, img_coord).rgb, 1.0);


	if(varyingImg_coord.z < 2)
		//gl_FragColor =  vec4(texture2D(bufImg, vec2(varyingAtlas_coord.x, varyingAtlas_coord.y)).rgb + varyingImg_coord.z*(texture2D(tempImg, img_coord).rgb), 1.0);
		gl_FragColor = texture2D(bufImg, vec2(varyingAtlas_coord.x, varyingAtlas_coord.y)).rgba + vec4(varyingImg_coord.z*(texture2D(tempImg, img_coord).rgb), 1);
	else {
		int label = int(varyingImg_coord.z) - 3;
		if (label < 0)
			gl_FragColor = vec4(texture2D(bufImg, vec2(varyingAtlas_coord.x, varyingAtlas_coord.y)).rgb, 1.0);
		else {
			int key = label - int(6.0 * floor(float(label) / 6.0));
			label = int(floor(float(label) / 6.0));
			float l_color = float(label) / 100.0;
			if (key == 0) {
				gl_FragColor = vec4(texture2D(bufImg, vec2(varyingAtlas_coord.x, varyingAtlas_coord.y)).rgb + vec3(0.0, 0.0, 1.0 - l_color), 1.0);
			}
			else if (key == 1) {
				gl_FragColor = vec4(texture2D(bufImg, vec2(varyingAtlas_coord.x, varyingAtlas_coord.y)).rgb + vec3(0.0, 1.0 - l_color, 0.0), 1.0);
			}
			else if (key == 2) {
				gl_FragColor = vec4(texture2D(bufImg, vec2(varyingAtlas_coord.x, varyingAtlas_coord.y)).rgb + vec3(0.0, 1.0 - l_color, 1.0 - l_color), 1.0);
			}
			else if (key == 3) {
				gl_FragColor = vec4(texture2D(bufImg, vec2(varyingAtlas_coord.x, varyingAtlas_coord.y)).rgb + vec3(1.0 - l_color, 0.0, 0.0), 1.0);
			}
			else if (key == 4) {
				gl_FragColor = vec4(texture2D(bufImg, vec2(varyingAtlas_coord.x, varyingAtlas_coord.y)).rgb + vec3(1.0 - l_color, 0.0, 1.0 - l_color), 1.0);
			}
			else if (key == 5) {
				gl_FragColor = vec4(texture2D(bufImg, vec2(varyingAtlas_coord.x, varyingAtlas_coord.y)).rgb + vec3(1.0 - l_color, 1.0 - l_color, 0.0), 1.0);
			}
		}
	}

	//gl_FragColor = texture2D(bufImg, vec2((gl_PointCoord.x+1.0)/2.0,(-gl_PointCoord.y+1.0)/2.0))+varyingImg_coord.z*texture2D(tempImg, varyingImg_coord.xy*2-1);
	//gl_FragColor = texture2D(bufImg, vec2((gl_FragCoord.x+1.0)/2.0,(gl_FragCoord.y+1.0)/2.0))+varyingImg_coord.z*texture2D(tempImg, varyingImg_coord.xy);
	
	//gl_FragColor =vec4(varyingImg_coord.z*texture2D(tempImg, varyingImg_coord.xy).rgb, 1.0);

}