#version 430 core

in vec3 vVertexPosition;

out vec3 fFragColour;

//No lighting

void main()
{
	fFragColour = vec3(1.0, 0.0, 0.0);
}