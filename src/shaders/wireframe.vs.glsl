#version 330 core

layout(location = 0) in vec3 aPosition;

out vec3 vViewSpacePosition;

uniform mat4 uModelViewProjMatrix;
uniform mat4 uModelViewMatrix;


out vec3 vVertexPosition;

void main()
{
	vVertexPosition = vec3(uModelViewMatrix*vec4(aPosition, 1.f));
	gl_Position = uModelViewProjMatrix*vec4(aPosition, 1.f);
}
