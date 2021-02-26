#version 430

in vec3 vViewSpacePosition;
in vec3 vViewSpaceNormal;
in vec2 vTexCoords;

out vec3 fColour;

struct DirectionalLight
{
   vec4 dir_vs;
   vec4 radiance;
};

layout(std430) buffer sDirectionalLight
{
   DirectionalLight directional;
};

uniform sampler2D uBaseColourTexture;

const float GAMMA = 2.2;
const float INV_GAMMA = 1.0 / GAMMA;
const float M_PI = 3.141592;
const float M_INV_PI = 1.0 / M_PI;


// Stolen here:
// https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/master/src/shaders/tonemapping.glsl
vec3 LINEARtoSRGB(vec3 colour) { return pow(colour, vec3(INV_GAMMA)); }
vec4 SRGBtoLINEAR(vec4 srgbIn) { return vec4(pow(srgbIn.rgb, vec3(GAMMA)), srgbIn.a); }


vec3 computeDiffuseDirectional(vec3 baseColour, DirectionalLight light, vec3 viewNormal, vec3 viewDir)
{
   return vec3(baseColour * M_INV_PI) * light.radiance.rgb * dot(viewNormal, light.dir_vs.xyz);
}

void main()
{
   // Need another normalization because interpolation of vertex attributes does not maintain unit length
   vec3 viewNormal = normalize(vViewSpaceNormal);
   vec3 viewDir = - normalize(vViewSpacePosition);


   vec4 baseColourTexture = SRGBtoLINEAR(texture(uBaseColourTexture, vTexCoords));
   // do shit

   fColour = LINEARtoSRGB(computeDiffuseDirectional(baseColourTexture.rgb, directional, viewNormal, viewDir));
}