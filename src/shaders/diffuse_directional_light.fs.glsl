#version 430

#define M_PI 3.141592

in vec3 vViewSpacePosition;
in vec3 vViewSpaceNormal;
in vec2 vTexCoords;

out vec3 fColor;
struct DirectionalLight
{
   vec4 dir_vs;
   vec4 radiance;
};

layout(std430) buffer sDirectionalLight
{
   DirectionalLight directional;
};


vec3 computeRadianceDirectional(DirectionalLight light, vec3 viewNormal, vec3 viewDir)
{
   return vec3(1 / M_PI) * light.radiance.rgb * dot(viewNormal, light.dir_vs.xyz);
}

void main()
{

   // Need another normalization because interpolation of vertex attributes does not maintain unit length
   vec3 viewNormal = normalize(vViewSpaceNormal);
   vec3 viewDir = - normalize(vViewSpacePosition);

   fColor = computeRadianceDirectional(directional, viewNormal, viewDir); }