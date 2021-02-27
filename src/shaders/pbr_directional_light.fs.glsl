#version 430

in vec3 vViewSpacePosition;
in vec3 vViewSpaceNormal;
in vec2 vTexCoords;

out vec3 fColour;

struct DirectionalLight
{
   vec4 dirView;
   vec4 radiance;
};

struct PBRMaterial
{
   vec4 baseColourFactor;
   vec4 emissiveFactor;
   double metallicFactor;
   double roughnessFactor;
   double occlusionStrength;
};

layout(std430) buffer sDirectionalLight
{
   DirectionalLight directional;
};

layout(std140) uniform bMaterial
{
   PBRMaterial material;
};

uniform sampler2D uBaseTexture;
uniform sampler2D uMetallicRoughnessTexture;
uniform sampler2D uEmissiveTexture;
uniform sampler2D uOcclusionTexture;

const float GAMMA = 2.2;
const float INV_GAMMA = 1.0 / GAMMA;
const float M_PI = 3.141592;
const float M_INV_PI = 1.0 / M_PI;

// see: https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-b-brdf-implementation
const vec3 DIELECTRIC_SPECULAR = vec3(0.04f);

float heavyside(float x) { return x > 0.f ? 1.f : 0.f; }

// Stolen here:
// https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/master/src/shaders/tonemapping.glsl
vec3 LINEARtoSRGB(vec3 colour) { return pow(colour, vec3(INV_GAMMA)); }
vec4 SRGBtoLINEAR(vec4 srgbIn) { return vec4(pow(srgbIn.rgb, vec3(GAMMA)), srgbIn.a); }

float computeMicrofacetDistribution(float NdotH, float alphaSquared)
{
   float denomRoot = NdotH * NdotH * (alphaSquared - 1) + 1;
   return denomRoot > 0.f
   ? M_INV_PI * alphaSquared * heavyside(NdotH) / (denomRoot * denomRoot)
   : 0.f;
}

float computeVisibilityFunction(float NdotL, float NdotV, float HdotL, float HdotV, float alphaSquared)
{
   float denom1 = abs(NdotL) + sqrt(alphaSquared + (1 - alphaSquared) * NdotL * NdotL);
   float denom2 = abs(NdotV) + sqrt(alphaSquared + (1 - alphaSquared) * NdotV * NdotV);
   return denom1 > 0.f && denom2 > 0.f
      ? (heavyside(HdotL) * heavyside(HdotV)) / (denom1 * denom2)
      : 0.f;
}

vec3 computeSpecularBrdf(float NdotL, float NdotV, float NdotH, float HdotL, float HdotV, float alphaSquared)
{
   return vec3(computeMicrofacetDistribution(NdotH, alphaSquared) * computeVisibilityFunction(NdotL, NdotV, HdotL, HdotV, alphaSquared));
}

vec3 computeDiffuseBrdf()
{
   return vec3(M_INV_PI);
}

void main()
{
   // Need another normalization because interpolation of vertex attributes does not maintain unit length
   const vec3 N = normalize(vViewSpaceNormal);
   const vec3 V = - normalize(vViewSpacePosition);
   const vec3 L = normalize(directional.dirView.xyz);
   const vec3 H = normalize(L + V);

    // colours
   vec3 baseColour = SRGBtoLINEAR(texture(uBaseTexture, vTexCoords)).rgb;
   baseColour *= material.baseColourFactor.rgb;

   vec3 emissiveColour = SRGBtoLINEAR(texture(uEmissiveTexture, vTexCoords)).rgb;
   emissiveColour *= material.emissiveFactor.rgb;

   // roughness and metal
   vec3 metallicRoughness = texture(uMetallicRoughnessTexture, vTexCoords).rgb;
   metallicRoughness *= vec3(1.f, material.roughnessFactor, material.metallicFactor);

   const float occlusion = texture(uOcclusionTexture, vTexCoords).r;

   const float roughness = metallicRoughness.g * float(material.roughnessFactor);
   const float metallic  = metallicRoughness.b;

   // some preliminary computations
   const float NdotL = clamp(dot(N, L), 0.f, 1.f);
   const float NdotV = clamp(dot(N, V), 0.f, 1.f);
   const float NdotH = clamp(dot(N, H), 0.f, 1.f);
   const float HdotL = clamp(dot(H, L), 0.f, 1.f);
   const float HdotV = clamp(dot(H, V), 0.f, 1.f);

   float shlickFactor = HdotV * HdotV;
   shlickFactor *= shlickFactor;
   shlickFactor *= HdotV;

   float alphaSquared = roughness * roughness;
   alphaSquared *= alphaSquared;

   const vec3 black = vec3(0.f);
   const vec3 f0 = mix(DIELECTRIC_SPECULAR, baseColour, metallic);
   const vec3 F = f0 + (1 - f0) * (1 - abs(shlickFactor));

   const vec3 colourDiff = mix(baseColour * (1 - DIELECTRIC_SPECULAR), black, metallic);

   vec3 diffuseColour  = (1 - F) * colourDiff;
   vec3 specularColour = F;


   // Apply bidirectional reflectance distribution functions (BRDF)
   diffuseColour   *= computeDiffuseBrdf();
   specularColour  *= computeSpecularBrdf(NdotL, NdotV, NdotH, HdotL, HdotV, alphaSquared);

   // Only diffuse indirect lighting should be affected by ambient occlusion.
   const vec3 occludedDiffuseColour = mix(diffuseColour, diffuseColour * occlusion, float(material.occlusionStrength));

   vec3 indirectColour = (occludedDiffuseColour + specularColour) * directional.radiance.rgb * NdotL;

   fColour = LINEARtoSRGB(indirectColour + emissiveColour);
}