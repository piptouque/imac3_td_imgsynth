#pragma once

#include "utils/GLFWHandle.hpp"
#include "utils/cameras.hpp"
#include "utils/filesystem.hpp"
#include "utils/shaders.hpp"

#include <tiny_gltf.h>

class ObjectModel;

class ViewerApplication
{
public:
  ViewerApplication(const fs::path &appPath, uint32_t width, uint32_t height,
      const std::vector<fs::path> & gltfFiles, const std::vector<float> &lookatArgs,
      const std::string &vertexShader, const std::string &fragmentShader,
      const fs::path &output);

  int run();

private:

  GLsizei m_nWindowWidth = 1280;
  GLsizei m_nWindowHeight = 720;

  const fs::path m_AppPath;
  const std::string m_AppName;
  const fs::path m_ShadersRootPath;


  std::vector<fs::path> m_gltfFilePaths;
  std::string m_vertexShader = "forward.vs.glsl";
  std::string m_fragmentShader = "pbr_directional_light.fs.glsl";

  bool m_hasUserCamera = false;
  Camera m_userCamera;

  fs::path m_OutputPath;

  // Order is important here, see comment below
  const std::string m_ImGuiIniFilename;
  // Last to be initialized, first to be destroyed:
  GLFWHandle m_GLFWHandle;
  /*
    ! THE ORDER OF DECLARATION OF MEMBER VARIABLES IS IMPORTANT !
    - m_ImGuiIniFilename.c_str() will be used by ImGUI in ImGui::Shutdown, which
    will be called in destructor of m_GLFWHandle. So we must declare
    m_ImGuiIniFilename before m_GLFWHandle so that m_ImGuiIniFilename
    destructor is called after.
    - m_GLFWHandle must be declared before the creation of any object managing
    OpenGL resources (e.g. GLProgram, GLShader) because it is responsible for
    the creation of a GLFW windows and thus a GL context which must exists
    before most of OpenGL function calls.
    As such, m_objectModels and other objects using OpenGL data
    must be declared after it.
  */
  std::vector<ObjectModel> m_objectModels;
  std::shared_ptr<GLProgram> m_pbrProgramme;
  std::shared_ptr<tinygltf::Material> m_boundMaterial;
};

// TD
class ObjectModel
{
public:
  /** -- methods -- **/
  ObjectModel(tinygltf::Model model,
      std::shared_ptr<GLProgram> programme,
      std::shared_ptr<tinygltf::Material> boundMaterial);
  ~ObjectModel();

  [[nodiscard]] inline const tinygltf::Model & getModel() const { return m_model; };

  void draw(GLuint materialBufferObject,
            const glm::mat4 & modelMatrix,
            const glm::mat4 & viewMatrix,
            const glm::mat4 & projMatrix,
            bool useOcclusion = true) const;
private:
  /** -- methods -- **/
  // A range of indices in a vector containing Vertex Array Objects
  struct VaoRange
  {
    GLsizei begin; // Index of first element in vertexArrayObjects
    GLsizei count; // Number of elements in range
  };

  void bindMaterial(GLuint materialBufferObject, int materialIdx, bool useOcclusion) const;

  /// \param model
  /// \return
  [[nodiscard]] static std::vector<GLuint> createBufferObjects(const tinygltf::Model & model);
  [[nodiscard]] static std::vector<GLuint> createVertexArrayObjects(
      const tinygltf::Model & model,
      const std::vector<GLuint> & bufferObjects,
      std::vector<VaoRange> & meshIndexToVaoRange);
  // This function might set texture sampler parameters if not specified.
  [[nodiscard]] static std::vector<GLuint> createTextureObjects(tinygltf::Model & model);

  [[nodiscard]] static GLuint getDefaultTextureObject();

  /** -- fields -- **/
  tinygltf::Model m_model;

  std::shared_ptr<GLProgram> m_programme;
  std::shared_ptr<tinygltf::Material> m_currentlyBoundMaterial;

  std::vector<GLuint> m_textureObjects;
  std::vector<GLuint> m_bufferObjects;
  std::vector<VaoRange> m_meshIndexToVaoRange;
  std::vector<GLuint> m_vertexArrayObjects;

  GLint m_baseTextureLocation;
  GLint m_metallicRoughnessTextureLocation;
  GLint m_emissionTextureLocation;
  GLint m_occlusionTextureLocation;
  GLint m_modelViewMatrixLocation;
  GLint m_modelViewProjMatrixLocation;
  GLint m_normalMatrixLocation;

  static GLuint s_defaultTextureObject;
};

class AmbientLight

{
public:
  AmbientLight() = default;
  virtual ~AmbientLight() = default;
  AmbientLight(glm::vec3 colour, float intensity) :
      m_colour(std::move(colour)), m_intensity(intensity)
  {
  }

  [[nodiscard]] inline glm::vec3 getRadiance() const
  {
    return m_colour * m_intensity;
  }
  [[nodiscard]] inline const glm::vec3 &getColour() const { return m_colour; }
  [[nodiscard]] inline float getIntensity() const { return m_intensity; }

  inline void setColour(glm::vec3 colour) { m_colour = glm::normalize(colour); }
  inline void setIntensity(float intensity) { m_intensity = intensity; }

private:
  glm::vec3 m_colour;
  float m_intensity;

};
class DirectionalLight : public AmbientLight
{
public:
  DirectionalLight() = default;
  DirectionalLight(glm::vec3 colour, float intensity, glm::vec3 dir)
      : AmbientLight(std::move(colour), intensity)
  {
    setDirection(dir);
  }

  [[nodiscard]] inline const glm::vec3 & getDirection() const { return m_dir; }
  [[nodiscard]] inline glm::vec2 getEulerAngles() const { return glm::vec2(m_theta, m_phi); }

  void setDirection(glm::vec3 a_dir)
  {
    m_dir = (1.f / a_dir.length()) * a_dir;
    const glm::vec2 euler = computeEulerAngles(m_dir);
    m_theta = euler.x;
    m_phi = euler.y;
  }

  void setDirection(float theta, float phi)
  {
    m_theta = theta;
    m_phi = phi;
    m_dir = computeDirection(m_theta, m_phi);
  }

private:
  [[nodiscard]] static glm::vec3 computeDirection(float theta, float phi)
  {
    const float sinTheta = glm::sin(theta);
    return glm::vec3(sinTheta * glm::cos(phi),
                     glm::cos(theta),
                     sinTheta * glm::sin(phi));
  }

  [[nodiscard]] static glm::vec2 computeEulerAngles(glm::vec3 dir)
  {
    // Not good.
    const float theta = glm::acos(dir.y);
    const float sinTheta = glm::sin(theta);
    const float phi = glm::abs(sinTheta) < std::numeric_limits<float>::min()
                      ? glm::half_pi<float>()
                      : glm::acos(dir.x / sinTheta);
    return glm::vec2(theta, phi);
  }

private:
  //
  glm::vec3 m_dir;
  // degrees
  float m_theta;
  float m_phi;
};

// OpenGL data
struct AmbientLightData
{
  glm::vec4 radiance;
};

struct DirectionalLightData
{
  glm::vec4 viewDir;
  glm::vec4 radiance;
};


struct MaterialData
{
  glm::vec4 baseColourFactor;
  glm::vec4 emissiveFactor;
  glm::float64 metallicFactor;
  glm::float64 roughnessFactor;
  glm::float64 occlusionStrength;
};
