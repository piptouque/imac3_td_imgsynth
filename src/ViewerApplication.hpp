#pragma once

#include "utils/GLFWHandle.hpp"
#include "utils/cameras.hpp"
#include "utils/filesystem.hpp"
#include "utils/shaders.hpp"

#include <tiny_gltf.h>

class ViewerApplication
{
public:
  ViewerApplication(const fs::path &appPath, uint32_t width, uint32_t height,
      const fs::path &gltfFile, const std::vector<float> &lookatArgs,
      const std::string &vertexShader, const std::string &fragmentShader,
      const fs::path &output);

  int run();

private:


  GLsizei m_nWindowWidth = 1280;
  GLsizei m_nWindowHeight = 720;

  const fs::path m_AppPath;
  const std::string m_AppName;
  const fs::path m_ShadersRootPath;

  fs::path m_gltfFilePath;
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
  */
};

// TD
class ObjectModel
{
public:
  /** -- methods -- **/
  explicit ObjectModel(const std::experimental::filesystem::path & modelPath, const GLProgram & programme);
  ~ObjectModel();

  [[nodiscard]] inline const tinygltf::Model & getModel() const { return m_model; };

  void draw(const glm::mat4 & modelMatrix,
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

  void bindMaterial(int materialIdx, bool useOcclusion) const;

  ///
  /// \param model
  /// \return Success
  static bool loadGltfFile(const std::experimental::filesystem::path & path,
                           tinygltf::Model & model);
  ///

  /// \param model
  /// \return
  [[nodiscard]] static std::vector<GLuint> createBufferObjects(const tinygltf::Model & model);
  [[nodiscard]] static std::vector<GLuint> createVertexArrayObjects(
      const tinygltf::Model & model,
      const std::vector<GLuint> & bufferObjects,
      std::vector<VaoRange> & meshIndexToVaoRange);
  // This function might set texture sampler parameters if not specified.
  [[nodiscard]] static std::vector<GLuint> createTextureObjects(tinygltf::Model & model);
  [[nodiscard]] static GLuint createDefaultTextureObject();

  /** -- fields -- **/
  tinygltf::Model m_model;

  std::vector<GLuint> m_textureObjects;
  std::vector<GLuint> m_bufferObjects;
  std::vector<VaoRange> m_meshIndexToVaoRange;
  std::vector<GLuint> m_vertexArrayObjects ;

  GLint m_baseTextureLocation;
  GLint m_metallicRoughnessTextureLocation;
  GLint m_emissionTextureLocation;
  GLint m_occlusionTextureLocation;
  GLint m_modelViewMatrixLocation;
  GLint m_modelViewProjMatrixLocation;
  GLint m_normalMatrixLocation;

  GLuint m_materialBufferObject;
  // could be static, but whatever.
  GLuint m_defaultTextureObject;
};
