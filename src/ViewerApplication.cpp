#include "ViewerApplication.hpp"

#include <iostream>
#include <numeric>
#include <future> // todo: async.

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "utils/cameras.hpp"
#include "utils/gltf.hpp"
#include "utils/images.hpp"

#include <misc/cpp/imgui_stdlib.h>

#include <stb_image_write.h>

// TD

static constexpr GLuint VERTEX_ATTRIB_POSITION_IDX  = 0;
static constexpr GLuint VERTEX_ATTRIB_NORMAL_IDX    = 1;
static constexpr GLuint VERTEX_ATTRIB_TEXCOORD0_IDX = 2;

static constexpr const char* DIRECTIONALLIGHT_STORAGE_BLOCK_NAME = "sDirectionalLight";
static constexpr GLuint DIRECTIONALLIGHT_STORAGE_BINDING = 1;

static constexpr const char* MATERIAL_UNIFORM_BUFFER_NAME = "bMaterial";
static constexpr GLuint MATERIAL_BLOCK_BINDING = 1;

static constexpr const char* AMBIENTLIGHT_UNIFORM_BUFFER_NAME = "bAmbientLight";
static constexpr GLuint AMBIENTLIGHT_BLOCK_BINDING = 2;

static constexpr const char* MVP_MATRIX_UNIFORM_NAME = "uModelViewProjMatrix";
static constexpr const char* MV_MATRIX_UNIFORM_NAME = "uModelViewMatrix";
static constexpr const char* N_MATRIX_UNIFORM_NAME = "uNormalMatrix";
static constexpr const char* BASE_TEX_UNIFORM_NAME = "uBaseTexture";
static constexpr const char* MR_TEX_UNIFORM_NAME = "uMetallicRoughnessTexture";
static constexpr const char* EM_TEX_UNIFORM_NAME = "uEmissiveTexture";
static constexpr const char* OC_TEX_UNIFORM_NAME = "uOcclusionTexture";

namespace {

bool loadGltfFile(const fs::path & path, tinygltf::Model & model) {
  tinygltf::TinyGLTF loader;
  std::string err, warn;
  bool success;
  const std::string filePath = path.string();
  if (path.has_extension() && path.extension().string() == ".glb")
  {
    // Load as binary
    success = loader.LoadBinaryFromFile(
        &model, &err, &warn,
        filePath);
  }
  else
  {
    // load as text.
    success = loader.LoadASCIIFromFile(
        &model, &err, &warn,
        filePath);
  }
  if (!err.empty()) {
    std::cerr << "Error while parsing GLTF file: " << err << std::endl;
  }
  if (!warn.empty()) {
    std::cerr << "Warning while parsing GLTF file: " << warn << std::endl;
  }
  if (!success) {
    std::cerr << "Failed to parse GLTF file at " << path << std::endl;
  }
  return success;
}

const tinygltf::Accessor & findAccessor(const tinygltf::Model & model, int accessorIdx) {
    return model.accessors.at(static_cast<std::size_t>(accessorIdx));
  }

  const tinygltf::BufferView & findBufferView(const tinygltf::Model & model, const tinygltf::Accessor & accessor) {
    // get the correct tinygltf::Accessor from model.accessors
    const auto bufferViewIdx = static_cast<std::size_t>(accessor.bufferView);
    // get the correct tinygltf::BufferView from model.bufferViews.
    return model.bufferViews.at(bufferViewIdx);
  }

  GLuint findBufferObject(const std::vector<GLuint> & bufferObjects, const tinygltf::BufferView & bufferView) {
    // get the index of the buffer used by the bufferView
    const auto bufferIdx = static_cast<std::size_t>(bufferView.buffer);
    return bufferObjects.at(bufferIdx);
  }

  std::size_t getByteOffset(const tinygltf::Accessor & accessor, const tinygltf::BufferView & bufferView) {
    return accessor.byteOffset + bufferView.byteOffset;
  }

  void keyCallback(
      GLFWwindow *window, int key, [[maybe_unused]] int scancode,
      int action, [[maybe_unused]] int mods)
  {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
      glfwSetWindowShouldClose(window, 1);
    }
  }
}


int ViewerApplication::run()
{
  // Loader shaders
  m_pbrProgramme = std::make_shared<GLProgram>(
      compileProgram({m_ShadersRootPath / m_vertexShader,
          m_ShadersRootPath / m_fragmentShader}));
  m_boundMaterial = std::make_shared<tinygltf::Material>();

  GLuint directionalLightBufferObject;
  // Directional SSBO stuff
  {
    glGenBuffers(1, &directionalLightBufferObject);

    const GLuint lightStorageBlockIndex =
        glGetProgramResourceIndex(m_pbrProgramme->glId(), GL_SHADER_STORAGE_BLOCK, DIRECTIONALLIGHT_STORAGE_BLOCK_NAME);
    assert(lightStorageBlockIndex != GL_INVALID_INDEX);

    // Alloc and binding
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, directionalLightBufferObject);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(sizeof(DirectionalLightData) * 1),
        nullptr,
        GL_DYNAMIC_DRAW
      );
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Linking to shader.
    glShaderStorageBlockBinding(m_pbrProgramme->glId(), lightStorageBlockIndex,
        DIRECTIONALLIGHT_STORAGE_BINDING);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, DIRECTIONALLIGHT_STORAGE_BINDING,
        directionalLightBufferObject);
  }
  // Ambient UBO stuff
  GLuint ambientLightBufferObject;
  {
    glGenBuffers(1 , &ambientLightBufferObject);

    glBindBuffer(GL_UNIFORM_BUFFER, ambientLightBufferObject);
    glBufferData(GL_UNIFORM_BUFFER,
                 sizeof(AmbientLightData),
                 nullptr,
                 GL_DYNAMIC_READ
                );
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint ambientBlockIndex = glGetUniformBlockIndex(
        m_pbrProgramme->glId(),
        AMBIENTLIGHT_UNIFORM_BUFFER_NAME);

    assert(ambientBlockIndex != GL_INVALID_INDEX);

    glUniformBlockBinding(
        m_pbrProgramme->glId(), ambientBlockIndex, AMBIENTLIGHT_BLOCK_BINDING);

    glBindBufferBase(GL_UNIFORM_BUFFER, AMBIENTLIGHT_BLOCK_BINDING, ambientLightBufferObject);
  }

  // Material
  GLuint materialBufferObject;
  {
    glGenBuffers(1 , &materialBufferObject);

    // alloc on GPU
    glBindBuffer(GL_UNIFORM_BUFFER, materialBufferObject);
    glBufferData(GL_UNIFORM_BUFFER,
                 sizeof(MaterialData),
                 nullptr,
                 GL_DYNAMIC_READ
                );
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint materialBlockIndex = glGetUniformBlockIndex(
        m_pbrProgramme->glId(),
        MATERIAL_UNIFORM_BUFFER_NAME);

    assert(materialBlockIndex != GL_INVALID_INDEX);

    glUniformBlockBinding(
        m_pbrProgramme->glId(), materialBlockIndex, MATERIAL_BLOCK_BINDING);

    glBindBufferBase(GL_UNIFORM_BUFFER, MATERIAL_BLOCK_BINDING, materialBufferObject);
  }

  // first load the models in order to set camera according to scene bounds.
  m_objectModels.reserve(m_gltfFilePaths.size());
  for (std::size_t i = 0; i < m_gltfFilePaths.size(); ++i)
  {
    tinygltf::Model model;
    bool success = loadGltfFile(m_gltfFilePaths.at(i), model);
    if (success)
    {
      m_objectModels.emplace_back(model, m_pbrProgramme, m_boundMaterial);
    }
    else
    {
      m_gltfFilePaths.erase(m_gltfFilePaths.begin() + static_cast<long>(i));
      --i;
    }
  }
  assert(!m_objectModels.empty());
  std::size_t currentObjectModelIdx = 0;

  // Build projection matrix
  glm::vec3 bboxMin, bboxMax;
  // todo: compute scene bounds from max of all model bounds.??
  computeSceneBounds(m_objectModels.at(currentObjectModelIdx).getModel(), bboxMin, bboxMax);

  const glm::vec3 diag = bboxMax - bboxMin;

  float maxDistance = glm::length(diag);
  maxDistance = maxDistance > 0.f ? maxDistance : 100.f;
  const auto projMatrix =
      glm::perspective(70.f,
          static_cast<float>(m_nWindowWidth) / static_cast<float>(m_nWindowHeight),
          0.001f * maxDistance,
          20.f * maxDistance);

  auto cameraController = static_cast<std::unique_ptr<CameraController>>(
      std::make_unique<TrackballCameraController>(
      m_GLFWHandle.window()));
  bool useTrackball = true;
  if (m_hasUserCamera) {
    cameraController->setCamera(m_userCamera);
  } else {
    const bool isSceneFlatOnZaxis = diag.z < 0.01;

    const glm::vec3 centre = bboxMin + diag * 0.5f;
    const glm::vec3 up = glm::vec3(0, 1, 0);
    const glm::vec3 eye = centre
          + (isSceneFlatOnZaxis
              ? diag
              : glm::cross(diag, up) * 0.75f);

    cameraController->setCamera(Camera(eye, centre, up));
    cameraController->setSpeed(maxDistance * 0.2f);
  }

  // Light object
  DirectionalLight directionalLight{
      glm::vec3(1.f),
      0.3f,
      glm::vec3(1.f, 0.f, 0.f)};
  AmbientLight ambientLight { glm::vec3(0.678f, 0.823f, 0.892f), 0.8f };

  bool useCameraLight = false;
  bool useOcclusion = true;

  //
  // Setup OpenGL state for rendering
  glEnable(GL_DEPTH_TEST);
  m_pbrProgramme->use();

  // Lambda function to draw the scene
  const auto drawScene = [&](const Camera &camera) {

    const auto viewMatrix = camera.getViewMatrix();

    // update directional light (SSBO)
    {
      const glm::vec4 viewDir = useCameraLight
          ? glm::vec4(glm::cross(glm::vec3(1.f, 0.f, 0.f), cameraController->getWorldUpAxis()), 0.f)
            : glm::normalize(viewMatrix * glm::vec4(directionalLight.getDirection(), 0.f));
      const DirectionalLightData data = { viewDir, glm::vec4(directionalLight.getRadiance(), 1.f) };


      glBindBuffer(GL_SHADER_STORAGE_BUFFER, directionalLightBufferObject);
      GLvoid *bufferPtr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
      assert(bufferPtr);
      // copy data
      std::memcpy(bufferPtr, &data, sizeof(DirectionalLightData));
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    // update ambient light (UBO)
    {
      const AmbientLightData data = { glm::vec4(ambientLight.getRadiance(), 1.f) };

      glBindBuffer(GL_UNIFORM_BUFFER, ambientLightBufferObject);
      GLvoid *bufferPtr = glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
      assert(bufferPtr);
      std::memcpy(bufferPtr, &data, sizeof(AmbientLightData));
      glUnmapBuffer(GL_UNIFORM_BUFFER);
      glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }

    // Flag objects
    glm::mat4 const rootModelMatrix = glm::identity<glm::mat4>();
      m_objectModels.at(currentObjectModelIdx)
          .draw(materialBufferObject, rootModelMatrix, viewMatrix, projMatrix,
              useOcclusion);
  };
  if (!m_OutputPath.empty())
  {
    const auto width = static_cast<std::size_t>(m_nWindowWidth);
    const auto height = static_cast<std::size_t>(m_nWindowHeight);
    constexpr std::size_t numComponents = 3; // RGB
    std::vector<unsigned char> pixels(
        static_cast<std::size_t>(width * height) * numComponents);

    renderToImage(
        width,
        height,
        numComponents,
        pixels.data(),
        [&cameraController, &drawScene]() { drawScene(cameraController->getCamera()); }
        );

    // need to flip y-axis.
    flipImageYAxis(width, height, numComponents, pixels.data());

    stbi_write_png(m_OutputPath.c_str(), m_nWindowWidth, m_nWindowHeight, numComponents, pixels.data(), 0);

    return 0;
  }

  // set clear colour
  {
    auto clearColour = ambientLight.getRadiance();
    glClearColor(clearColour.r, clearColour.g, clearColour.b, 1.f);
  }

  // Loop until the user closes the window
  for (auto iterationCount = 0u; !m_GLFWHandle.shouldClose();
       ++iterationCount) {
    const double time = glfwGetTime();

    glViewport(0, 0, m_nWindowWidth, m_nWindowHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const auto camera = cameraController->getCamera();
    drawScene(camera);

    // GUI code:
    imguiNewFrame();

    {
      ImGui::Begin("GUI");
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
          1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      if (ImGui::CollapsingHeader("Camera")) {
        ImGui::Text("eye: %.3f %.3f %.3f", camera.eye().x, camera.eye().y,
            camera.eye().z);
        ImGui::Text("center: %.3f %.3f %.3f", camera.center().x,
            camera.center().y, camera.center().z);
        ImGui::Text(
            "up: %.3f %.3f %.3f", camera.up().x, camera.up().y, camera.up().z);

        ImGui::Text("front: %.3f %.3f %.3f", camera.front().x, camera.front().y,
            camera.front().z);
        ImGui::Text("left: %.3f %.3f %.3f", camera.left().x, camera.left().y,
            camera.left().z);

        if (ImGui::Button("CLI camera args to clipboard")) {
          std::stringstream ss;
          ss << "--lookat " << camera.eye().x << "," << camera.eye().y << ","
             << camera.eye().z << "," << camera.center().x << ","
             << camera.center().y << "," << camera.center().z << ","
             << camera.up().x << "," << camera.up().y << "," << camera.up().z;
          const auto str = ss.str();
          glfwSetClipboardString(m_GLFWHandle.window(), str.c_str());
        }
        if (ImGui::RadioButton("Use Trackball", useTrackball)) {
          Camera copy = cameraController->getCamera();
          if (useTrackball) {
            cameraController = std::make_unique<FirstPersonCameraController>(
                m_GLFWHandle.window(), cameraController->getSpeed(),
                cameraController->getWorldUpAxis());
          } else {
            cameraController = std::make_unique<TrackballCameraController>(
                m_GLFWHandle.window(), cameraController->getSpeed(),
                cameraController->getWorldUpAxis());
          }
          cameraController->setCamera(copy);
          useTrackball = !useTrackball;
        }
      }
      if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::RadioButton("Enable occlusions", useOcclusion)) {
          useOcclusion = !useOcclusion;
        }
        if (ImGui::TreeNode("Ambient")) {
          glm::vec3 colourGlm = ambientLight.getColour();
          ImVec4 colour = ImVec4(colourGlm.r, colourGlm.g, colourGlm.b, 0.f);
          float intensity = ambientLight.getIntensity();

          if (ImGui::ColorEdit3("Colour", &colour.x)) {
            colourGlm = glm::vec3(colour.x, colour.y, colour.z);
            ambientLight.setColour(colourGlm);
            auto clearColour = ambientLight.getRadiance();
            glClearColor(clearColour.r, clearColour.g, clearColour.b, 1.f);
          }
          if (ImGui::SliderFloat("Intensity", &intensity, 0.f, 1.f)) {
            ambientLight.setIntensity(intensity);
            auto clearColour = ambientLight.getRadiance();
            glClearColor(clearColour.r, clearColour.g, clearColour.b, 1.f);
          }
          ImGui::TreePop();
        }

        if (ImGui::TreeNode("Directional")) {
          glm::vec2 euler = directionalLight.getEulerAngles();
          glm::vec3 colourGlm = directionalLight.getColour();
          ImVec4 colour = ImVec4(colourGlm.r, colourGlm.g, colourGlm.b, 0.f);
          float intensity = directionalLight.getIntensity();

          if (ImGui::ColorEdit3("Colour", &colour.x)) {
            colourGlm = glm::vec3(colour.x, colour.y, colour.z);
            directionalLight.setColour(colourGlm);
          }
          if (ImGui::SliderFloat("Intensity", &intensity, 0.f, 1.f)) {
            directionalLight.setIntensity(intensity);
          }

          if (ImGui::RadioButton("From Camera", useCameraLight)) {
            useCameraLight = !useCameraLight;
          }
          bool hasLightDirChanged = false;
          hasLightDirChanged |= ImGui::SliderAngle("Theta", &euler.x ,-180.f, 180.f);
          hasLightDirChanged |= ImGui::SliderAngle("Phi",   &euler.y ,-180.f, 180.f);

          if (hasLightDirChanged) {
            directionalLight.setDirection(euler.x, euler.y);
          }
          ImGui::TreePop();
        }
      }
      if (ImGui::CollapsingHeader("Scene")) {

        ImGui::Text("Current model path: %s", m_gltfFilePaths.at(currentObjectModelIdx).c_str());

        std::string inputPath;

        // todo: InputText does not work.
        if (ImGui::InputText("Load model path: ", &inputPath, ImGuiInputTextFlags_EnterReturnsTrue))
        {
          tinygltf::Model model;
          bool success = loadGltfFile(inputPath, model);
          if (success)
          {
            m_objectModels.emplace_back(model, m_pbrProgramme, m_boundMaterial);
            m_gltfFilePaths.emplace_back(inputPath);
          }
          inputPath.clear();
        }

        int modelIdx = static_cast<int>(currentObjectModelIdx);
        if (ImGui::SliderInt("Model", &modelIdx, 0, static_cast<int>(m_objectModels.size()) - 1)) {
          currentObjectModelIdx = static_cast<std::size_t>(modelIdx);
        }
      }
      ImGui::End();
    }

    imguiRenderFrame();

    glfwPollEvents(); // Poll for and process events
    const auto elapsedTime = glfwGetTime() - time;

    auto guiHasFocus =
        ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard;
    if (!guiHasFocus) {
      cameraController->update(float(elapsedTime));
    }


    m_GLFWHandle.swapBuffers(); // Swap front and back buffers
  }

  // TODO clean up allocated GL data

  return 0;
}

ViewerApplication::ViewerApplication(
    const fs::path &appPath, uint32_t width, uint32_t height,
    const std::vector<fs::path> &gltfFiles,
    const std::vector<float> &lookatArgs, const std::string &vertexShader,
    const std::string &fragmentShader, const fs::path &output) :
    m_nWindowWidth(static_cast<GLsizei>(width)),
    m_nWindowHeight(static_cast<GLsizei>(height)),
    m_AppPath{appPath},
    m_AppName{m_AppPath.stem().string()},
    m_ShadersRootPath{m_AppPath.parent_path() / "shaders"},
    m_gltfFilePaths{gltfFiles},
    m_OutputPath{output},
    m_ImGuiIniFilename{m_AppName + ".imgui.ini"},
    m_GLFWHandle {int(m_nWindowWidth), int(m_nWindowHeight),
      "glTF Viewer",
      m_OutputPath.empty()} // show the window only if m_OutputPath is empty
{
  if (!lookatArgs.empty()) {
    m_hasUserCamera = true;
    m_userCamera =
        Camera{glm::vec3(lookatArgs[0], lookatArgs[1], lookatArgs[2]),
            glm::vec3(lookatArgs[3], lookatArgs[4], lookatArgs[5]),
            glm::vec3(lookatArgs[6], lookatArgs[7], lookatArgs[8])};
  }

  if (!vertexShader.empty()) {
    m_vertexShader = vertexShader;
  }

  if (!fragmentShader.empty()) {
    m_fragmentShader = fragmentShader;
  }

  ImGui::GetIO().IniFilename =
      m_ImGuiIniFilename.c_str(); // At exit, ImGUI will store its windows
                                  // positions in this file

  glfwSetKeyCallback(m_GLFWHandle.window(), keyCallback);

  printGLVersion();
}
ObjectModel::ObjectModel(tinygltf::Model model,
                         std::shared_ptr<GLProgram> programme,
                         std::shared_ptr<tinygltf::Material> boundMaterial)
{
  m_model = std::move(model);

  m_textureObjects = createTextureObjects(m_model);
  m_bufferObjects = createBufferObjects(m_model);
  m_vertexArrayObjects = createVertexArrayObjects(
      m_model, m_bufferObjects, m_meshIndexToVaoRange);

  m_programme = std::move(programme);
  m_currentlyBoundMaterial = std::move(boundMaterial);

  m_baseTextureLocation =
      glGetUniformLocation(m_programme->glId(), BASE_TEX_UNIFORM_NAME);
  m_metallicRoughnessTextureLocation =
      glGetUniformLocation(m_programme->glId(), MR_TEX_UNIFORM_NAME);
  m_emissionTextureLocation =
      glGetUniformLocation(m_programme->glId(), EM_TEX_UNIFORM_NAME);
  m_occlusionTextureLocation =
      glGetUniformLocation(m_programme->glId(), OC_TEX_UNIFORM_NAME);

  m_modelViewMatrixLocation =
      glGetUniformLocation(m_programme->glId(), MV_MATRIX_UNIFORM_NAME);
  m_modelViewProjMatrixLocation =
      glGetUniformLocation(m_programme->glId(), MVP_MATRIX_UNIFORM_NAME);
  m_normalMatrixLocation =
      glGetUniformLocation(m_programme->glId(), N_MATRIX_UNIFORM_NAME);

  assert(m_modelViewMatrixLocation          != -1);
  assert(m_modelViewProjMatrixLocation      != -1);

}

ObjectModel::~ObjectModel()
{
  glDeleteTextures(static_cast<GLsizei>(m_textureObjects.size()), m_textureObjects.data());
  glDeleteBuffers(static_cast<GLsizei>(m_bufferObjects.size()), m_bufferObjects.data());
  glDeleteVertexArrays(static_cast<GLsizei>(m_vertexArrayObjects.size()), m_vertexArrayObjects.data());
}

void ObjectModel::bindMaterial(GLuint materialBufferObject, int materialIdx, bool useOcclusion) const
{
  MaterialData data;

  // all of this should be defined in the shaders
  // (and used, otherwise they get compiled-out.)

  if (materialIdx >= 0) {
    // Material binding
    const tinygltf::Material material = m_model.materials.at(static_cast<std::size_t>(materialIdx));
    if (*m_currentlyBoundMaterial == material)
    {
      // nothing to do!
      return;
    }
    const tinygltf::PbrMetallicRoughness & roughness = material.pbrMetallicRoughness;

    data.baseColourFactor = glm::make_vec4(roughness.baseColorFactor.data());
    data.metallicFactor = roughness.metallicFactor;
    data.roughnessFactor = roughness.roughnessFactor;
    data.emissiveFactor = glm::make_vec4(material.emissiveFactor.data());
    data.occlusionStrength = useOcclusion ? material.occlusionTexture.strength : 0.f;

    const int textureIdx = roughness.baseColorTexture.index;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureIdx >= 0
                                 ? m_textureObjects.at(static_cast<std::size_t>(textureIdx))
                                 : getDefaultTextureObject());
    // update
    glUniform1i(m_baseTextureLocation, 0);

    const int metallicRoughnessTextureIdx = roughness.metallicRoughnessTexture.index;
    if (metallicRoughnessTextureIdx >= 0) {
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D,
                    m_textureObjects.at(static_cast<std::size_t>(metallicRoughnessTextureIdx)));
      // update
      glUniform1i(m_metallicRoughnessTextureLocation, 1);
    }

    const int emissionTextureIdx = material.emissiveTexture.index;
    if (emissionTextureIdx >= 0) {
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D,
                    m_textureObjects.at(static_cast<std::size_t>(emissionTextureIdx)));
      glUniform1i(m_emissionTextureLocation, 2);
    }
    const int occlusionTextureIdx = material.occlusionTexture.index;
    if (occlusionTextureIdx >= 0) {
      glActiveTexture(GL_TEXTURE3);
      glBindTexture(GL_TEXTURE_2D,
                    m_textureObjects.at(static_cast<std::size_t>(occlusionTextureIdx)));
      glUniform1i(m_occlusionTextureLocation, 3);
    }

    // update the currently bound material
    *m_currentlyBoundMaterial = material;
  }
  else
  {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, getDefaultTextureObject());
    glUniform1i(m_baseTextureLocation, 0);
  }
  // update Material UBO data
  glBindBuffer(GL_UNIFORM_BUFFER, materialBufferObject);
  GLvoid *bufferPtr = glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
  assert(bufferPtr);
  std::memcpy(bufferPtr, &data, sizeof(MaterialData));
  glUnmapBuffer(GL_UNIFORM_BUFFER);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

void ObjectModel::draw(GLuint materialBufferObject,
    const glm::mat4 & modelMatrix,
    const glm::mat4 & viewMatrix,
    const glm::mat4 & projMatrix,
    bool useOcclusion) const
{

  assert(m_normalMatrixLocation             != -1);

  assert(m_baseTextureLocation              != -1);
  assert(m_metallicRoughnessTextureLocation != -1);
  assert(m_emissionTextureLocation          != -1);
  assert(m_occlusionTextureLocation          != -1);

  // The recursive function that should draw a node
  // We use a std::function because a simple lambda cannot be recursive
  const std::function<void(int, const glm::mat4 &)> drawNode =
      [&](int nodeIdx, const glm::mat4 &parentMatrix) {
        const tinygltf::Node & node = m_model.nodes.at(static_cast<std::size_t>(nodeIdx));
        const glm::mat4 childModelMatrix = getLocalToWorldMatrix(node, parentMatrix);

        // if the node references a mesh
        if (node.mesh >= 0) {
          const glm::mat4 modelViewMatrix = viewMatrix * childModelMatrix;
          const glm::mat4 modelViewProjectionMatrix = projMatrix * modelViewMatrix;
          const glm::mat4 normalMatrix = glm::transpose(glm::inverse(modelViewMatrix));

          glUniformMatrix4fv(
              m_modelViewMatrixLocation,
              1,
              GL_FALSE,
              glm::value_ptr(modelViewMatrix)
                            );
          glUniformMatrix4fv(
              m_modelViewProjMatrixLocation,
              1,
              GL_FALSE,
              glm::value_ptr(modelViewProjectionMatrix)
                            );
          glUniformMatrix4fv(
              m_normalMatrixLocation,
              1,
              GL_FALSE,
              glm::value_ptr(normalMatrix)
                            );

          const auto meshIdx = static_cast<std::size_t>(node.mesh);
          const tinygltf::Mesh mesh = m_model.meshes.at(meshIdx);

          std::size_t primitiveIdx = 0;
          for (const auto & primitive : mesh.primitives) {
            const VaoRange vaoRange = m_meshIndexToVaoRange.at(meshIdx);
            const GLuint vao = m_vertexArrayObjects.at(
                static_cast<std::size_t>(vaoRange.begin) + primitiveIdx);

            bindMaterial(
                materialBufferObject, primitive.material, useOcclusion);

            glBindVertexArray(vao);

            // if the primitive uses indices.
            if (primitive.indices >= 0) {
              const tinygltf::Accessor & accessor = findAccessor(m_model, static_cast<int>(primitive.indices));
              const tinygltf::BufferView & bufferView = findBufferView(m_model, accessor);
              const std::size_t byteOffset = getByteOffset(accessor, bufferView);
              const auto type = static_cast<GLenum>(accessor.componentType);
              glDrawElements(
                  static_cast<GLenum>(primitive.mode),
                  static_cast<GLsizei>(accessor.count),
                  type,
                  reinterpret_cast<GLvoid *>(byteOffset)
                            );
            } else {
              const auto accessorIdx = static_cast<std::size_t>(std::begin(
                  primitive.attributes)->second);
              const tinygltf::Accessor & accessor = m_model.accessors.at(accessorIdx);
              glDrawArrays(
                  static_cast<GLenum>(primitive.mode),
                  0,
                  static_cast<GLsizei>(accessor.count)
                          );
            }
            for (const int childIdx : node.children) {
              drawNode(childIdx, childModelMatrix);
            }
            ++primitiveIdx;
          }
        }
      };

  // Draw the scene referenced by gltf file
  if (m_model.defaultScene >= 0) {
    const auto sceneIdx = static_cast<std::size_t>(m_model.defaultScene);
    const auto & nodes = m_model.scenes[sceneIdx].nodes;
    std::for_each(nodes.cbegin(), nodes.cend(),
                  [&](int nodeIdx) -> void {
                    drawNode(nodeIdx, modelMatrix);
                  });
  }
}

std::vector<GLuint> ObjectModel::createBufferObjects(
    const tinygltf::Model &model) {
  std::vector<GLuint> bufferObjects(model.buffers.size());
  glGenBuffers(static_cast<GLsizei>(bufferObjects.size()), bufferObjects.data());
  for (std::size_t i = 0; i < bufferObjects.size(); ++i) {
    glBindBuffer(GL_ARRAY_BUFFER, bufferObjects.at(i));
    glBufferStorage(
        GL_ARRAY_BUFFER,
        model.buffers.at(i).data.size(),
        model.buffers.at(i).data.data(),
        GL_DYNAMIC_STORAGE_BIT
                   );
  }
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  return bufferObjects;
}

std::vector<GLuint> ObjectModel::createVertexArrayObjects(
    const tinygltf::Model & model,
    const std::vector<GLuint> & bufferObjects,
    std::vector<VaoRange> & meshIndexToVaoRange) {

  std::vector<GLuint> vertexArrayObjects;

  // should be empty, but better safe than sorry.
  meshIndexToVaoRange.clear();
  meshIndexToVaoRange.reserve(model.meshes.size());
  for (const auto &mesh : model.meshes) {
    const std::size_t vaoOffset = vertexArrayObjects.size();

    vertexArrayObjects.resize(vaoOffset + mesh.primitives.size());
    //
    glGenVertexArrays(
        static_cast<GLsizei>(mesh.primitives.size()),
        vertexArrayObjects.data() + vaoOffset
                     );

    std::size_t primitiveIdx = 0;
    for (const auto &primitive : mesh.primitives) {
      const GLuint primitiveVao =
          vertexArrayObjects.at(vaoOffset + primitiveIdx);
      glBindVertexArray(primitiveVao);

      static const std::array<std::pair<std::string, int>, 3> attributes = {
          {{"POSITION", VERTEX_ATTRIB_POSITION_IDX},
              {"NORMAL", VERTEX_ATTRIB_NORMAL_IDX},
              {"TEXCOORD_0", VERTEX_ATTRIB_TEXCOORD0_IDX}}};

      for (const auto &attribute : attributes) {
        const auto iterator = primitive.attributes.find(attribute.first);
        // If "POSITION", "NORMAL", etc. has not been found in the map
        if (iterator == std::end(primitive.attributes)) {
          continue;
        }
        // (*iterator).first is the key,
        // (*iterator).second is the value, ie. the index of the accessor for this attribute
        const tinygltf::Accessor & accessor = findAccessor(model, iterator->second);
        const tinygltf::BufferView & bufferView = findBufferView(model, accessor);
        const GLuint bufferObject = findBufferObject(bufferObjects, bufferView);

        // checking that the buffer view indeed targets ABO.
        assert(bufferView.target == GL_ARRAY_BUFFER);

        // Enable the vertex attrib array corresponding to POSITION with glEnableVertexAttribArray (you need to use VERTEX_ATTRIB_POSITION_IDX which has to be defined at the top of the cpp file)
        glEnableVertexAttribArray(attribute.second);
        // Bind the buffer object to GL_ARRAY_BUFFER
        glBindBuffer(GL_ARRAY_BUFFER, bufferObject);
        // Compute the total byte offset using the accessor and the buffer view
        const std::size_t byteOffset = getByteOffset(accessor, bufferView);
        // Call glVertexAttribPointer with the correct arguments.
        // Remember size is obtained with accessor.type,
        // type is obtained with accessor.componentType.
        // The stride is obtained in the bufferView,
        // normalized is always GL_FALSE,
        // and pointer is the byteOffset (don't forget the cast).
        glVertexAttribPointer(
            attribute.second,
            accessor.type,
            static_cast<GLenum>(accessor.componentType),
            GL_FALSE,
            static_cast<GLsizei>(bufferView.byteStride),
            reinterpret_cast<GLvoid *>(byteOffset)
                             );
      }
      if (primitive.indices >= 0) {
        const tinygltf::Accessor & accessor = findAccessor(model, primitive.indices);
        const tinygltf::BufferView & bufferView = findBufferView(model, accessor);
        const GLuint bufferObject = findBufferObject(bufferObjects, bufferView);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObject);
      }
      ++primitiveIdx;

      glBindVertexArray(0);

      meshIndexToVaoRange.push_back(VaoRange{
          static_cast<GLsizei>(vaoOffset),
          static_cast<GLsizei>(mesh.primitives.size())});
    }
  }
  return vertexArrayObjects;
}

GLuint ObjectModel::s_defaultTextureObject = GL_INVALID_INDEX;

GLuint ObjectModel::getDefaultTextureObject()
{
  if (s_defaultTextureObject != GL_INVALID_INDEX)
  {
    return s_defaultTextureObject;
  }
  glGenTextures(1, &s_defaultTextureObject);

  glBindTexture(GL_TEXTURE_2D, s_defaultTextureObject);

  static constexpr const float white[] = { 1.f, 1.f, 1.f, 1.f };
  // single white pixel.
  glTexImage2D(
      GL_TEXTURE_2D,
      0,
      GL_RGBA, // internal format
      1, 1,
      0, // border
      GL_RGBA,
      GL_FLOAT, white
              );

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT);

  glBindTexture(GL_TEXTURE_2D, 0);

  return s_defaultTextureObject;
}

std::vector<GLuint> ObjectModel::createTextureObjects(tinygltf::Model & model)
{
  std::vector<GLuint> textureObjects(model.textures.size());

  glGenTextures(static_cast<GLsizei>(textureObjects.size()), textureObjects.data());

  static tinygltf::Sampler defaultSampler;
  {
    defaultSampler.minFilter = GL_LINEAR;
    defaultSampler.magFilter = GL_LINEAR;
    defaultSampler.wrapS = GL_REPEAT;
    defaultSampler.wrapT = GL_REPEAT;
    defaultSampler.wrapR = GL_REPEAT;
  }

  std::size_t textureIdx = 0;
  for (const auto & texture : model.textures)
  {
    assert(texture.source >= 0);

    const tinygltf::Image & image = model.images.at(
        static_cast<std::size_t>(texture.source));

    tinygltf::Sampler & sampler = texture.sampler >= 0
                                  ? model.samplers.at(static_cast<std::size_t>(texture.sampler))
                                  : defaultSampler;

    // all of these are optional.
    if (sampler.minFilter < 0) {
      sampler.minFilter = defaultSampler.minFilter;
    }
    if (sampler.magFilter < 0) {
      sampler.magFilter = defaultSampler.magFilter;
    }
    if (sampler.wrapS < 0) {
      sampler.wrapS = defaultSampler.wrapS;
    }
    if (sampler.wrapR < 0) {
      sampler.wrapR = defaultSampler.wrapR;
    }
    if (sampler.wrapT < 0) {
      sampler.wrapT = defaultSampler.wrapT;
    }


    // bind and fill.
    const GLuint textureObject = textureObjects.at(textureIdx);
    glBindTexture(GL_TEXTURE_2D, textureObject);
    // fill the texture data on the GPU
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA, // internal format
        image.width, image.height,
        0, // border
        GL_RGBA,
        image.pixel_type, image.image.data()
                );


    // setting texture parameters.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, sampler.minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, sampler.magFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, sampler.wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, sampler.wrapR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, sampler.wrapT);

    // Generate mipmaps if necessary
    if (sampler.minFilter == GL_NEAREST_MIPMAP_NEAREST
        || sampler.minFilter == GL_NEAREST_MIPMAP_LINEAR
        || sampler.minFilter == GL_LINEAR_MIPMAP_NEAREST
        || sampler.minFilter == GL_LINEAR_MIPMAP_LINEAR)
    {
      glGenerateMipmap(GL_TEXTURE_2D);
    }
    // unbind!
    glBindTexture(GL_TEXTURE_2D, 0);

    ++textureIdx;
  }
  return textureObjects;
}

