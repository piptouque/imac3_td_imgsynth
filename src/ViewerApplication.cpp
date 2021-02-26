#include "ViewerApplication.hpp"

#include <iostream>
#include <numeric>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/io.hpp>

#include "utils/cameras.hpp"
#include "utils/gltf.hpp"
#include "utils/images.hpp"

#include <stb_image_write.h>
#include <tiny_gltf.h>

// TD

static constexpr GLuint VERTEX_ATTRIB_POSITION_IDX  = 0;
static constexpr GLuint VERTEX_ATTRIB_NORMAL_IDX    = 1;
static constexpr GLuint VERTEX_ATTRIB_TEXCOORD0_IDX = 2;

static constexpr const char* DIRECTIONALLIGHT_STORAGE_BLOCK_NAME = "sDirectionalLight";
static constexpr GLuint DIRECTIONALLIGHT_BINDING_INDEX = 1;

namespace {
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

  std::size_t getByteOffset(const tinygltf::Accessor & accessor, const tinygltf::BufferView bufferView) {
    return accessor.byteOffset + bufferView.byteOffset;
  }

  class DirectionalLight
  {
  public:
    DirectionalLight() = default;
    DirectionalLight(glm::vec3 colour, float intensity, glm::vec3 dir)
      : m_colour(colour), m_intensity(intensity)
    {
      setDirection(dir);
    }

    [[nodiscard]] inline const glm::vec3 & getDirection() const { return m_dir; }
    [[nodiscard]] inline glm::vec3 getRadiance() const { return m_colour * m_intensity; }
    [[nodiscard]] inline const glm::vec3 & getColour() const { return m_colour; }
    [[nodiscard]] inline float getIntensity() const { return m_intensity; }
    [[nodiscard]] inline glm::vec2 getEulerAngles() const { return glm::vec2(m_theta, m_phi); }
    inline void setColour(glm::vec3 colour) { m_colour = glm::normalize(colour); }
    inline void setIntensity(float intensity) { m_intensity = intensity; }

    void setDirection(glm::vec3 a_dir)
    {
      m_dir = a_dir;
      const glm::vec3 euler = computeEulerAngles(m_dir);
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
                        glm::sin(theta) * glm::sin(phi));
    }

    [[nodiscard]] static glm::vec3 computeEulerAngles(glm::vec3 dir)
    {
      const glm::quat quat = glm::rotation(glm::vec3(), dir);
      return glm::eulerAngles(quat);
    }

  private:
    glm::vec3 m_colour;
    float m_intensity;
    //
    glm::vec3 m_dir;
    // degrees
    float m_theta;
    float m_phi;
  };

  // OpenGL data
  struct DirectionalLightData
  {
    glm::vec4 viewDir;
    glm::vec4 radiance;
  };
}

bool ViewerApplication::loadGltfFile(tinygltf::Model & model) const {
  tinygltf::TinyGLTF loader;
  std::string err, warn;
  bool success = loader.LoadASCIIFromFile(
      &model, &err, &warn,
      m_gltfFilePath.string());
  if (!err.empty()) {
    std::cerr << "Error while parsing GLTF file: " << err << std::endl;
  }
  if (!warn.empty()) {
    std::cerr << "Warning while parsing GLTF file: " << warn << std::endl;
  }
  if (!success) {
    std::cerr << "Failed to parse GLTF file at " << m_gltfFilePath << std::endl;
  }
  return success;
}

std::vector<GLuint> ViewerApplication::createBufferObjects(
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

std::vector<GLuint> ViewerApplication::createVertexArrayObjects(
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

void keyCallback(
    GLFWwindow *window, int key, [[maybe_unused]] int scancode,
    int action, [[maybe_unused]] int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
    glfwSetWindowShouldClose(window, 1);
  }
}


int ViewerApplication::run()
{
  // Loader shaders
  const auto glslProgram =
      compileProgram({m_ShadersRootPath / m_vertexShader,
          m_ShadersRootPath / m_fragmentShader});

  const auto modelViewProjMatrixLocation =
      glGetUniformLocation(glslProgram.glId(), "uModelViewProjMatrix");
  const auto modelViewMatrixLocation =
      glGetUniformLocation(glslProgram.glId(), "uModelViewMatrix");
  const auto normalMatrixLocation =
      glGetUniformLocation(glslProgram.glId(), "uNormalMatrix");


  GLuint lightBuffer;
  // Light SSBO stuff
  {
    glGenBuffers(1, &lightBuffer);

    const GLuint lightStorageBlockIndex =
        glGetProgramResourceIndex(glslProgram.glId(), GL_SHADER_STORAGE_BLOCK, "sDirectionalLight");
    assert(lightStorageBlockIndex != GL_INVALID_INDEX);

    // Alloc and binding
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, lightBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(sizeof(DirectionalLightData) * 1),
        nullptr,
        GL_DYNAMIC_DRAW
      );
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Linking to shader.
    glShaderStorageBlockBinding(glslProgram.glId(), lightStorageBlockIndex,
        DIRECTIONALLIGHT_BINDING_INDEX);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, DIRECTIONALLIGHT_BINDING_INDEX, lightBuffer);
  }

  // first load the model in order to set camera according to scene bounds.
  tinygltf::Model model;
  if (!loadGltfFile(model)) {
    // no gl object has been allocated, can return safely (?)
    return 1;
  }

  // Build projection matrix
  glm::vec3 bboxMin, bboxMax;
  computeSceneBounds(model, bboxMin, bboxMax);

  const glm::vec3 diag = bboxMax - bboxMin;

  float maxDistance = glm::length(diag);
  maxDistance = maxDistance > 0.f ? maxDistance : 100.f;
  const auto projMatrix =
      glm::perspective(70.f,
          static_cast<float>(m_nWindowWidth) / static_cast<float>(m_nWindowHeight),
          0.001f * maxDistance, 1.5f * maxDistance);

  auto cameraController = static_cast<std::unique_ptr<CameraController>>(
      std::make_unique<TrackballCameraController>(
      m_GLFWHandle.window(), 0.5f * maxDistance));
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
              : glm::cross(diag, up) * 2.f);

    cameraController->setCamera(Camera(eye, centre, up)); cameraController->setSpeed(maxDistance * 3.f);
  }

  // Light object
  DirectionalLight light { glm::vec3(1.f, 0.f, 0.f), 1.f, glm::vec3(1.f, 0.f, 0.f)};
  bool useCameraLight = false;

  std::vector<GLuint> bufferObjects = createBufferObjects(model);

  std::vector<VaoRange> meshIndexToVaoRange;
  std::vector<GLuint> vertexArrayObjects = createVertexArrayObjects(
      model, bufferObjects, meshIndexToVaoRange);

  // Setup OpenGL state for rendering
  glEnable(GL_DEPTH_TEST);
  glslProgram.use();

  // Lambda function to draw the scene
  const auto drawScene = [&](const Camera &camera) {
    glViewport(0, 0, m_nWindowWidth, m_nWindowHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const auto viewMatrix = camera.getViewMatrix();

    // update light
    {
      const glm::vec4 viewDir = useCameraLight
          ? glm::vec4(glm::cross(glm::vec3(1.f, 0.f, 0.f), cameraController->getWorldUpAxis()), 0.f)
            : glm::normalize(viewMatrix * glm::vec4(light.getDirection(), 0.f));
      const DirectionalLightData data = { viewDir, glm::vec4(light.getRadiance(), 1.f) };
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, lightBuffer);
      GLvoid *bufferPtr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
      assert(bufferPtr);
      // copy data
      std::memcpy(bufferPtr, &data, sizeof(DirectionalLightData) * 1);
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    // The recursive function that should draw a node
    // We use a std::function because a simple lambda cannot be recursive
    const std::function<void(int, const glm::mat4 &)> drawNode =
        [&](int nodeIdx, const glm::mat4 &parentMatrix) {
          const tinygltf::Node & node = model.nodes.at(static_cast<std::size_t>(nodeIdx));
          const glm::mat4 modelMatrix = getLocalToWorldMatrix(node, parentMatrix);

          // if the node references a mesh
          if (node.mesh >= 0) {
            const glm::mat4 modelViewMatrix = viewMatrix * modelMatrix;
            const glm::mat4 modelViewProjectionMatrix = projMatrix * modelViewMatrix;
            const glm::mat4 normalMatrix = glm::transpose(glm::inverse(modelViewMatrix));

            glUniformMatrix4fv(
              modelViewMatrixLocation,
                1,
                GL_FALSE,
                glm::value_ptr(modelViewMatrix)
            );
            glUniformMatrix4fv(
                modelViewProjMatrixLocation,
                1,
                GL_FALSE,
                glm::value_ptr(modelViewProjectionMatrix)
            );
            glUniformMatrix4fv(
                normalMatrixLocation,
                1,
                GL_FALSE,
                glm::value_ptr(normalMatrix)
            );

            const auto meshIdx = static_cast<std::size_t>(node.mesh);
            const tinygltf::Mesh mesh = model.meshes.at(meshIdx);

            std::size_t primitiveIdx = 0;
            for (const auto & primitive : mesh.primitives) {
              const VaoRange vaoRange = meshIndexToVaoRange.at(meshIdx);
              const GLuint vao = vertexArrayObjects.at(
                  static_cast<std::size_t>(vaoRange.begin) + primitiveIdx);

              glBindVertexArray(vao);

              // if the primitive uses indices.
              if (primitive.indices >= 0) {
                const tinygltf::Accessor & accessor = findAccessor(model, static_cast<int>(primitive.indices));
                const tinygltf::BufferView & bufferView = findBufferView(model, accessor);
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
                const tinygltf::Accessor & accessor = model.accessors.at(accessorIdx);
                glDrawArrays(
                    static_cast<GLenum>(primitive.mode),
                    0,
                    static_cast<GLsizei>(accessor.count)
                );
              }
              for (const int childIdx : node.children) {
                drawNode(childIdx, modelMatrix);
              }
              ++primitiveIdx;
            }

          }
        };

    // Draw the scene referenced by gltf file
    if (model.defaultScene >= 0) {
      const auto sceneIdx = static_cast<std::size_t>(model.defaultScene);
      const auto & nodes = model.scenes[sceneIdx].nodes;
      std::for_each(nodes.cbegin(), nodes.cend(),
        [&](int nodeIdx) -> void {
          drawNode(nodeIdx, glm::identity<glm::mat4>());
      });

    }
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

  // Loop until the user closes the window
  for (auto iterationCount = 0u; !m_GLFWHandle.shouldClose();
       ++iterationCount) {
    const auto seconds = glfwGetTime();

    const auto camera = cameraController->getCamera();
    drawScene(camera);

    // GUI code:
    imguiNewFrame();

    {
      ImGui::Begin("GUI");
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
          1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
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
        if (ImGui::RadioButton("Toggle Trackball", useTrackball)) {
          Camera copy = cameraController->getCamera();
          if (useTrackball) {
            cameraController = std::make_unique<FirstPersonCameraController>(
                m_GLFWHandle.window(),
                cameraController->getSpeed(),
                cameraController->getWorldUpAxis()
            );
          }
          else
          {
            cameraController = std::make_unique<TrackballCameraController>(
                m_GLFWHandle.window(),
                cameraController->getSpeed(),
                cameraController->getWorldUpAxis()
            );
          }
          cameraController->setCamera(copy);
          useTrackball = !useTrackball;
        }
        if (ImGui::CollapsingHeader("Light"))
        {
          glm::vec2 euler = light.getEulerAngles();
          glm::vec3 colourGlm = light.getColour();
          ImVec4 colour = ImVec4(colourGlm.r, colourGlm.g, colourGlm.b, 0.f);
          float intensity = light.getIntensity();

          bool hasLightDirChanged = false;
          hasLightDirChanged |= ImGui::SliderAngle("Theta", &euler.x ,-180.f, 180.f);
          hasLightDirChanged |= ImGui::SliderAngle("Phi",   &euler.y ,-180.f, 180.f);

          if (hasLightDirChanged)
          {
            light.setDirection(euler.x, euler.y);
          }

          if (ImGui::ColorEdit3("Colour", &colour.x))
          {
            colourGlm = glm::vec3(colour.x, colour.y, colour.z);
            light.setColour(colourGlm);
          }
          if (ImGui::SliderFloat("Intensity", &intensity, 0.f, 1.f))
          {
            light.setIntensity(intensity);
          }

          if (ImGui::RadioButton("From Camera", useCameraLight))
          {
            useCameraLight = !useCameraLight;
          }

        }
      }
      ImGui::End();
    }

    imguiRenderFrame();

    glfwPollEvents(); // Poll for and process events

    auto ellapsedTime = glfwGetTime() - seconds;
    auto guiHasFocus =
        ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard;
    if (!guiHasFocus) {
      cameraController->update(float(ellapsedTime));
    }

    m_GLFWHandle.swapBuffers(); // Swap front and back buffers
  }

  // TODO clean up allocated GL data

  return 0;
}

ViewerApplication::ViewerApplication(const fs::path &appPath, uint32_t width,
    uint32_t height, const fs::path &gltfFile,
    const std::vector<float> &lookatArgs, const std::string &vertexShader,
    const std::string &fragmentShader, const fs::path &output) :
    m_nWindowWidth(static_cast<GLsizei>(width)),
    m_nWindowHeight(static_cast<GLsizei>(height)),
    m_AppPath{appPath},
    m_AppName{m_AppPath.stem().string()},
    m_ShadersRootPath{m_AppPath.parent_path() / "shaders"},
    m_gltfFilePath{gltfFile},
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
