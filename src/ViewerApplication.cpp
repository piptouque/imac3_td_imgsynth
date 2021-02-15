#include "ViewerApplication.hpp"

#include <iostream>
#include <numeric>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/io.hpp>

#include "utils/cameras.hpp"

#include <stb_image_write.h>
#include <tiny_gltf.h>

// TD

static constexpr GLuint VERTEX_ATTRIB_POSITION_IDX  = 0;
static constexpr GLuint VERTEX_ATTRIB_NORMAL_IDX    = 1;
static constexpr GLuint VERTEX_ATTRIB_TEXCOORD0_IDX = 2;

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
    // todo: is this right?
    //  see: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBufferStorage.xhtml
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
  // todo
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
      const auto findAccessor = [&model](int accessorIdx) -> const tinygltf::Accessor & {
        return model.accessors.at(static_cast<std::size_t>(accessorIdx));
      };
      const auto findBufferView = [&model](const tinygltf::Accessor & accessor) -> const tinygltf::BufferView & {
        // get the correct tinygltf::Accessor from model.accessors
        const auto bufferViewIdx = static_cast<std::size_t>(accessor.bufferView);
        // get the correct tinygltf::BufferView from model.bufferViews.
        return model.bufferViews.at(bufferViewIdx);
      };
      const auto findBufferObject = [&bufferObjects](const tinygltf::BufferView & bufferView)
          -> GLuint {
        // get the index of the buffer used by the bufferView
        const auto bufferIdx = static_cast<std::size_t>(bufferView.buffer);
        return bufferObjects.at(bufferIdx);
      };
      for (const auto &attribute : attributes) {
        // I'm opening a scope because I want to reuse the variable iterator in the code for NORMAL and TEXCOORD_0
        const auto iterator = primitive.attributes.find(attribute.first);
        // If "POSITION" has been found in the map
        if (iterator == std::end(primitive.attributes)) {
          continue;
        }
        // (*iterator).first is the key "POSITION",
        // (*iterator).second is the value, ie. the index of the accessor for this attribute
        const tinygltf::Accessor & accessor = findAccessor(iterator->second);
        const tinygltf::BufferView & bufferView = findBufferView(accessor);
        const GLuint bufferObject = findBufferObject(bufferView);

        // Enable the vertex attrib array corresponding to POSITION with glEnableVertexAttribArray (you need to use VERTEX_ATTRIB_POSITION_IDX which has to be defined at the top of the cpp file)
        glEnableVertexAttribArray(attribute.second);
        // Bind the buffer object to GL_ARRAY_BUFFER
        glBindBuffer(GL_ARRAY_BUFFER, bufferObject);
        // Compute the total byte offset using the accessor and the buffer view
        const auto byteOffset = accessor.byteOffset + bufferView.byteLength;
        // Call glVertexAttribPointer with the correct arguments.
        glVertexAttribPointer(attribute.second, accessor.type,
              static_cast<GLenum>(accessor.componentType),
              GL_FALSE,
              static_cast<GLsizei>(bufferView.byteStride),
              reinterpret_cast<GLvoid *>(byteOffset)
        );
      }
      if (primitive.indices >= 0) {
        const tinygltf::Accessor & accessor = findAccessor(primitive.indices);
        const tinygltf::BufferView & bufferView = findBufferView(accessor);
        const GLuint bufferObject = findBufferObject(bufferView);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObject);
      }
      // Remember size is obtained with accessor.type, type is obtained with accessor.componentType. The stride is obtained in the bufferView, normalized is always GL_FALSE, and pointer is the byteOffset (don't forget the cast).
      ++primitiveIdx;

      glBindVertexArray(0);
      meshIndexToVaoRange.push_back(VaoRange{static_cast<GLsizei>(vaoOffset),
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

  [[maybe_unused]] const auto modelViewProjMatrixLocation =
      glGetUniformLocation(glslProgram.glId(), "uModelViewProjMatrix");
  [[maybe_unused]] const auto modelViewMatrixLocation =
      glGetUniformLocation(glslProgram.glId(), "uModelViewMatrix");
  [[maybe_unused]] const auto normalMatrixLocation =
      glGetUniformLocation(glslProgram.glId(), "uNormalMatrix");

  // Build projection matrix
  auto maxDistance = 500.f; // TODO use scene bounds instead to compute this
  maxDistance = maxDistance > 0.f ? maxDistance : 100.f;
  [[maybe_unused]] const auto projMatrix =
      glm::perspective(70.f,
          static_cast<float>(m_nWindowWidth) / static_cast<float>(m_nWindowHeight),
          0.001f * maxDistance, 1.5f * maxDistance);

  // TODO Implement a new CameraController model and use it instead. Propose the
  // choice from the GUI
  FirstPersonCameraController cameraController{
      m_GLFWHandle.window(), 0.5f * maxDistance};
  if (m_hasUserCamera) {
    cameraController.setCamera(m_userCamera);
  } else {
    // TODO Use scene bounds to compute a better default camera
    cameraController.setCamera(
        Camera{glm::vec3(0, 0, 0), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0)});
  }

  tinygltf::Model model;
  if (!loadGltfFile(model)) {
    // no gl object has been allocated, can return safely (?)
    return 1;
  }

  std::vector<GLuint> bufferObjects = createBufferObjects(model);

  // TODO Creation of Vertex Array Objects

  // Setup OpenGL state for rendering
  glEnable(GL_DEPTH_TEST);
  glslProgram.use();

  // Lambda function to draw the scene
  const auto drawScene = [&](const Camera &camera) {
    glViewport(0, 0, m_nWindowWidth, m_nWindowHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    [[maybe_unused]] const auto viewMatrix = camera.getViewMatrix();

    // The recursive function that should draw a node
    // We use a std::function because a simple lambda cannot be recursive
    const std::function<void(int, const glm::mat4 &)> drawNode =
        [&]([[maybe_unused]] int nodeIdx, [[maybe_unused]] const glm::mat4 &parentMatrix) {
          // TODO The drawNode function
        };

    // Draw the scene referenced by gltf file
    if (model.defaultScene >= 0) {
      // TODO Draw all nodes
    }
  };

  // Loop until the user closes the window
  for (auto iterationCount = 0u; !m_GLFWHandle.shouldClose();
       ++iterationCount) {
    const auto seconds = glfwGetTime();

    const auto camera = cameraController.getCamera();
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
      }
      ImGui::End();
    }

    imguiRenderFrame();

    glfwPollEvents(); // Poll for and process events

    auto ellapsedTime = glfwGetTime() - seconds;
    auto guiHasFocus =
        ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard;
    if (!guiHasFocus) {
      cameraController.update(float(ellapsedTime));
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
    m_ImGuiIniFilename{m_AppName + ".imgui.ini"}
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
