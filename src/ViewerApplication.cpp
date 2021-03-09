#include "ViewerApplication.hpp"

#include <iostream>
#include <numeric>
#include <random>
#include <exception>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/io.hpp>

#include "utils/cameras.hpp"
#include "utils/gltf.hpp"
#include "utils/images.hpp"

#include <stb_image_write.h>
#include <tiny_gltf.h>

#include <cant/physics/HookeSpringLink.hpp>
#include <cant/physics/UniformForceField.hpp>
#include <cant/physics/PhysicsSimulation.hpp>

using Simulation = cant::physics::PhysicsSimulation<3, double>;
using Object = Simulation::Object;
using Force = Simulation::Force;
using ForceField = Simulation::ForceField;

using HookSpring = cant::physics::HookeSpringLink<3, double>;
using Gravity = cant::physics::UniformForceField<3, double>;

using RandomGenerator = std::default_random_engine;
using UniformDoubleDist = std::uniform_real_distribution<double>;

// TD

static constexpr GLuint VERTEX_ATTRIB_POSITION_IDX  = 0;
static constexpr GLuint VERTEX_ATTRIB_NORMAL_IDX    = 1;
static constexpr GLuint VERTEX_ATTRIB_TEXCOORD0_IDX = 2;

static constexpr const char* DIRECTIONALLIGHT_STORAGE_BLOCK_NAME = "sDirectionalLight";
static constexpr GLuint DIRECTIONALLIGHT_STORAGE_BINDING = 1;

static constexpr const char* MATERIAL_UNIFORM_BUFFER_NAME = "bMaterial";
static constexpr GLuint MATERIAL_BLOCK_BINDING = 1;

static constexpr const char* MVP_MATRIX_UNIFORM_NAME = "uModelViewProjMatrix";
static constexpr const char* MV_MATRIX_UNIFORM_NAME = "uModelViewMatrix";
static constexpr const char* N_MATRIX_UNIFORM_NAME = "uNormalMatrix";
static constexpr const char* BASE_TEX_UNIFORM_NAME = "uBaseTexture";
static constexpr const char* MR_TEX_UNIFORM_NAME = "uMetallicRoughnessTexture";
static constexpr const char* EM_TEX_UNIFORM_NAME = "uEmissiveTexture";
static constexpr const char* OC_TEX_UNIFORM_NAME = "uOcclusionTexture";

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

  struct MaterialData
  {
    glm::vec4 baseColourFactor;
    glm::vec4 emissiveFactor;
    glm::float64 metallicFactor;
    glm::float64 roughnessFactor;
    glm::float64 occlusionStrength;
  };
}

bool ViewerApplication::loadGltfFile(tinygltf::Model & model) const {
  tinygltf::TinyGLTF loader;
  std::string err, warn;
  bool success;
  const std::string filePath = m_gltfFilePath.string();
  if (m_gltfFilePath.has_extension() && m_gltfFilePath.extension().string() == ".glb")
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

GLuint ViewerApplication::createDefaultTextureObject()
{
  GLuint textureObject;
  glGenTextures(1, &textureObject);

  glBindTexture(GL_TEXTURE_2D, textureObject);

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

  return textureObject;
}

std::vector<GLuint> ViewerApplication::createTextureObjects(tinygltf::Model & model)
{
  std::vector<GLuint> textureObjects(model.textures.size());

  std::size_t textureIdx = 0;
  glGenTextures(static_cast<GLsizei>(textureObjects.size()), textureObjects.data());

  static tinygltf::Sampler defaultSampler;
  {
    defaultSampler.minFilter = GL_LINEAR;
    defaultSampler.magFilter = GL_LINEAR;
    defaultSampler.wrapS = GL_REPEAT;
    defaultSampler.wrapT = GL_REPEAT;
    defaultSampler.wrapR = GL_REPEAT;
  }

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
      glGetUniformLocation(glslProgram.glId(), MVP_MATRIX_UNIFORM_NAME);
  const auto modelViewMatrixLocation =
      glGetUniformLocation(glslProgram.glId(), MV_MATRIX_UNIFORM_NAME);
  const auto normalMatrixLocation =
      glGetUniformLocation(glslProgram.glId(), N_MATRIX_UNIFORM_NAME);
  const auto baseTextureLocation =
      glGetUniformLocation(glslProgram.glId(), BASE_TEX_UNIFORM_NAME);
  const auto metallicRoughnessTextureLocation =
      glGetUniformLocation(glslProgram.glId(), MR_TEX_UNIFORM_NAME);
  const auto emissionTextureLocation =
      glGetUniformLocation(glslProgram.glId(), EM_TEX_UNIFORM_NAME);
  const auto occlusionTextureLocation =
      glGetUniformLocation(glslProgram.glId(), OC_TEX_UNIFORM_NAME);

  // all of this should be defined in the shaders
  // (and used, otherwise they get compiled-out.)
  assert(modelViewProjMatrixLocation      != -1);
  assert(modelViewMatrixLocation          != -1);
  assert(normalMatrixLocation             != -1);
  assert(baseTextureLocation              != -1);
  assert(metallicRoughnessTextureLocation != -1);
  assert(emissionTextureLocation          != -1);
  assert(occlusionTextureLocation          != -1);

  GLuint lightBufferObject;
  // Light SSBO stuff
  {
    glGenBuffers(1, &lightBufferObject);

    const GLuint lightStorageBlockIndex =
        glGetProgramResourceIndex(glslProgram.glId(), GL_SHADER_STORAGE_BLOCK, DIRECTIONALLIGHT_STORAGE_BLOCK_NAME);
    assert(lightStorageBlockIndex != GL_INVALID_INDEX);

    // Alloc and binding
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, lightBufferObject);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(sizeof(DirectionalLightData) * 1),
        nullptr,
        GL_DYNAMIC_DRAW
      );
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Linking to shader.
    glShaderStorageBlockBinding(glslProgram.glId(), lightStorageBlockIndex,
        DIRECTIONALLIGHT_STORAGE_BINDING);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, DIRECTIONALLIGHT_STORAGE_BINDING,
        lightBufferObject);
  }

  // material UBO stuff
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

    GLuint materialBlockIndex = glGetUniformBlockIndex(glslProgram.glId(),
        MATERIAL_UNIFORM_BUFFER_NAME);

    assert(materialBlockIndex != GL_INVALID_INDEX);

    glUniformBlockBinding(glslProgram.glId(), materialBlockIndex, MATERIAL_BLOCK_BINDING);

    glBindBufferBase(GL_UNIFORM_BUFFER, MATERIAL_BLOCK_BINDING, materialBufferObject);
  }

  // first load the model in order to set camera according to scene bounds.
  tinygltf::Model model;
  if (!loadGltfFile(model)) {
    // no gl object has been allocated, can return safely (?)
    return 1;
  }

  // init random
  unsigned int seed;
  try
  {
    std::random_device rd;
    seed = rd();
  }
  catch (std::exception & e)
  {
    // random device is not available.
    seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
  }
  RandomGenerator rng(seed);
  UniformDoubleDist dist;

  // the objects
  double m = 1.0; // mass
  std::size_t numberSpheres = 1;
  std::vector<std::shared_ptr<Object>> spheres;
  spheres.reserve(numberSpheres);
  {
    // auto par = dist.param();
    auto par = UniformDoubleDist ::param_type(
        - static_cast<double>(numberSpheres) / 2.0,
        static_cast<double>(numberSpheres) / 2.0
                                             );
    dist.param(par);
    auto random = std::bind(dist, rng);
    for (std::size_t i = 0; i < numberSpheres; ++i)
    {
      Object::Position pos = { random(), random(), random() };
      spheres.push_back(std::make_shared<Object>(m, pos));
    }
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
          0.001f * maxDistance,
          1.5f * maxDistance * static_cast<float>(numberSpheres));

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
  DirectionalLight light {
      glm::vec3(1.f, 1.f, 1.f),
      1.f,
      glm::vec3(1.f, 0.f, 0.f)};

  bool useCameraLight = true;
  bool useOcclusion = true;


  // Physics

  bool enabledPhysicsUpdate = false;
  double simulationStep = 0.05;

  float gravityTheta = 0.f;
  float gravityPhi = glm::pi<float>();
  const auto computeDirFromEuler = [](float theta, float phi) -> glm::vec3
  {
    const float sinTheta = glm::sin(theta);
    return glm::vec3(sinTheta * glm::cos(phi),
                     glm::cos(theta),
                     sinTheta * glm::sin(phi));
  };

  double gravityStrength = 9.81;

  Simulation simulation;

  // -- objects
  for (auto & obj : spheres)
  {
    simulation.addKinematicObject(obj);
  }

  /*
  auto o1 = std::make_shared<Object>(m);
  auto o2 = std::make_shared<Object>(m);

  simulation.addKinematicObject(o1);
  simulation.addKinematicObject(o2);

  // -- forces

  double k = 20.0;
  double l0 = 0.5;

   // fixme: for some reason I need to cast it to Force outside of addForce.
  auto spring = static_cast<std::shared_ptr<Force>>(std::make_shared<HookSpring>(k, l0, o1, o2));
  simulation.addForce(spring);
   */

  std::shared_ptr<Gravity> gravity;
  {
    const auto gravityDirGlm = computeDirFromEuler(gravityTheta, gravityPhi);
    gravity = std::make_shared<Gravity>(
        Force::Vector({ gravityDirGlm.x, gravityDirGlm.y, gravityDirGlm.z })
        * gravityStrength);
  }
  simulation.addForceField(gravity);

  //
  const GLuint defaultTextureObject = createDefaultTextureObject();
  std::vector<GLuint> textureObjects = createTextureObjects(model);

  std::vector<GLuint> bufferObjects = createBufferObjects(model);

  std::vector<VaoRange> meshIndexToVaoRange;
  std::vector<GLuint> vertexArrayObjects = createVertexArrayObjects(
      model, bufferObjects, meshIndexToVaoRange);

  // Setup OpenGL state for rendering
  glEnable(GL_DEPTH_TEST);
  glslProgram.use();

  const auto bindMaterial = [&](const int materialIndex) {
    MaterialData data;

    if (materialIndex >= 0) {
      // Material binding
      const tinygltf::Material material = model.materials.at(static_cast<std::size_t>(materialIndex));
      const tinygltf::PbrMetallicRoughness & roughness = material.pbrMetallicRoughness;

      data.baseColourFactor = glm::make_vec4(roughness.baseColorFactor.data());
      data.metallicFactor = roughness.metallicFactor;
      data.roughnessFactor = roughness.roughnessFactor;
      data.emissiveFactor = glm::make_vec4(material.emissiveFactor.data());
      data.occlusionStrength = useOcclusion ? material.occlusionTexture.strength : 0.f;

      const int textureIdx = roughness.baseColorTexture.index;
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, textureIdx >= 0
         ? textureObjects.at(static_cast<std::size_t>(textureIdx))
         : defaultTextureObject
      );
      // update
      glUniform1i(baseTextureLocation, 0);

      const int metallicRoughnessTextureIdx = roughness.metallicRoughnessTexture.index;
      if (metallicRoughnessTextureIdx >= 0) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D,
            textureObjects.at(static_cast<std::size_t>(metallicRoughnessTextureIdx)));
        // update
        glUniform1i(metallicRoughnessTextureLocation, 1);
      }

      const int emissionTextureIdx = material.emissiveTexture.index;
      if (emissionTextureIdx >= 0) {
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D,
            textureObjects.at(static_cast<std::size_t>(emissionTextureIdx)));
        glUniform1i(emissionTextureLocation, 2);
      }
      const int occlusionTextureIdx = material.occlusionTexture.index;
      if (occlusionTextureIdx >= 0) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D,
            textureObjects.at(static_cast<std::size_t>(occlusionTextureIdx)));
        glUniform1i(occlusionTextureLocation, 3);
      }
    }
    else
    {
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, defaultTextureObject);
      glUniform1i(baseTextureLocation, 0);
    }


    // update UBO data
    glBindBuffer(GL_UNIFORM_BUFFER, materialBufferObject);
    GLvoid *bufferPtr = glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
    std::memcpy(bufferPtr, &data, sizeof(MaterialData));
    glUnmapBuffer(GL_UNIFORM_BUFFER);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

  };

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
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, lightBufferObject);
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

              bindMaterial(primitive.material);

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
      for (const auto & obj : spheres)
      {
        auto pos = obj->getPosition();
        glm::mat4 const rootModelMatrix = glm::translate(
            glm::identity<glm::mat4>(), glm::vec3(pos.get<0>(), pos.get<1>(), pos.get<2>())
            );
        std::for_each(nodes.cbegin(), nodes.cend(),
                      [&](int nodeIdx) -> void {
                        drawNode(nodeIdx, rootModelMatrix);
                      });
      }

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

  double accH = 0.0;
  double time = glfwGetTime();
  // Loop until the user closes the window
  for (auto iterationCount = 0u; !m_GLFWHandle.shouldClose();
       ++iterationCount) {
    const double currentTime = glfwGetTime();
    const double deltaTime = currentTime - time;
    time = currentTime;

    if (enabledPhysicsUpdate)
    {
      accH += deltaTime;
      if (accH >= simulationStep) {
        // update physics.
        simulation.stepDelta(accH);
        accH = 0.0;
      }
    }

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
      if (ImGui::CollapsingHeader("Light")) {
        glm::vec2 euler = light.getEulerAngles();
        glm::vec3 colourGlm = light.getColour();
        ImVec4 colour = ImVec4(colourGlm.r, colourGlm.g, colourGlm.b, 0.f);
        float intensity = light.getIntensity();

        bool hasLightDirChanged = false;
        hasLightDirChanged |= ImGui::SliderAngle("LightTheta", &euler.x ,-180.f, 180.f);
        hasLightDirChanged |= ImGui::SliderAngle("LightPhi",   &euler.y ,-180.f, 180.f);

        if (hasLightDirChanged) {
          light.setDirection(euler.x, euler.y);
        }

        if (ImGui::RadioButton("From Camera", useCameraLight)) {
          useCameraLight = !useCameraLight;
        }

        if (ImGui::ColorEdit3("Colour", &colour.x)) {
          colourGlm = glm::vec3(colour.x, colour.y, colour.z);
          light.setColour(colourGlm);
        }
        if (ImGui::SliderFloat("Intensity", &intensity, 0.f, 1.f)) {
          light.setIntensity(intensity);
        }

        if (ImGui::RadioButton("Use occlusions", useOcclusion)) {
          useOcclusion = !useOcclusion;
        }
      }
      if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto step = static_cast<float>(simulationStep);
        if (ImGui::SliderFloat("Simulation Step", &step, 0.f, 1.f))
        {
          simulationStep = step;
        }

        glm::vec3 gravityDirGlm;
        auto strength = static_cast<float>(gravityStrength);

        if (ImGui::RadioButton("Enable Physics", enabledPhysicsUpdate)) {
          enabledPhysicsUpdate = !enabledPhysicsUpdate;
        }

        bool hasGravityChanged = false;
        hasGravityChanged |= ImGui::SliderAngle("GravityTheta", &gravityTheta ,-180.f, 180.f);
        hasGravityChanged |= ImGui::SliderAngle("GravityPhi",   &gravityPhi,-180.f, 180.f);

        hasGravityChanged = ImGui::SliderFloat("GravityStrength", &strength, 0.f, 20.f);

        if (hasGravityChanged) {
          gravityStrength = strength;
          const float sinTheta = glm::sin(gravityTheta);
          gravityDirGlm = glm::vec3(sinTheta * glm::cos(gravityPhi),
                                    glm::cos(gravityTheta),
                                    sinTheta * glm::sin(gravityPhi));
          gravity->setVector(Force::Vector({ gravityDirGlm.x, gravityDirGlm.y, gravityDirGlm.z })
                             * gravityStrength);
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