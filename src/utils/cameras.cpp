#include "cameras.hpp"
#include "glfw.hpp"

#include <iostream>

// Good reference here to map camera movements to lookAt calls
// http://learnwebgl.brown37.net/07_cameras/camera_movement.html

using namespace glm;

struct ViewFrame
{
  vec3 left;
  vec3 up;
  vec3 front;
  vec3 eye;

  ViewFrame(vec3 l, vec3 u, vec3 f, vec3 e) : left(l), up(u), front(f), eye(e)
  {
  }
};

ViewFrame fromViewToWorldMatrix(const mat4 &viewToWorldMatrix)
{
  return ViewFrame{-vec3(viewToWorldMatrix[0]), vec3(viewToWorldMatrix[1]),
      -vec3(viewToWorldMatrix[2]), vec3(viewToWorldMatrix[3])};
}

bool FirstPersonCameraController::update(float elapsedTime)
{
  if (glfwGetMouseButton(m_pWindow, GLFW_MOUSE_BUTTON_MIDDLE) &&
      !m_wasLeftButtonLastPressed) {
    m_wasLeftButtonLastPressed = true;
    glfwGetCursorPos(
        m_pWindow, &m_cursorLastPressedPosition.x, &m_cursorLastPressedPosition.y);
  } else if (!glfwGetMouseButton(m_pWindow, GLFW_MOUSE_BUTTON_MIDDLE) &&
             m_wasLeftButtonLastPressed) {
    m_wasLeftButtonLastPressed = false;
  }

  const auto cursorDelta = ([&]() {
    if (m_wasLeftButtonLastPressed) {
      dvec2 cursorPosition;
      glfwGetCursorPos(m_pWindow, &cursorPosition.x, &cursorPosition.y);
      const auto delta = cursorPosition - m_cursorLastPressedPosition;
      m_cursorLastPressedPosition = cursorPosition;
      return delta;
    }
    return dvec2(0);
  })();

  float truckLeft = 0.f;
  float pedestalUp = 0.f;
  float dollyIn = 0.f;
  float rollRightAngle = 0.f;

  if (glfwGetKey(m_pWindow, GLFW_KEY_W)) {
    dollyIn += m_fSpeed * elapsedTime;
  }

  // Truck left
  if (glfwGetKey(m_pWindow, GLFW_KEY_A)) {
    truckLeft += m_fSpeed * elapsedTime;
  }

  // Pedestal up
  if (glfwGetKey(m_pWindow, GLFW_KEY_UP)) {
    pedestalUp += m_fSpeed * elapsedTime;
  }

  // Dolly out
  if (glfwGetKey(m_pWindow, GLFW_KEY_S)) {
    dollyIn -= m_fSpeed * elapsedTime;
  }

  // Truck right
  if (glfwGetKey(m_pWindow, GLFW_KEY_D)) {
    truckLeft -= m_fSpeed * elapsedTime;
  }

  // Pedestal down
  if (glfwGetKey(m_pWindow, GLFW_KEY_DOWN)) {
    pedestalUp -= m_fSpeed * elapsedTime;
  }

  if (glfwGetKey(m_pWindow, GLFW_KEY_Q)) {
    rollRightAngle -= 0.001f;
  }
  if (glfwGetKey(m_pWindow, GLFW_KEY_E)) {
    rollRightAngle += 0.001f;
  }

  // cursor going right, so minus because we want pan left angle:
  const float panLeftAngle = -0.01f * float(cursorDelta.x);
  const float tiltDownAngle = 0.01f * float(cursorDelta.y);

  const auto hasMoved =
      truckLeft != 0.0f
      || pedestalUp != 0.0f
      || dollyIn != 0.0f
      || panLeftAngle != 0.0f
      || tiltDownAngle != 0.0f
      || rollRightAngle != 0.0f;
  if (!hasMoved) {
    return false;
  }

  m_camera.moveLocal(truckLeft, pedestalUp, dollyIn);
  m_camera.rotateLocal(rollRightAngle, tiltDownAngle, 0.f);
  m_camera.rotateWorld(panLeftAngle, m_worldUpAxis);

  return true;
}

bool TrackballCameraController::update(float elapsedTime) {

  const bool isMiddleButtonBeingPressed = glfwGetMouseButton(m_pWindow, GLFW_MOUSE_BUTTON_MIDDLE);
  const bool isShiftPressed = glfwGetKey(m_pWindow, GLFW_KEY_LEFT_SHIFT);
  const bool isControlPressed = glfwGetKey(m_pWindow, GLFW_KEY_LEFT_CONTROL);

  // if the middle button has just started being pressed.
  const bool hasMiddleButtonChanged = m_wasMiddleButtonLastPressed
                                      xor isMiddleButtonBeingPressed;
  dvec2 cursorPosition;
  glfwGetCursorPos(m_pWindow, &cursorPosition.x, &cursorPosition.y);

  if (hasMiddleButtonChanged && isMiddleButtonBeingPressed) {
    m_cursorLastPressedPosition = cursorPosition;
  }
  m_wasMiddleButtonLastPressed = isMiddleButtonBeingPressed;

  const dvec2 cursorVec = (cursorPosition - m_cursorLastPressedPosition);

  bool hasMoved = false;

  if (isMiddleButtonBeingPressed) {
    if (isShiftPressed) {
      const float truckLeft = 0.2f * elapsedTime * static_cast<float>(cursorVec.x);
      const float pedestalUp = 0.2f * elapsedTime * static_cast<float>(cursorVec.y);


      hasMoved = std::abs(truckLeft) > 0.01 || std::abs(pedestalUp) > 0.01;
      if (hasMoved)
      {
        m_camera.moveLocal(truckLeft, pedestalUp, 0.f);
      }

    }
    else if (isControlPressed) {
      const vec3 viewVec = m_camera.center() - m_camera.eye();
      const float offset = std::min(
          0.2f * elapsedTime * static_cast<float>(cursorVec.x),
          length(viewVec) - 0.1f
          );
      const vec3 zoomVec = normalize(viewVec) * offset;

      hasMoved = std::abs(offset) > 0.01;

      if (hasMoved) {
        const vec3 eye = m_camera.eye() + zoomVec;
        const vec3 centre = m_camera.center();
        const vec3 up = m_camera.up();

        m_camera = Camera(eye, centre, up);
      }
    }
    else {
      // rotate around centre.
      const auto longitudeAngle = static_cast<float>(0.08 * elapsedTime * cursorVec.y);
      const auto latitudeAngle = static_cast<float>(0.08 * elapsedTime * cursorVec.x);

      hasMoved = std::abs(longitudeAngle) > 0.01f || std::abs(latitudeAngle) > 0.01f;

      if (hasMoved) {
        const vec3 depthVec = m_camera.eye() - m_camera.center();

        const mat4 latitudeRotationMatrix = rotate(identity<mat4>(), latitudeAngle, m_worldUpAxis);
        const mat4 longitudeRotationMatrix = rotate(identity<mat4>(), longitudeAngle, vec3(1.f, 0.f, 0.f));

        // first rotate around x-axis, then up axis.
        const mat4 totalRotationMatrix = longitudeRotationMatrix * latitudeRotationMatrix;

        const vec3 rotatedVec = vec3(totalRotationMatrix * vec4(depthVec, 0.f));
        const vec3 rotatedUp = vec3(totalRotationMatrix * vec4(m_camera.up(), 0.f));

        const vec3 eye = m_camera.center() + rotatedVec;
        const vec3 centre = m_camera.center();
        const vec up = rotatedUp;

        m_camera = Camera(eye, centre, up);

      }
    }
  }

  return hasMoved;
}
