#pragma once
#include "Math.h"

class Camera
{
  public:
    Camera(Math::Vec3 position, Math::Vec3 up, float yaw, float pitch);

    Math::Mat4 GetViewMatrix() const;
    Math::Mat4 GetProjectionMatrix(float aspect, float fovY = 45.0f * (3.141592f / 180.0f)) const;

    void ProcessKeyboard(int direction, float deltaTime);
    void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);

    Math::Vec3 Position;
    Math::Vec3 Front;
    Math::Vec3 Up;
    Math::Vec3 Right;
    Math::Vec3 WorldUp;

    float Yaw;
    float Pitch;
    float MovementSpeed;
    float MouseSensitivity;

  private:
    void updateCameraVectors();
};
