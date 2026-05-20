#include "Camera.h"
#include <cmath>

Camera::Camera(Math::Vec3 position, Math::Vec3 up, float yaw, float pitch)
    : Front(Math::Vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(50.0f), MouseSensitivity(0.1f)
{
    Position = position;
    WorldUp = up;
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors();
}

Math::Mat4 Camera::GetViewMatrix() const
{
    return Math::Mat4::LookAt(Position, Position + Front, Up);
}

Math::Mat4 Camera::GetProjectionMatrix(float aspect, float fovY) const
{
    return Math::Mat4::Perspective(fovY, aspect, 0.1f, 5000.0f);
}

// 방향: 0=앞, 1=뒤, 2=왼쪽, 3=오른쪽
void Camera::ProcessKeyboard(int direction, float deltaTime)
{
    float velocity = MovementSpeed * deltaTime;
    if (direction == 0)
        Position += Front * velocity;
    if (direction == 1)
        Position -= Front * velocity;
    if (direction == 2)
        Position -= Right * velocity;
    if (direction == 3)
        Position += Right * velocity;
}

void Camera::ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch)
{
    xoffset *= MouseSensitivity;
    yoffset *= MouseSensitivity;

    Yaw += xoffset;
    Pitch += yoffset;

    if (constrainPitch)
    {
        if (Pitch > 89.0f)
            Pitch = 89.0f;
        if (Pitch < -89.0f)
            Pitch = -89.0f;
    }

    updateCameraVectors();
}

void Camera::updateCameraVectors()
{
    Math::Vec3 front;
    float yawRad = Yaw * (3.141592f / 180.0f);
    float pitchRad = Pitch * (3.141592f / 180.0f);

    front.x = std::cos(yawRad) * std::cos(pitchRad);
    front.y = std::sin(pitchRad);
    front.z = std::sin(yawRad) * std::cos(pitchRad);
    Front = front.Normalize();

    Right = Math::Vec3::Cross(Front, WorldUp).Normalize();
    Up = Math::Vec3::Cross(Right, Front).Normalize();
}
