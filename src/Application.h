#pragma once
#define WIN32_LEAN_AND_MEAN
#include "RenderConfig.h"
#include <chrono>
#include <windows.h>

class VulkanContext;
class VulkanSwapchain;
class Camera;
class Model;
class DeferredRenderer;

class Application
{
  public:
    Application(int width, int height, const char *title);
    ~Application();

    void Run();

    HWND GetWindowHandle() const
    {
        return m_hwnd;
    }
    int GetWidth() const
    {
        return m_width;
    }
    int GetHeight() const
    {
        return m_height;
    }

  private:
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    void ProcessInput(float deltaTime);
    void Update(float deltaTime);
    void Render();

    HWND m_hwnd;
    int m_width;
    int m_height;
    bool m_running;

    // 시간 추적
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;

    // 입력 상태
    bool m_keys[256] = {false};
    float m_mouseX = 0.0f;
    float m_mouseY = 0.0f;
    bool m_firstMouse = true;

    // Vulkan 및 렌더링 하위 시스템
    VulkanContext *m_context = nullptr;
    VulkanSwapchain *m_swapchain = nullptr;
    Camera *m_camera = nullptr;
    Model *m_model = nullptr;
    DeferredRenderer *m_renderer = nullptr;

    RenderConfig m_config;
    bool m_rightMouseDown = false;
};
