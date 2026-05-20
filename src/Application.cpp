#include "Application.h"
#include "Camera.h"
#include "DeferredRenderer.h"
#include "Model.h"
#include "VulkanContext.h"
#include "VulkanSwapchain.h"
#include "imgui.h"
#include <iostream>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

Application *g_App = nullptr;

LRESULT CALLBACK Application::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hwnd, uMsg, wParam, lParam))
        return true;

    if (g_App)
    {
        if (uMsg == WM_CLOSE || uMsg == WM_DESTROY)
        {
            g_App->m_running = false;
            PostQuitMessage(0);
            return 0;
        }
        if (uMsg == WM_KEYDOWN)
        {
            if (wParam < 256)
                g_App->m_keys[wParam] = true;
            if (wParam == VK_ESCAPE)
                g_App->m_running = false; // ESC로 종료
        }
        if (uMsg == WM_KEYUP)
        {
            if (wParam < 256)
                g_App->m_keys[wParam] = false;
        }
        if (uMsg == WM_RBUTTONDOWN)
        {
            if (!ImGui::GetIO().WantCaptureMouse)
            {
                g_App->m_rightMouseDown = true;
                SetCapture(hwnd);
                ShowCursor(FALSE);
            }
            return 0;
        }
        if (uMsg == WM_RBUTTONUP)
        {
            if (g_App->m_rightMouseDown)
            {
                g_App->m_rightMouseDown = false;
                ReleaseCapture();
                ShowCursor(TRUE);
            }
            return 0;
        }
        if (uMsg == WM_INPUT)
        {
            UINT dwSize = 0;
            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, NULL, &dwSize, sizeof(RAWINPUTHEADER));
            LPBYTE lpb = new BYTE[dwSize];
            if (GetRawInputData((HRAWINPUT)lParam, RID_INPUT, lpb, &dwSize, sizeof(RAWINPUTHEADER)) == dwSize)
            {
                RAWINPUT *raw = (RAWINPUT *)lpb;
                if (raw->header.dwType == RIM_TYPEMOUSE)
                {
                    // 상대 마우스 이동량 확인
                    float dx = (float)raw->data.mouse.lLastX;
                    float dy = (float)raw->data.mouse.lLastY;
                    if (g_App && g_App->m_camera && g_App->m_rightMouseDown)
                    {
                        // Windows Raw Input은 아래쪽이 양수 Y이고, 카메라 피치는 위쪽이 양수이므로 -dy를 전달한다.
                        g_App->m_camera->ProcessMouseMovement(dx, -dy);
                    }
                }
            }
            delete[] lpb;
        }
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

Application::Application(int width, int height, const char *title)
    : m_width(width), m_height(height), m_running(true)
{
    g_App = this;

    HINSTANCE hInstance = GetModuleHandle(NULL);

    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"SponzaWindowClass";

    RegisterClass(&wc);

    RECT rect = {0, 0, width, height};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    int titleLen = MultiByteToWideChar(CP_UTF8, 0, title, -1, NULL, 0);
    wchar_t *wTitle = new wchar_t[titleLen];
    MultiByteToWideChar(CP_UTF8, 0, title, -1, wTitle, titleLen);

    m_hwnd = CreateWindowEx(
        0, wc.lpszClassName, wTitle,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top,
        NULL, NULL, hInstance, NULL);

    delete[] wTitle;

    // 부드러운 360도 카메라 회전을 위해 마우스 Raw Input 등록
    RAWINPUTDEVICE Rid[1];
    Rid[0].usUsagePage = 0x01; // HID 일반 사용 페이지
    Rid[0].usUsage = 0x02;     // HID 일반 마우스
    Rid[0].dwFlags = 0;
    Rid[0].hwndTarget = m_hwnd;
    RegisterRawInputDevices(Rid, 1, sizeof(Rid[0]));

    ShowWindow(m_hwnd, SW_SHOW);
    UpdateWindow(m_hwnd);

    // Vulkan 컨텍스트, 스왑체인, 카메라, 모델, 렌더러 초기화
    m_context = new VulkanContext(m_hwnd, hInstance);
    m_swapchain = new VulkanSwapchain(m_context, m_width, m_height);

    // 설정 로드
    m_config.Load("config.txt");

    // Sponza 중앙에서 약간 높은 위치에 카메라 배치
    m_camera = new Camera(m_config.cameraPos, Math::Vec3(0.0f, 1.0f, 0.0f), m_config.cameraYaw, m_config.cameraPitch);
    m_camera->MovementSpeed = m_config.cameraSpeed;
    m_camera->MouseSensitivity = m_config.cameraSensitivity;

    m_model = new Model(m_context, "data/sponza.obj");
    m_renderer = new DeferredRenderer(m_context, m_swapchain);
    m_renderer->Init(m_model, m_hwnd, &m_config);

    m_lastFrameTime = std::chrono::high_resolution_clock::now();
}

Application::~Application()
{
    // 리소스 해제 전에 논리 장치의 실행 완료 대기
    if (m_context)
    {
        vkDeviceWaitIdle(m_context->GetDevice());
    }

    delete m_renderer;
    delete m_model;
    delete m_camera;
    delete m_swapchain;
    delete m_context;
}

void Application::Run()
{
    MSG msg = {};
    while (m_running)
    {
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (!m_running)
            break;

        auto currentFrameTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = currentFrameTime - m_lastFrameTime;
        float deltaTime = duration.count();
        m_lastFrameTime = currentFrameTime;

        ProcessInput(deltaTime);
        Update(deltaTime);
        Render();
    }
}

void Application::ProcessInput(float deltaTime)
{
    if (!m_camera)
        return;

    // ImGui가 키보드 입력을 사용하는 동안 카메라 키보드 이동 차단
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard)
        return;

    // WASD 키로 카메라 이동 처리
    if (m_keys['W'])
        m_camera->ProcessKeyboard(0, deltaTime);
    if (m_keys['S'])
        m_camera->ProcessKeyboard(1, deltaTime);
    if (m_keys['A'])
        m_camera->ProcessKeyboard(2, deltaTime);
    if (m_keys['D'])
        m_camera->ProcessKeyboard(3, deltaTime);
}

void Application::Update(float deltaTime)
{
    // 필요한 경우 장면 갱신 로직 추가
}

void Application::Render()
{
    if (m_renderer && m_camera)
    {
        m_renderer->DrawFrame(m_camera);
    }
}
