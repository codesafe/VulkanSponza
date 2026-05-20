#pragma once
#define VK_USE_PLATFORM_WIN32_KHR
#include <vector>
#include <vulkan/vulkan.h>

class VulkanContext
{
  public:
    VulkanContext(void *hwnd, void *hinstance);
    ~VulkanContext();

    VkInstance GetInstance() const
    {
        return m_instance;
    }
    VkPhysicalDevice GetPhysicalDevice() const
    {
        return m_physicalDevice;
    }
    VkDevice GetDevice() const
    {
        return m_device;
    }
    VkSurfaceKHR GetSurface() const
    {
        return m_surface;
    }
    VkQueue GetGraphicsQueue() const
    {
        return m_graphicsQueue;
    }
    VkQueue GetPresentQueue() const
    {
        return m_presentQueue;
    }
    uint32_t GetGraphicsQueueFamily() const
    {
        return m_graphicsQueueFamily;
    }
    uint32_t GetPresentQueueFamily() const
    {
        return m_presentQueueFamily;
    }
    VkCommandPool GetCommandPool() const
    {
        return m_commandPool;
    }

  private:
    void createInstance();
    void createSurface(void *hwnd, void *hinstance);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();

    VkInstance m_instance = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_presentQueue = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueFamily = -1;
    uint32_t m_presentQueueFamily = -1;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
};
