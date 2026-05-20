#pragma once
#include <vector>
#include <vulkan/vulkan.h>

class VulkanContext;

class VulkanSwapchain
{
  public:
    VulkanSwapchain(VulkanContext *context, int width, int height);
    ~VulkanSwapchain();

    VkSwapchainKHR GetSwapchain() const
    {
        return m_swapchain;
    }
    VkFormat GetImageFormat() const
    {
        return m_imageFormat;
    }
    VkExtent2D GetExtent() const
    {
        return m_extent;
    }
    const std::vector<VkImage> &GetImages() const
    {
        return m_images;
    }
    const std::vector<VkImageView> &GetImageViews() const
    {
        return m_imageViews;
    }

    void Recreate(int width, int height);

  private:
    void createSwapchain(int width, int height);
    void createImageViews();
    void cleanup();

    VulkanContext *m_context;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_imageViews;
    VkFormat m_imageFormat;
    VkExtent2D m_extent;
};
