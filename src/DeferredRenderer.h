#pragma once
#define WIN32_LEAN_AND_MEAN
#include "Math.h"
#include <vector>
#include <vulkan/vulkan.h>
#include <windows.h>

class VulkanContext;
class VulkanSwapchain;
class Model;
class Camera;

struct UniformBufferObject
{
    Math::Mat4 model;
    Math::Mat4 view;
    Math::Mat4 proj;
};

struct LightBufferObject
{
    Math::Vec4 lightDirAndAmbient;     // xyz = 조명 방향, w = 주변광 강도
    Math::Vec4 viewPosAndSpecular;     // xyz = 시점 위치, w = 정반사 강도
    Math::Vec4 lightColorAndSpecPower; // xyz = 조명 색상, w = 정반사 지수
    Math::Mat4 lightSpaceMatrix;
};

struct RenderConfig;

class DeferredRenderer
{
  public:
    DeferredRenderer(VulkanContext *context, VulkanSwapchain *swapchain);
    ~DeferredRenderer();

    void Init(Model *model, HWND hwnd, RenderConfig *config);
    void DrawFrame(Camera *camera);
    void Recreate();

  private:
    void createRenderPasses();
    void createGBuffer();
    void createShadowMap();
    void createFramebuffers();
    void createPipelines();
    void createUniformBuffers();
    void createDescriptorSetLayouts();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, Camera *camera);
    void updateUniformBuffer(uint32_t currentImage, Camera *camera);
    VkShaderModule createShaderModule(const std::vector<char> &code);

    VulkanContext *m_context;
    VulkanSwapchain *m_swapchain;
    Model *m_model = nullptr;

    VkRenderPass m_geometryPass;
    VkRenderPass m_compositionPass;
    VkRenderPass m_shadowPass;

    VkPipelineLayout m_geometryPipelineLayout;
    VkPipeline m_geometryPipeline;

    VkPipelineLayout m_compositionPipelineLayout;
    VkPipeline m_compositionPipeline;

    VkPipelineLayout m_shadowPipelineLayout;
    VkPipeline m_shadowPipeline;

    // G-Buffer 첨부 이미지
    VkImage m_positionImage;
    VkDeviceMemory m_positionImageMemory;
    VkImageView m_positionImageView;

    VkImage m_albedoImage;
    VkDeviceMemory m_albedoImageMemory;
    VkImageView m_albedoImageView;

    VkImage m_specularImage;
    VkDeviceMemory m_specularImageMemory;
    VkImageView m_specularImageView;

    VkImage m_normalImage;
    VkDeviceMemory m_normalImageMemory;
    VkImageView m_normalImageView;

    VkImage m_depthImage;
    VkDeviceMemory m_depthImageMemory;
    VkImageView m_depthImageView;

    // 섀도우 맵
    VkImage m_shadowImage;
    VkDeviceMemory m_shadowImageMemory;
    VkImageView m_shadowImageView;
    VkSampler m_shadowSampler;
    VkSampler m_colorSampler;

    VkImage m_defaultTextureImage;
    VkDeviceMemory m_defaultTextureImageMemory;
    VkImageView m_defaultTextureImageView;

    std::vector<VkFramebuffer> m_geometryFramebuffers;
    std::vector<VkFramebuffer> m_compositionFramebuffers;
    VkFramebuffer m_shadowFramebuffer;

    VkDescriptorSetLayout m_geometryDescriptorSetLayout;
    VkDescriptorSetLayout m_compositionDescriptorSetLayout;
    VkDescriptorSetLayout m_shadowDescriptorSetLayout;

    VkDescriptorPool m_descriptorPool;
    std::vector<std::vector<VkDescriptorSet>> m_geometryDescriptorSets;
    std::vector<VkDescriptorSet> m_compositionDescriptorSets;
    std::vector<std::vector<VkDescriptorSet>> m_shadowDescriptorSets;

    std::vector<VkBuffer> m_uniformBuffers;
    std::vector<VkDeviceMemory> m_uniformBuffersMemory;

    std::vector<VkBuffer> m_lightBuffers;
    std::vector<VkDeviceMemory> m_lightBuffersMemory;

    std::vector<VkCommandBuffer> m_commandBuffers;

    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;

    uint32_t m_currentFrame = 0;
    const int MAX_FRAMES_IN_FLIGHT = 2;

    HWND m_hwnd = NULL;
    RenderConfig *m_config = nullptr;
    VkDescriptorPool m_imguiDescriptorPool = VK_NULL_HANDLE;
};
