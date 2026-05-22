#include "DeferredRenderer.h"
#include "Camera.h"
#include "Model.h"
#include "RenderConfig.h"
#include "VulkanContext.h"
#include "VulkanSwapchain.h"
#include "VulkanUtils.h"
#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_win32.h"
#include "imgui.h"
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>

static std::vector<char> readFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file: " + filename);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

static Math::Vec4 multiply(const Math::Mat4 &matrix, const Math::Vec4 &value)
{
    return Math::Vec4(
        matrix.m[0][0] * value.x + matrix.m[1][0] * value.y + matrix.m[2][0] * value.z + matrix.m[3][0] * value.w,
        matrix.m[0][1] * value.x + matrix.m[1][1] * value.y + matrix.m[2][1] * value.z + matrix.m[3][1] * value.w,
        matrix.m[0][2] * value.x + matrix.m[1][2] * value.y + matrix.m[2][2] * value.z + matrix.m[3][2] * value.w,
        matrix.m[0][3] * value.x + matrix.m[1][3] * value.y + matrix.m[2][3] * value.z + matrix.m[3][3] * value.w);
}

static bool clipToPlane(Math::Vec4 &a, Math::Vec4 &b, float da, float db)
{
    if (da >= 0.0f && db >= 0.0f)
    {
        return true;
    }

    if (da < 0.0f && db < 0.0f)
    {
        return false;
    }

    float t = da / (da - db);
    Math::Vec4 clipped(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t);

    if (da < 0.0f)
    {
        a = clipped;
    }
    else
    {
        b = clipped;
    }

    return true;
}

static bool clipLineToClipSpace(Math::Vec4 &a, Math::Vec4 &b)
{
    return clipToPlane(a, b, a.x + a.w, b.x + b.w) &&
           clipToPlane(a, b, a.w - a.x, b.w - b.x) &&
           clipToPlane(a, b, a.y + a.w, b.y + b.w) &&
           clipToPlane(a, b, a.w - a.y, b.w - b.y) &&
           clipToPlane(a, b, a.z, b.z) &&
           clipToPlane(a, b, a.w - a.z, b.w - b.z);
}

static ImVec2 clipToScreen(const Math::Vec4 &clip, VkExtent2D extent)
{
    float ndcX = clip.x / clip.w;
    float ndcY = clip.y / clip.w;
    return ImVec2((ndcX * 0.5f + 0.5f) * extent.width, (ndcY * 0.5f + 0.5f) * extent.height);
}

static void transitionImageLayout(VulkanContext *context, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkCommandBuffer commandBuffer = VulkanUtils::BeginSingleTimeCommands(context);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && (format == VK_FORMAT_D32_SFLOAT))
    {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    VulkanUtils::EndSingleTimeCommands(context, commandBuffer);
}

DeferredRenderer::DeferredRenderer(VulkanContext *context, VulkanSwapchain *swapchain)
    : m_context(context), m_swapchain(swapchain)
{
}

DeferredRenderer::~DeferredRenderer()
{
    VkDevice device = m_context->GetDevice();
    vkDeviceWaitIdle(device);

    if (ImGui::GetCurrentContext() != nullptr)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, m_renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, m_imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, m_inFlightFences[i], nullptr);

        vkDestroyBuffer(device, m_uniformBuffers[i], nullptr);
        vkFreeMemory(device, m_uniformBuffersMemory[i], nullptr);

        vkDestroyBuffer(device, m_lightBuffers[i], nullptr);
        vkFreeMemory(device, m_lightBuffersMemory[i], nullptr);

        vkDestroyBuffer(device, m_hdrSettingsBuffers[i], nullptr);
        vkFreeMemory(device, m_hdrSettingsBuffersMemory[i], nullptr);

        for (size_t m = 0; m < m_materialSettingsBuffers[i].size(); m++)
        {
            vkDestroyBuffer(device, m_materialSettingsBuffers[i][m], nullptr);
            vkFreeMemory(device, m_materialSettingsBuffersMemory[i][m], nullptr);
        }
    }

    // 프레임버퍼 정리
    for (auto fb : m_geometryFramebuffers)
    {
        vkDestroyFramebuffer(device, fb, nullptr);
    }
    for (auto fb : m_compositionFramebuffers)
    {
        vkDestroyFramebuffer(device, fb, nullptr);
    }
    for (auto fb : m_imguiFramebuffers)
    {
        vkDestroyFramebuffer(device, fb, nullptr);
    }
    if (m_hdrFramebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(device, m_hdrFramebuffer, nullptr);
        m_hdrFramebuffer = VK_NULL_HANDLE;
    }
    if (m_shadowFramebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(device, m_shadowFramebuffer, nullptr);
    }

    // 파이프라인과 레이아웃 정리
    vkDestroyPipeline(device, m_geometryPipeline, nullptr);
    vkDestroyPipelineLayout(device, m_geometryPipelineLayout, nullptr);
    vkDestroyPipeline(device, m_compositionPipeline, nullptr);
    vkDestroyPipelineLayout(device, m_compositionPipelineLayout, nullptr);
    vkDestroyPipeline(device, m_finalPipeline, nullptr);
    vkDestroyPipelineLayout(device, m_finalPipelineLayout, nullptr);
    vkDestroyPipeline(device, m_shadowPipeline, nullptr);
    vkDestroyPipelineLayout(device, m_shadowPipelineLayout, nullptr);

    // 디스크립터 레이아웃과 풀 정리
    vkDestroyDescriptorSetLayout(device, m_geometryDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, m_compositionDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, m_finalDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, m_shadowDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);

    // G-Buffer 첨부 이미지 정리
    vkDestroyImageView(device, m_positionImageView, nullptr);
    vkDestroyImage(device, m_positionImage, nullptr);
    vkFreeMemory(device, m_positionImageMemory, nullptr);

    vkDestroyImageView(device, m_normalImageView, nullptr);
    vkDestroyImage(device, m_normalImage, nullptr);
    vkFreeMemory(device, m_normalImageMemory, nullptr);

    vkDestroyImageView(device, m_albedoImageView, nullptr);
    vkDestroyImage(device, m_albedoImage, nullptr);
    vkFreeMemory(device, m_albedoImageMemory, nullptr);

    vkDestroyImageView(device, m_specularImageView, nullptr);
    vkDestroyImage(device, m_specularImage, nullptr);
    vkFreeMemory(device, m_specularImageMemory, nullptr);

    vkDestroyImageView(device, m_depthImageView, nullptr);
    vkDestroyImage(device, m_depthImage, nullptr);
    vkFreeMemory(device, m_depthImageMemory, nullptr);

    vkDestroyImageView(device, m_hdrColorImageView, nullptr);
    vkDestroyImage(device, m_hdrColorImage, nullptr);
    vkFreeMemory(device, m_hdrColorImageMemory, nullptr);

    // 섀도우 맵 정리
    vkDestroyImageView(device, m_shadowImageView, nullptr);
    vkDestroyImage(device, m_shadowImage, nullptr);
    vkFreeMemory(device, m_shadowImageMemory, nullptr);

    vkDestroySampler(device, m_shadowSampler, nullptr);
    vkDestroySampler(device, m_colorSampler, nullptr);

    // 기본 대체 텍스처 정리
    vkDestroyImageView(device, m_defaultTextureImageView, nullptr);
    vkDestroyImage(device, m_defaultTextureImage, nullptr);
    vkFreeMemory(device, m_defaultTextureImageMemory, nullptr);

    // 렌더 패스 정리
    vkDestroyRenderPass(device, m_geometryPass, nullptr);
    vkDestroyRenderPass(device, m_compositionPass, nullptr);
    vkDestroyRenderPass(device, m_finalPass, nullptr);
    vkDestroyRenderPass(device, m_imguiPass, nullptr);
    vkDestroyRenderPass(device, m_shadowPass, nullptr);
}

void DeferredRenderer::Init(Model *model, HWND hwnd, RenderConfig *config)
{
    m_model = model;
    m_hwnd = hwnd;
    m_config = config;

    // 모델에 로드된 디퓨즈 맵이 없을 때 검증 레이어 오류를 피하기 위한 1x1 기본 디퓨즈 텍스처
    uint32_t whitePixel = 0xFFFFFFFF;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtils::CreateBuffer(m_context, 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(m_context->GetDevice(), stagingBufferMemory, 0, 4, 0, &data);
    memcpy(data, &whitePixel, 4);
    vkUnmapMemory(m_context->GetDevice(), stagingBufferMemory);

    VulkanUtils::CreateImage(m_context, 1, 1, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_defaultTextureImage, m_defaultTextureImageMemory);

    transitionImageLayout(m_context, m_defaultTextureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkCommandBuffer commandBuffer = VulkanUtils::BeginSingleTimeCommands(m_context);
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {1, 1, 1};

    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, m_defaultTextureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VulkanUtils::EndSingleTimeCommands(m_context, commandBuffer);

    transitionImageLayout(m_context, m_defaultTextureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(m_context->GetDevice(), stagingBuffer, nullptr);
    vkFreeMemory(m_context->GetDevice(), stagingBufferMemory, nullptr);

    m_defaultTextureImageView = VulkanUtils::CreateImageView(m_context, m_defaultTextureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);

    createGBuffer();
    createShadowMap();
    createRenderPasses();
    createFramebuffers();
    createDescriptorSetLayouts();
    createPipelines();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();

    // ImGui 초기화
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplWin32_Init(m_hwnd);
    // 단일 창 모드이므로 뷰포트 플래그를 비활성화하여 Vulkan 백엔드의 뷰포트 assert/초기화를 방지한다.
    io.BackendFlags &= ~ImGuiBackendFlags_PlatformHasViewports;

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = m_context->GetInstance();
    init_info.PhysicalDevice = m_context->GetPhysicalDevice();
    init_info.Device = m_context->GetDevice();
    init_info.QueueFamily = m_context->GetGraphicsQueueFamily();
    init_info.Queue = m_context->GetGraphicsQueue();
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPoolSize = 1000;
    init_info.MinImageCount = 2;
    init_info.ImageCount = static_cast<uint32_t>(m_swapchain->GetImages().size());

    init_info.PipelineInfoMain.RenderPass = m_imguiPass;
    init_info.PipelineInfoMain.Subpass = 0;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);
}

void DeferredRenderer::createGBuffer()
{
    VkExtent2D extent = m_swapchain->GetExtent();

    VulkanUtils::CreateImage(m_context, extent.width, extent.height,
                             VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_positionImage, m_positionImageMemory);
    m_positionImageView = VulkanUtils::CreateImageView(m_context, m_positionImage, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

    VulkanUtils::CreateImage(m_context, extent.width, extent.height,
                             VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_normalImage, m_normalImageMemory);
    m_normalImageView = VulkanUtils::CreateImageView(m_context, m_normalImage, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

    VulkanUtils::CreateImage(m_context, extent.width, extent.height,
                             VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_albedoImage, m_albedoImageMemory);
    m_albedoImageView = VulkanUtils::CreateImageView(m_context, m_albedoImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);

    VulkanUtils::CreateImage(m_context, extent.width, extent.height,
                             VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_specularImage, m_specularImageMemory);
    m_specularImageView = VulkanUtils::CreateImageView(m_context, m_specularImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

    VulkanUtils::CreateImage(m_context, extent.width, extent.height,
                             VK_FORMAT_D32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_depthImage, m_depthImageMemory);
    m_depthImageView = VulkanUtils::CreateImageView(m_context, m_depthImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);

    VulkanUtils::CreateImage(m_context, extent.width, extent.height,
                             VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_hdrColorImage, m_hdrColorImageMemory);
    m_hdrColorImageView = VulkanUtils::CreateImageView(m_context, m_hdrColorImage, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    if (vkCreateSampler(m_context->GetDevice(), &samplerInfo, nullptr, &m_colorSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create G-Buffer sampler!");
    }
}

void DeferredRenderer::createShadowMap()
{
    uint32_t width = 2048;
    uint32_t height = 2048;

    VulkanUtils::CreateImage(m_context, width, height,
                             VK_FORMAT_D32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_shadowImage, m_shadowImageMemory);
    m_shadowImageView = VulkanUtils::CreateImageView(m_context, m_shadowImage, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    if (vkCreateSampler(m_context->GetDevice(), &samplerInfo, nullptr, &m_shadowSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shadow sampler!");
    }
}

void DeferredRenderer::createFramebuffers()
{
    VkDevice device = m_context->GetDevice();
    VkExtent2D extent = m_swapchain->GetExtent();
    size_t swapchainImageCount = m_swapchain->GetImages().size();

    VkFramebufferCreateInfo shadowFbInfo{};
    shadowFbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    shadowFbInfo.renderPass = m_shadowPass;
    shadowFbInfo.attachmentCount = 1;
    shadowFbInfo.pAttachments = &m_shadowImageView;
    shadowFbInfo.width = 2048;
    shadowFbInfo.height = 2048;
    shadowFbInfo.layers = 1;

    if (vkCreateFramebuffer(device, &shadowFbInfo, nullptr, &m_shadowFramebuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shadow framebuffer!");
    }

    VkFramebufferCreateInfo hdrFbInfo{};
    hdrFbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    hdrFbInfo.renderPass = m_compositionPass;
    hdrFbInfo.attachmentCount = 1;
    hdrFbInfo.pAttachments = &m_hdrColorImageView;
    hdrFbInfo.width = extent.width;
    hdrFbInfo.height = extent.height;
    hdrFbInfo.layers = 1;

    if (vkCreateFramebuffer(device, &hdrFbInfo, nullptr, &m_hdrFramebuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create HDR framebuffer!");
    }

    m_geometryFramebuffers.resize(swapchainImageCount);
    for (size_t i = 0; i < swapchainImageCount; i++)
    {
        std::array<VkImageView, 5> attachments = {
            m_positionImageView,
            m_normalImageView,
            m_albedoImageView,
            m_specularImageView,
            m_depthImageView};

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_geometryPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        fbInfo.pAttachments = attachments.data();
        fbInfo.width = extent.width;
        fbInfo.height = extent.height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_geometryFramebuffers[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create geometry framebuffer!");
        }
    }

    m_compositionFramebuffers.resize(swapchainImageCount);
    m_imguiFramebuffers.resize(swapchainImageCount);
    for (size_t i = 0; i < swapchainImageCount; i++)
    {
        VkImageView attachments[] = {
            m_swapchain->GetImageViews()[i]};

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_finalPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = attachments;
        fbInfo.width = extent.width;
        fbInfo.height = extent.height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_compositionFramebuffers[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create composition framebuffer!");
        }

        VkFramebufferCreateInfo imguiFbInfo{};
        imguiFbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        imguiFbInfo.renderPass = m_imguiPass;
        imguiFbInfo.attachmentCount = 1;
        imguiFbInfo.pAttachments = attachments;
        imguiFbInfo.width = extent.width;
        imguiFbInfo.height = extent.height;
        imguiFbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &imguiFbInfo, nullptr, &m_imguiFramebuffers[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create ImGui framebuffer!");
        }
    }
}

void DeferredRenderer::createRenderPasses()
{
    VkDevice device = m_context->GetDevice();

    VkAttachmentDescription shadowDepthAttachment{};
    shadowDepthAttachment.format = VK_FORMAT_D32_SFLOAT;
    shadowDepthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    shadowDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    shadowDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    shadowDepthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    shadowDepthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    shadowDepthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    shadowDepthAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference shadowDepthAttachmentRef{};
    shadowDepthAttachmentRef.attachment = 0;
    shadowDepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription shadowSubpass{};
    shadowSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    shadowSubpass.colorAttachmentCount = 0;
    shadowSubpass.pDepthStencilAttachment = &shadowDepthAttachmentRef;

    std::array<VkSubpassDependency, 2> shadowDependencies;
    shadowDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    shadowDependencies[0].dstSubpass = 0;
    shadowDependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    shadowDependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    shadowDependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    shadowDependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    shadowDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    shadowDependencies[1].srcSubpass = 0;
    shadowDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    shadowDependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    shadowDependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    shadowDependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    shadowDependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    shadowDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo shadowRenderPassInfo{};
    shadowRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    shadowRenderPassInfo.attachmentCount = 1;
    shadowRenderPassInfo.pAttachments = &shadowDepthAttachment;
    shadowRenderPassInfo.subpassCount = 1;
    shadowRenderPassInfo.pSubpasses = &shadowSubpass;
    shadowRenderPassInfo.dependencyCount = static_cast<uint32_t>(shadowDependencies.size());
    shadowRenderPassInfo.pDependencies = shadowDependencies.data();

    if (vkCreateRenderPass(device, &shadowRenderPassInfo, nullptr, &m_shadowPass) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shadow render pass!");
    }

    std::array<VkAttachmentDescription, 5> attachments{};

    attachments[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    attachments[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    attachments[2].format = VK_FORMAT_R8G8B8A8_SRGB;
    attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    attachments[3].format = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[3].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    attachments[4].format = VK_FORMAT_D32_SFLOAT;
    attachments[4].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[4].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[4].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[4].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[4].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::array<VkAttachmentReference, 4> colorAttachmentRefs{};
    colorAttachmentRefs[0] = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    colorAttachmentRefs[1] = {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    colorAttachmentRefs[2] = {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    colorAttachmentRefs[3] = {3, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 4;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(colorAttachmentRefs.size());
    subpass.pColorAttachments = colorAttachmentRefs.data();
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    std::array<VkSubpassDependency, 2> dependencies;
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &m_geometryPass) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create geometry render pass!");
    }

    VkAttachmentDescription compColorAttachment{};
    compColorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    compColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    compColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    compColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    compColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    compColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    compColorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    compColorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference compColorAttachmentRef{};
    compColorAttachmentRef.attachment = 0;
    compColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription compSubpass{};
    compSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    compSubpass.colorAttachmentCount = 1;
    compSubpass.pColorAttachments = &compColorAttachmentRef;
    compSubpass.pDepthStencilAttachment = nullptr;

    std::array<VkSubpassDependency, 2> compDependencies;
    compDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    compDependencies[0].dstSubpass = 0;
    compDependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    compDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    compDependencies[0].srcAccessMask = 0;
    compDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    compDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    compDependencies[1].srcSubpass = 0;
    compDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    compDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    compDependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    compDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    compDependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    compDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo compRenderPassInfo{};
    compRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    compRenderPassInfo.attachmentCount = 1;
    compRenderPassInfo.pAttachments = &compColorAttachment;
    compRenderPassInfo.subpassCount = 1;
    compRenderPassInfo.pSubpasses = &compSubpass;
    compRenderPassInfo.dependencyCount = static_cast<uint32_t>(compDependencies.size());
    compRenderPassInfo.pDependencies = compDependencies.data();

    if (vkCreateRenderPass(device, &compRenderPassInfo, nullptr, &m_compositionPass) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create composition render pass!");
    }

    VkAttachmentDescription finalColorAttachment{};
    finalColorAttachment.format = m_swapchain->GetImageFormat();
    finalColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    finalColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    finalColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    finalColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    finalColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    finalColorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    finalColorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference finalColorAttachmentRef{};
    finalColorAttachmentRef.attachment = 0;
    finalColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription finalSubpass{};
    finalSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    finalSubpass.colorAttachmentCount = 1;
    finalSubpass.pColorAttachments = &finalColorAttachmentRef;
    finalSubpass.pDepthStencilAttachment = nullptr;

    std::array<VkSubpassDependency, 2> finalDependencies{};
    finalDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    finalDependencies[0].dstSubpass = 0;
    finalDependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    finalDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    finalDependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    finalDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    finalDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    finalDependencies[1].srcSubpass = 0;
    finalDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    finalDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    finalDependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    finalDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    finalDependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    finalDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo finalRenderPassInfo{};
    finalRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    finalRenderPassInfo.attachmentCount = 1;
    finalRenderPassInfo.pAttachments = &finalColorAttachment;
    finalRenderPassInfo.subpassCount = 1;
    finalRenderPassInfo.pSubpasses = &finalSubpass;
    finalRenderPassInfo.dependencyCount = static_cast<uint32_t>(finalDependencies.size());
    finalRenderPassInfo.pDependencies = finalDependencies.data();

    if (vkCreateRenderPass(device, &finalRenderPassInfo, nullptr, &m_finalPass) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create final render pass!");
    }

    VkAttachmentDescription imguiColorAttachment{};
    imguiColorAttachment.format = m_swapchain->GetImageFormat();
    imguiColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    imguiColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    imguiColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    imguiColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    imguiColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    imguiColorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    imguiColorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference imguiColorAttachmentRef{};
    imguiColorAttachmentRef.attachment = 0;
    imguiColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription imguiSubpass{};
    imguiSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    imguiSubpass.colorAttachmentCount = 1;
    imguiSubpass.pColorAttachments = &imguiColorAttachmentRef;
    imguiSubpass.pDepthStencilAttachment = nullptr;

    std::array<VkSubpassDependency, 2> imguiDependencies{};
    imguiDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    imguiDependencies[0].dstSubpass = 0;
    imguiDependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    imguiDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    imguiDependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imguiDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imguiDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    imguiDependencies[1].srcSubpass = 0;
    imguiDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    imguiDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    imguiDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    imguiDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imguiDependencies[1].dstAccessMask = 0;
    imguiDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo imguiRenderPassInfo{};
    imguiRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    imguiRenderPassInfo.attachmentCount = 1;
    imguiRenderPassInfo.pAttachments = &imguiColorAttachment;
    imguiRenderPassInfo.subpassCount = 1;
    imguiRenderPassInfo.pSubpasses = &imguiSubpass;
    imguiRenderPassInfo.dependencyCount = static_cast<uint32_t>(imguiDependencies.size());
    imguiRenderPassInfo.pDependencies = imguiDependencies.data();

    if (vkCreateRenderPass(device, &imguiRenderPassInfo, nullptr, &m_imguiPass) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create ImGui render pass!");
    }
}

void DeferredRenderer::createDescriptorSetLayouts()
{
    VkDevice device = m_context->GetDevice();

    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding diffuseSamplerLayoutBinding{};
    diffuseSamplerLayoutBinding.binding = 1;
    diffuseSamplerLayoutBinding.descriptorCount = 1;
    diffuseSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    diffuseSamplerLayoutBinding.pImmutableSamplers = nullptr;
    diffuseSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding normalSamplerLayoutBinding{};
    normalSamplerLayoutBinding.binding = 2;
    normalSamplerLayoutBinding.descriptorCount = 1;
    normalSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    normalSamplerLayoutBinding.pImmutableSamplers = nullptr;
    normalSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding alphaSamplerLayoutBinding{};
    alphaSamplerLayoutBinding.binding = 3;
    alphaSamplerLayoutBinding.descriptorCount = 1;
    alphaSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    alphaSamplerLayoutBinding.pImmutableSamplers = nullptr;
    alphaSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding specularSamplerLayoutBinding{};
    specularSamplerLayoutBinding.binding = 4;
    specularSamplerLayoutBinding.descriptorCount = 1;
    specularSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    specularSamplerLayoutBinding.pImmutableSamplers = nullptr;
    specularSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding materialSettingsLayoutBinding{};
    materialSettingsLayoutBinding.binding = 5;
    materialSettingsLayoutBinding.descriptorCount = 1;
    materialSettingsLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    materialSettingsLayoutBinding.pImmutableSamplers = nullptr;
    materialSettingsLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 6> geomBindings = {uboLayoutBinding, diffuseSamplerLayoutBinding, normalSamplerLayoutBinding, alphaSamplerLayoutBinding, specularSamplerLayoutBinding, materialSettingsLayoutBinding};
    VkDescriptorSetLayoutCreateInfo geomLayoutInfo{};
    geomLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    geomLayoutInfo.bindingCount = static_cast<uint32_t>(geomBindings.size());
    geomLayoutInfo.pBindings = geomBindings.data();

    if (vkCreateDescriptorSetLayout(device, &geomLayoutInfo, nullptr, &m_geometryDescriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create geometry descriptor set layout!");
    }

    VkDescriptorSetLayoutBinding shadowModelBinding{};
    shadowModelBinding.binding = 0;
    shadowModelBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    shadowModelBinding.descriptorCount = 1;
    shadowModelBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    shadowModelBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding shadowLightBinding{};
    shadowLightBinding.binding = 1;
    shadowLightBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    shadowLightBinding.descriptorCount = 1;
    shadowLightBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    shadowLightBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding shadowAlphaBinding{};
    shadowAlphaBinding.binding = 2;
    shadowAlphaBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowAlphaBinding.descriptorCount = 1;
    shadowAlphaBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    shadowAlphaBinding.pImmutableSamplers = nullptr;

    std::array<VkDescriptorSetLayoutBinding, 3> shadowBindings = {shadowModelBinding, shadowLightBinding, shadowAlphaBinding};
    VkDescriptorSetLayoutCreateInfo shadowLayoutInfo{};
    shadowLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    shadowLayoutInfo.bindingCount = static_cast<uint32_t>(shadowBindings.size());
    shadowLayoutInfo.pBindings = shadowBindings.data();

    if (vkCreateDescriptorSetLayout(device, &shadowLayoutInfo, nullptr, &m_shadowDescriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shadow descriptor set layout!");
    }

    std::array<VkDescriptorSetLayoutBinding, 6> compBindings{};
    for (uint32_t i = 0; i < 5; i++)
    {
        compBindings[i].binding = i;
        compBindings[i].descriptorCount = 1;
        compBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        compBindings[i].pImmutableSamplers = nullptr;
        compBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    compBindings[5].binding = 5;
    compBindings[5].descriptorCount = 1;
    compBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    compBindings[5].pImmutableSamplers = nullptr;
    compBindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo compLayoutInfo{};
    compLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    compLayoutInfo.bindingCount = static_cast<uint32_t>(compBindings.size());
    compLayoutInfo.pBindings = compBindings.data();

    if (vkCreateDescriptorSetLayout(device, &compLayoutInfo, nullptr, &m_compositionDescriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create composition descriptor set layout!");
    }

    VkDescriptorSetLayoutBinding hdrSamplerBinding{};
    hdrSamplerBinding.binding = 0;
    hdrSamplerBinding.descriptorCount = 1;
    hdrSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    hdrSamplerBinding.pImmutableSamplers = nullptr;
    hdrSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding hdrSettingsBinding{};
    hdrSettingsBinding.binding = 1;
    hdrSettingsBinding.descriptorCount = 1;
    hdrSettingsBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    hdrSettingsBinding.pImmutableSamplers = nullptr;
    hdrSettingsBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> finalBindings = {hdrSamplerBinding, hdrSettingsBinding};
    VkDescriptorSetLayoutCreateInfo finalLayoutInfo{};
    finalLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    finalLayoutInfo.bindingCount = static_cast<uint32_t>(finalBindings.size());
    finalLayoutInfo.pBindings = finalBindings.data();

    if (vkCreateDescriptorSetLayout(device, &finalLayoutInfo, nullptr, &m_finalDescriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create final descriptor set layout!");
    }
}

void DeferredRenderer::createPipelines()
{
    VkDevice device = m_context->GetDevice();

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicStateInfo.pDynamicStates = dynamicStates.data();

    auto shadowVertCode = readFile("shaders/shadow.vert.spv");
    auto shadowFragCode = readFile("shaders/shadow.frag.spv");
    auto geomVertCode = readFile("shaders/geometry.vert.spv");
    auto geomFragCode = readFile("shaders/geometry.frag.spv");
    auto compVertCode = readFile("shaders/composition.vert.spv");
    auto compFragCode = readFile("shaders/composition.frag.spv");
    auto finalVertCode = readFile("shaders/final.vert.spv");
    auto finalFragCode = readFile("shaders/final.frag.spv");

    VkShaderModule shadowVertModule = createShaderModule(shadowVertCode);
    VkShaderModule shadowFragModule = createShaderModule(shadowFragCode);
    VkShaderModule geomVertModule = createShaderModule(geomVertCode);
    VkShaderModule geomFragModule = createShaderModule(geomFragCode);
    VkShaderModule compVertModule = createShaderModule(compVertCode);
    VkShaderModule compFragModule = createShaderModule(compFragCode);
    VkShaderModule finalVertModule = createShaderModule(finalVertCode);
    VkShaderModule finalFragModule = createShaderModule(finalFragCode);

    VkPipelineShaderStageCreateInfo shadowVertStageInfo{};
    shadowVertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowVertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    shadowVertStageInfo.module = shadowVertModule;
    shadowVertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shadowFragStageInfo{};
    shadowFragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowFragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shadowFragStageInfo.module = shadowFragModule;
    shadowFragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shadowStages[] = {shadowVertStageInfo, shadowFragStageInfo};

    auto bindingDesc = Vertex::getBindingDescription();
    auto attribDescs = Vertex::getAttributeDescriptions();
    std::array<VkVertexInputAttributeDescription, 2> shadowAttribDescs = {
        attribDescs[0],
        attribDescs[2]};

    VkPipelineVertexInputStateCreateInfo shadowVertexInputInfo{};
    shadowVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    shadowVertexInputInfo.vertexBindingDescriptionCount = 1;
    shadowVertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    shadowVertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(shadowAttribDescs.size());
    shadowVertexInputInfo.pVertexAttributeDescriptions = shadowAttribDescs.data();

    VkPipelineInputAssemblyStateCreateInfo shadowInputAssembly{};
    shadowInputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    shadowInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    shadowInputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport shadowViewport{};
    shadowViewport.x = 0.0f;
    shadowViewport.y = 0.0f;
    shadowViewport.width = 2048.0f;
    shadowViewport.height = 2048.0f;
    shadowViewport.minDepth = 0.0f;
    shadowViewport.maxDepth = 1.0f;

    VkRect2D shadowScissor{};
    shadowScissor.offset = {0, 0};
    shadowScissor.extent = {2048, 2048};

    VkPipelineViewportStateCreateInfo shadowViewportState{};
    shadowViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    shadowViewportState.viewportCount = 1;
    shadowViewportState.pViewports = &shadowViewport;
    shadowViewportState.scissorCount = 1;
    shadowViewportState.pScissors = &shadowScissor;

    VkPipelineRasterizationStateCreateInfo shadowRasterizer{};
    shadowRasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    shadowRasterizer.depthClampEnable = VK_FALSE;
    shadowRasterizer.rasterizerDiscardEnable = VK_FALSE;
    shadowRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    shadowRasterizer.lineWidth = 1.0f;
    shadowRasterizer.cullMode = VK_CULL_MODE_NONE;
    shadowRasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    shadowRasterizer.depthBiasEnable = VK_TRUE;
    shadowRasterizer.depthBiasConstantFactor = 0.5f;
    shadowRasterizer.depthBiasClamp = 0.0f;
    shadowRasterizer.depthBiasSlopeFactor = 1.0f;

    VkPipelineMultisampleStateCreateInfo shadowMultisampling{};
    shadowMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    shadowMultisampling.sampleShadingEnable = VK_FALSE;
    shadowMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo shadowDepthStencil{};
    shadowDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    shadowDepthStencil.depthTestEnable = VK_TRUE;
    shadowDepthStencil.depthWriteEnable = VK_TRUE;
    shadowDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    shadowDepthStencil.depthBoundsTestEnable = VK_FALSE;
    shadowDepthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo shadowColorBlending{};
    shadowColorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    shadowColorBlending.logicOpEnable = VK_FALSE;
    shadowColorBlending.attachmentCount = 0;
    shadowColorBlending.pAttachments = nullptr;

    VkPipelineLayoutCreateInfo shadowPipelineLayoutInfo{};
    shadowPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    shadowPipelineLayoutInfo.setLayoutCount = 1;
    shadowPipelineLayoutInfo.pSetLayouts = &m_shadowDescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &shadowPipelineLayoutInfo, nullptr, &m_shadowPipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shadow pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo shadowPipelineInfo{};
    shadowPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    shadowPipelineInfo.stageCount = 2;
    shadowPipelineInfo.pStages = shadowStages;
    shadowPipelineInfo.pVertexInputState = &shadowVertexInputInfo;
    shadowPipelineInfo.pInputAssemblyState = &shadowInputAssembly;
    shadowPipelineInfo.pViewportState = &shadowViewportState;
    shadowPipelineInfo.pRasterizationState = &shadowRasterizer;
    shadowPipelineInfo.pMultisampleState = &shadowMultisampling;
    shadowPipelineInfo.pDepthStencilState = &shadowDepthStencil;
    shadowPipelineInfo.pColorBlendState = &shadowColorBlending;
    shadowPipelineInfo.pDynamicState = &dynamicStateInfo;
    shadowPipelineInfo.layout = m_shadowPipelineLayout;
    shadowPipelineInfo.renderPass = m_shadowPass;
    shadowPipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &shadowPipelineInfo, nullptr, &m_shadowPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shadow graphics pipeline!");
    }

    VkPipelineShaderStageCreateInfo geomVertStageInfo{};
    geomVertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    geomVertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    geomVertStageInfo.module = geomVertModule;
    geomVertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo geomFragStageInfo{};
    geomFragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    geomFragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    geomFragStageInfo.module = geomFragModule;
    geomFragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo geomStages[] = {geomVertStageInfo, geomFragStageInfo};

    VkPipelineVertexInputStateCreateInfo geomVertexInputInfo{};
    geomVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    geomVertexInputInfo.vertexBindingDescriptionCount = 1;
    geomVertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    geomVertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribDescs.size());
    geomVertexInputInfo.pVertexAttributeDescriptions = attribDescs.data();

    VkPipelineInputAssemblyStateCreateInfo geomInputAssembly{};
    geomInputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    geomInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    geomInputAssembly.primitiveRestartEnable = VK_FALSE;

    VkExtent2D extent = m_swapchain->GetExtent();
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)extent.width;
    viewport.height = (float)extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;

    VkPipelineViewportStateCreateInfo geomViewportState{};
    geomViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    geomViewportState.viewportCount = 1;
    geomViewportState.pViewports = &viewport;
    geomViewportState.scissorCount = 1;
    geomViewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo geomRasterizer{};
    geomRasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    geomRasterizer.depthClampEnable = VK_FALSE;
    geomRasterizer.rasterizerDiscardEnable = VK_FALSE;
    geomRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    geomRasterizer.lineWidth = 1.0f;
    geomRasterizer.cullMode = VK_CULL_MODE_NONE;
    geomRasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    geomRasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo geomMultisampling{};
    geomMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    geomMultisampling.sampleShadingEnable = VK_FALSE;
    geomMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo geomDepthStencil{};
    geomDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    geomDepthStencil.depthTestEnable = VK_TRUE;
    geomDepthStencil.depthWriteEnable = VK_TRUE;
    geomDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    geomDepthStencil.depthBoundsTestEnable = VK_FALSE;
    geomDepthStencil.stencilTestEnable = VK_FALSE;

    std::array<VkPipelineColorBlendAttachmentState, 4> geomBlendAttachments{};
    for (uint32_t i = 0; i < 4; i++)
    {
        geomBlendAttachments[i].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        geomBlendAttachments[i].blendEnable = VK_FALSE;
    }

    VkPipelineColorBlendStateCreateInfo geomColorBlending{};
    geomColorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    geomColorBlending.logicOpEnable = VK_FALSE;
    geomColorBlending.attachmentCount = static_cast<uint32_t>(geomBlendAttachments.size());
    geomColorBlending.pAttachments = geomBlendAttachments.data();

    VkPipelineLayoutCreateInfo geomPipelineLayoutInfo{};
    geomPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    geomPipelineLayoutInfo.setLayoutCount = 1;
    geomPipelineLayoutInfo.pSetLayouts = &m_geometryDescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &geomPipelineLayoutInfo, nullptr, &m_geometryPipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create geometry pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo geomPipelineInfo{};
    geomPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    geomPipelineInfo.stageCount = 2;
    geomPipelineInfo.pStages = geomStages;
    geomPipelineInfo.pVertexInputState = &geomVertexInputInfo;
    geomPipelineInfo.pInputAssemblyState = &geomInputAssembly;
    geomPipelineInfo.pViewportState = &geomViewportState;
    geomPipelineInfo.pRasterizationState = &geomRasterizer;
    geomPipelineInfo.pMultisampleState = &geomMultisampling;
    geomPipelineInfo.pDepthStencilState = &geomDepthStencil;
    geomPipelineInfo.pColorBlendState = &geomColorBlending;
    geomPipelineInfo.pDynamicState = &dynamicStateInfo;
    geomPipelineInfo.layout = m_geometryPipelineLayout;
    geomPipelineInfo.renderPass = m_geometryPass;
    geomPipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &geomPipelineInfo, nullptr, &m_geometryPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create geometry graphics pipeline!");
    }

    VkPipelineShaderStageCreateInfo compVertStageInfo{};
    compVertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compVertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    compVertStageInfo.module = compVertModule;
    compVertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo compFragStageInfo{};
    compFragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compFragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    compFragStageInfo.module = compFragModule;
    compFragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo compStages[] = {compVertStageInfo, compFragStageInfo};

    VkPipelineShaderStageCreateInfo finalVertStageInfo{};
    finalVertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    finalVertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    finalVertStageInfo.module = finalVertModule;
    finalVertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo finalFragStageInfo{};
    finalFragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    finalFragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    finalFragStageInfo.module = finalFragModule;
    finalFragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo finalStages[] = {finalVertStageInfo, finalFragStageInfo};

    VkPipelineVertexInputStateCreateInfo compVertexInputInfo{};
    compVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    compVertexInputInfo.vertexBindingDescriptionCount = 0;
    compVertexInputInfo.pVertexBindingDescriptions = nullptr;
    compVertexInputInfo.vertexAttributeDescriptionCount = 0;
    compVertexInputInfo.pVertexAttributeDescriptions = nullptr;

    VkPipelineInputAssemblyStateCreateInfo compInputAssembly{};
    compInputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    compInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    compInputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo compViewportState{};
    compViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    compViewportState.viewportCount = 1;
    compViewportState.pViewports = &viewport;
    compViewportState.scissorCount = 1;
    compViewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo compRasterizer{};
    compRasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    compRasterizer.depthClampEnable = VK_FALSE;
    compRasterizer.rasterizerDiscardEnable = VK_FALSE;
    compRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    compRasterizer.lineWidth = 1.0f;
    compRasterizer.cullMode = VK_CULL_MODE_NONE;
    compRasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    compRasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo compMultisampling{};
    compMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    compMultisampling.sampleShadingEnable = VK_FALSE;
    compMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo compDepthStencil{};
    compDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    compDepthStencil.depthTestEnable = VK_FALSE;
    compDepthStencil.depthWriteEnable = VK_FALSE;
    compDepthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    compDepthStencil.depthBoundsTestEnable = VK_FALSE;
    compDepthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState compBlendAttachment{};
    compBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    compBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo compColorBlending{};
    compColorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    compColorBlending.logicOpEnable = VK_FALSE;
    compColorBlending.attachmentCount = 1;
    compColorBlending.pAttachments = &compBlendAttachment;

    VkPipelineLayoutCreateInfo compPipelineLayoutInfo{};
    compPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    compPipelineLayoutInfo.setLayoutCount = 1;
    compPipelineLayoutInfo.pSetLayouts = &m_compositionDescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &compPipelineLayoutInfo, nullptr, &m_compositionPipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create composition pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo compPipelineInfo{};
    compPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    compPipelineInfo.stageCount = 2;
    compPipelineInfo.pStages = compStages;
    compPipelineInfo.pVertexInputState = &compVertexInputInfo;
    compPipelineInfo.pInputAssemblyState = &compInputAssembly;
    compPipelineInfo.pViewportState = &compViewportState;
    compPipelineInfo.pRasterizationState = &compRasterizer;
    compPipelineInfo.pMultisampleState = &compMultisampling;
    compPipelineInfo.pDepthStencilState = &compDepthStencil;
    compPipelineInfo.pColorBlendState = &compColorBlending;
    compPipelineInfo.pDynamicState = &dynamicStateInfo;
    compPipelineInfo.layout = m_compositionPipelineLayout;
    compPipelineInfo.renderPass = m_compositionPass;
    compPipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &compPipelineInfo, nullptr, &m_compositionPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create composition graphics pipeline!");
    }

    VkPipelineLayoutCreateInfo finalPipelineLayoutInfo{};
    finalPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    finalPipelineLayoutInfo.setLayoutCount = 1;
    finalPipelineLayoutInfo.pSetLayouts = &m_finalDescriptorSetLayout;

    if (vkCreatePipelineLayout(device, &finalPipelineLayoutInfo, nullptr, &m_finalPipelineLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create final pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo finalPipelineInfo = compPipelineInfo;
    finalPipelineInfo.pStages = finalStages;
    finalPipelineInfo.layout = m_finalPipelineLayout;
    finalPipelineInfo.renderPass = m_finalPass;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &finalPipelineInfo, nullptr, &m_finalPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create final graphics pipeline!");
    }

    vkDestroyShaderModule(device, shadowVertModule, nullptr);
    vkDestroyShaderModule(device, shadowFragModule, nullptr);
    vkDestroyShaderModule(device, geomVertModule, nullptr);
    vkDestroyShaderModule(device, geomFragModule, nullptr);
    vkDestroyShaderModule(device, compVertModule, nullptr);
    vkDestroyShaderModule(device, compFragModule, nullptr);
    vkDestroyShaderModule(device, finalVertModule, nullptr);
    vkDestroyShaderModule(device, finalFragModule, nullptr);
}

void DeferredRenderer::createUniformBuffers()
{
    VkDeviceSize uboSize = sizeof(UniformBufferObject);
    VkDeviceSize lightSize = sizeof(LightBufferObject);
    VkDeviceSize hdrSettingsSize = sizeof(HdrSettingsBufferObject);
    VkDeviceSize materialSettingsSize = sizeof(MaterialSettingsBufferObject);
    size_t materialCount = m_model ? m_model->GetMaterials().size() : 1;
    if (materialCount == 0)
    {
        materialCount = 1;
    }

    m_uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    m_uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    m_lightBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    m_lightBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    m_hdrSettingsBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    m_hdrSettingsBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    m_materialSettingsBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    m_materialSettingsBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        VulkanUtils::CreateBuffer(m_context, uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_uniformBuffers[i], m_uniformBuffersMemory[i]);
        VulkanUtils::CreateBuffer(m_context, lightSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_lightBuffers[i], m_lightBuffersMemory[i]);
        VulkanUtils::CreateBuffer(m_context, hdrSettingsSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_hdrSettingsBuffers[i], m_hdrSettingsBuffersMemory[i]);

        m_materialSettingsBuffers[i].resize(materialCount);
        m_materialSettingsBuffersMemory[i].resize(materialCount);
        for (size_t m = 0; m < materialCount; m++)
        {
            VulkanUtils::CreateBuffer(m_context, materialSettingsSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_materialSettingsBuffers[i][m], m_materialSettingsBuffersMemory[i][m]);
        }
    }
}

void DeferredRenderer::createDescriptorPool()
{
    VkDevice device = m_context->GetDevice();

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 250);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 250);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 300);

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void DeferredRenderer::createDescriptorSets()
{
    VkDevice device = m_context->GetDevice();
    size_t materialCount = m_model ? m_model->GetMaterials().size() : 1;
    if (materialCount == 0)
        materialCount = 1;

    m_geometryDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        m_geometryDescriptorSets[i].resize(materialCount);
    }

    m_shadowDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        m_shadowDescriptorSets[i].resize(materialCount);
    }
    m_compositionDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    m_finalDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

    // 지오메트리 디스크립터 세트 할당
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        std::vector<VkDescriptorSetLayout> geomLayouts(materialCount, m_geometryDescriptorSetLayout);
        VkDescriptorSetAllocateInfo geomAllocInfo{};
        geomAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        geomAllocInfo.descriptorPool = m_descriptorPool;
        geomAllocInfo.descriptorSetCount = static_cast<uint32_t>(materialCount);
        geomAllocInfo.pSetLayouts = geomLayouts.data();

        if (vkAllocateDescriptorSets(device, &geomAllocInfo, m_geometryDescriptorSets[i].data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate geometry descriptor sets!");
        }
    }

    // 섀도우 디스크립터 세트 할당
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        std::vector<VkDescriptorSetLayout> shadowLayouts(materialCount, m_shadowDescriptorSetLayout);
        VkDescriptorSetAllocateInfo shadowAllocInfo{};
        shadowAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        shadowAllocInfo.descriptorPool = m_descriptorPool;
        shadowAllocInfo.descriptorSetCount = static_cast<uint32_t>(materialCount);
        shadowAllocInfo.pSetLayouts = shadowLayouts.data();

        if (vkAllocateDescriptorSets(device, &shadowAllocInfo, m_shadowDescriptorSets[i].data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate shadow descriptor sets!");
        }
    }

    // 컴포지션 디스크립터 세트 할당
    std::vector<VkDescriptorSetLayout> compLayouts(MAX_FRAMES_IN_FLIGHT, m_compositionDescriptorSetLayout);
    VkDescriptorSetAllocateInfo compAllocInfo{};
    compAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    compAllocInfo.descriptorPool = m_descriptorPool;
    compAllocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    compAllocInfo.pSetLayouts = compLayouts.data();

    if (vkAllocateDescriptorSets(device, &compAllocInfo, m_compositionDescriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate composition descriptor sets!");
    }

    std::vector<VkDescriptorSetLayout> finalLayouts(MAX_FRAMES_IN_FLIGHT, m_finalDescriptorSetLayout);
    VkDescriptorSetAllocateInfo finalAllocInfo{};
    finalAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    finalAllocInfo.descriptorPool = m_descriptorPool;
    finalAllocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    finalAllocInfo.pSetLayouts = finalLayouts.data();

    if (vkAllocateDescriptorSets(device, &finalAllocInfo, m_finalDescriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate final descriptor sets!");
    }

    // 디스크립터 세트 채우기
    const auto &modelMaterials = m_model ? m_model->GetMaterials() : std::vector<Material>();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        // 각 머티리얼의 지오메트리 디스크립터 세트 채우기
        for (size_t m = 0; m < materialCount; m++)
        {
            VkDescriptorBufferInfo geomBufferInfo{};
            geomBufferInfo.buffer = m_uniformBuffers[i];
            geomBufferInfo.offset = 0;
            geomBufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo geomDiffuseImageInfo{};
            geomDiffuseImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo geomNormalImageInfo{};
            geomNormalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo geomAlphaImageInfo{};
            geomAlphaImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo geomSpecularImageInfo{};
            geomSpecularImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkDescriptorBufferInfo geomMaterialSettingsBufferInfo{};
            geomMaterialSettingsBufferInfo.buffer = m_materialSettingsBuffers[i][m];
            geomMaterialSettingsBufferInfo.offset = 0;
            geomMaterialSettingsBufferInfo.range = sizeof(MaterialSettingsBufferObject);

            if (m < modelMaterials.size())
            {
                geomDiffuseImageInfo.imageView = modelMaterials[m].diffuseImageView;
                geomDiffuseImageInfo.sampler = m_model->GetTextureSampler();
                geomNormalImageInfo.imageView = modelMaterials[m].normalImageView;
                geomNormalImageInfo.sampler = m_model->GetTextureSampler();
                geomAlphaImageInfo.imageView = modelMaterials[m].alphaImageView;
                geomAlphaImageInfo.sampler = m_model->GetTextureSampler();
                geomSpecularImageInfo.imageView = modelMaterials[m].specularImageView;
                geomSpecularImageInfo.sampler = m_model->GetTextureSampler();
            }
            else
            {
                geomDiffuseImageInfo.imageView = m_defaultTextureImageView;
                geomDiffuseImageInfo.sampler = m_colorSampler;
                geomNormalImageInfo.imageView = m_defaultTextureImageView;
                geomNormalImageInfo.sampler = m_colorSampler;
                geomAlphaImageInfo.imageView = m_defaultTextureImageView;
                geomAlphaImageInfo.sampler = m_colorSampler;
                geomSpecularImageInfo.imageView = m_defaultTextureImageView;
                geomSpecularImageInfo.sampler = m_colorSampler;
            }

            std::array<VkWriteDescriptorSet, 6> geomWrites{};
            geomWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            geomWrites[0].dstSet = m_geometryDescriptorSets[i][m];
            geomWrites[0].dstBinding = 0;
            geomWrites[0].dstArrayElement = 0;
            geomWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            geomWrites[0].descriptorCount = 1;
            geomWrites[0].pBufferInfo = &geomBufferInfo;

            geomWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            geomWrites[1].dstSet = m_geometryDescriptorSets[i][m];
            geomWrites[1].dstBinding = 1;
            geomWrites[1].dstArrayElement = 0;
            geomWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            geomWrites[1].descriptorCount = 1;
            geomWrites[1].pImageInfo = &geomDiffuseImageInfo;

            geomWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            geomWrites[2].dstSet = m_geometryDescriptorSets[i][m];
            geomWrites[2].dstBinding = 2;
            geomWrites[2].dstArrayElement = 0;
            geomWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            geomWrites[2].descriptorCount = 1;
            geomWrites[2].pImageInfo = &geomNormalImageInfo;

            geomWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            geomWrites[3].dstSet = m_geometryDescriptorSets[i][m];
            geomWrites[3].dstBinding = 3;
            geomWrites[3].dstArrayElement = 0;
            geomWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            geomWrites[3].descriptorCount = 1;
            geomWrites[3].pImageInfo = &geomAlphaImageInfo;

            geomWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            geomWrites[4].dstSet = m_geometryDescriptorSets[i][m];
            geomWrites[4].dstBinding = 4;
            geomWrites[4].dstArrayElement = 0;
            geomWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            geomWrites[4].descriptorCount = 1;
            geomWrites[4].pImageInfo = &geomSpecularImageInfo;

            geomWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            geomWrites[5].dstSet = m_geometryDescriptorSets[i][m];
            geomWrites[5].dstBinding = 5;
            geomWrites[5].dstArrayElement = 0;
            geomWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            geomWrites[5].descriptorCount = 1;
            geomWrites[5].pBufferInfo = &geomMaterialSettingsBufferInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(geomWrites.size()), geomWrites.data(), 0, nullptr);
        }

        for (size_t m = 0; m < materialCount; m++)
        {
            VkDescriptorBufferInfo shadowModelBufferInfo{};
            shadowModelBufferInfo.buffer = m_uniformBuffers[i];
            shadowModelBufferInfo.offset = 0;
            shadowModelBufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorBufferInfo shadowLightBufferInfo{};
            shadowLightBufferInfo.buffer = m_lightBuffers[i];
            shadowLightBufferInfo.offset = 0;
            shadowLightBufferInfo.range = sizeof(LightBufferObject);

            VkDescriptorImageInfo shadowAlphaImageInfo{};
            shadowAlphaImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (m < modelMaterials.size())
            {
                shadowAlphaImageInfo.imageView = modelMaterials[m].alphaImageView;
                shadowAlphaImageInfo.sampler = m_model->GetTextureSampler();
            }
            else
            {
                shadowAlphaImageInfo.imageView = m_defaultTextureImageView;
                shadowAlphaImageInfo.sampler = m_colorSampler;
            }

            std::array<VkWriteDescriptorSet, 3> shadowWrites{};
            shadowWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            shadowWrites[0].dstSet = m_shadowDescriptorSets[i][m];
            shadowWrites[0].dstBinding = 0;
            shadowWrites[0].dstArrayElement = 0;
            shadowWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            shadowWrites[0].descriptorCount = 1;
            shadowWrites[0].pBufferInfo = &shadowModelBufferInfo;

            shadowWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            shadowWrites[1].dstSet = m_shadowDescriptorSets[i][m];
            shadowWrites[1].dstBinding = 1;
            shadowWrites[1].dstArrayElement = 0;
            shadowWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            shadowWrites[1].descriptorCount = 1;
            shadowWrites[1].pBufferInfo = &shadowLightBufferInfo;

            shadowWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            shadowWrites[2].dstSet = m_shadowDescriptorSets[i][m];
            shadowWrites[2].dstBinding = 2;
            shadowWrites[2].dstArrayElement = 0;
            shadowWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            shadowWrites[2].descriptorCount = 1;
            shadowWrites[2].pImageInfo = &shadowAlphaImageInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(shadowWrites.size()), shadowWrites.data(), 0, nullptr);
        }

        std::array<VkDescriptorImageInfo, 5> compImageInfos{};
        compImageInfos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        compImageInfos[0].imageView = m_positionImageView;
        compImageInfos[0].sampler = m_colorSampler;

        compImageInfos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        compImageInfos[1].imageView = m_normalImageView;
        compImageInfos[1].sampler = m_colorSampler;

        compImageInfos[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        compImageInfos[2].imageView = m_albedoImageView;
        compImageInfos[2].sampler = m_colorSampler;

        compImageInfos[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        compImageInfos[3].imageView = m_shadowImageView;
        compImageInfos[3].sampler = m_shadowSampler;

        compImageInfos[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        compImageInfos[4].imageView = m_specularImageView;
        compImageInfos[4].sampler = m_colorSampler;

        VkDescriptorBufferInfo compLightBufferInfo{};
        compLightBufferInfo.buffer = m_lightBuffers[i];
        compLightBufferInfo.offset = 0;
        compLightBufferInfo.range = sizeof(LightBufferObject);

        std::array<VkWriteDescriptorSet, 6> compWrites{};
        for (uint32_t b = 0; b < 5; b++)
        {
            compWrites[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            compWrites[b].dstSet = m_compositionDescriptorSets[i];
            compWrites[b].dstBinding = b;
            compWrites[b].dstArrayElement = 0;
            compWrites[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            compWrites[b].descriptorCount = 1;
            compWrites[b].pImageInfo = &compImageInfos[b];
        }

        compWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        compWrites[5].dstSet = m_compositionDescriptorSets[i];
        compWrites[5].dstBinding = 5;
        compWrites[5].dstArrayElement = 0;
        compWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        compWrites[5].descriptorCount = 1;
        compWrites[5].pBufferInfo = &compLightBufferInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(compWrites.size()), compWrites.data(), 0, nullptr);

        VkDescriptorImageInfo hdrImageInfo{};
        hdrImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        hdrImageInfo.imageView = m_hdrColorImageView;
        hdrImageInfo.sampler = m_colorSampler;

        VkDescriptorBufferInfo hdrSettingsBufferInfo{};
        hdrSettingsBufferInfo.buffer = m_hdrSettingsBuffers[i];
        hdrSettingsBufferInfo.offset = 0;
        hdrSettingsBufferInfo.range = sizeof(HdrSettingsBufferObject);

        std::array<VkWriteDescriptorSet, 2> finalWrites{};
        finalWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        finalWrites[0].dstSet = m_finalDescriptorSets[i];
        finalWrites[0].dstBinding = 0;
        finalWrites[0].dstArrayElement = 0;
        finalWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        finalWrites[0].descriptorCount = 1;
        finalWrites[0].pImageInfo = &hdrImageInfo;

        finalWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        finalWrites[1].dstSet = m_finalDescriptorSets[i];
        finalWrites[1].dstBinding = 1;
        finalWrites[1].dstArrayElement = 0;
        finalWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        finalWrites[1].descriptorCount = 1;
        finalWrites[1].pBufferInfo = &hdrSettingsBufferInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(finalWrites.size()), finalWrites.data(), 0, nullptr);
    }
}

void DeferredRenderer::createCommandBuffers()
{
    m_commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_context->GetCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());

    if (vkAllocateCommandBuffers(m_context->GetDevice(), &allocInfo, m_commandBuffers.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void DeferredRenderer::createSyncObjects()
{
    m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkDevice device = m_context->GetDevice();
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &m_inFlightFences[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}

void DeferredRenderer::Recreate()
{
    VkDevice device = m_context->GetDevice();
    vkDeviceWaitIdle(device);

    for (auto fb : m_geometryFramebuffers)
    {
        vkDestroyFramebuffer(device, fb, nullptr);
    }
    for (auto fb : m_compositionFramebuffers)
    {
        vkDestroyFramebuffer(device, fb, nullptr);
    }
    for (auto fb : m_imguiFramebuffers)
    {
        vkDestroyFramebuffer(device, fb, nullptr);
    }
    if (m_hdrFramebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(device, m_hdrFramebuffer, nullptr);
        m_hdrFramebuffer = VK_NULL_HANDLE;
    }

    vkDestroyImageView(device, m_positionImageView, nullptr);
    vkDestroyImage(device, m_positionImage, nullptr);
    vkFreeMemory(device, m_positionImageMemory, nullptr);

    vkDestroyImageView(device, m_normalImageView, nullptr);
    vkDestroyImage(device, m_normalImage, nullptr);
    vkFreeMemory(device, m_normalImageMemory, nullptr);

    vkDestroyImageView(device, m_albedoImageView, nullptr);
    vkDestroyImage(device, m_albedoImage, nullptr);
    vkFreeMemory(device, m_albedoImageMemory, nullptr);

    vkDestroyImageView(device, m_specularImageView, nullptr);
    vkDestroyImage(device, m_specularImage, nullptr);
    vkFreeMemory(device, m_specularImageMemory, nullptr);

    vkDestroyImageView(device, m_depthImageView, nullptr);
    vkDestroyImage(device, m_depthImage, nullptr);
    vkFreeMemory(device, m_depthImageMemory, nullptr);

    vkDestroyImageView(device, m_hdrColorImageView, nullptr);
    vkDestroyImage(device, m_hdrColorImage, nullptr);
    vkFreeMemory(device, m_hdrColorImageMemory, nullptr);

    createGBuffer();
    createFramebuffers();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        std::array<VkDescriptorImageInfo, 5> compImageInfos{};
        compImageInfos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        compImageInfos[0].imageView = m_positionImageView;
        compImageInfos[0].sampler = m_colorSampler;

        compImageInfos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        compImageInfos[1].imageView = m_normalImageView;
        compImageInfos[1].sampler = m_colorSampler;

        compImageInfos[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        compImageInfos[2].imageView = m_albedoImageView;
        compImageInfos[2].sampler = m_colorSampler;

        compImageInfos[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        compImageInfos[3].imageView = m_shadowImageView;
        compImageInfos[3].sampler = m_shadowSampler;

        compImageInfos[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        compImageInfos[4].imageView = m_specularImageView;
        compImageInfos[4].sampler = m_colorSampler;

        std::array<VkWriteDescriptorSet, 5> compWrites{};
        for (uint32_t b = 0; b < 5; b++)
        {
            compWrites[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            compWrites[b].dstSet = m_compositionDescriptorSets[i];
            compWrites[b].dstBinding = b;
            compWrites[b].dstArrayElement = 0;
            compWrites[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            compWrites[b].descriptorCount = 1;
            compWrites[b].pImageInfo = &compImageInfos[b];
        }

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(compWrites.size()), compWrites.data(), 0, nullptr);

        VkDescriptorImageInfo hdrImageInfo{};
        hdrImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        hdrImageInfo.imageView = m_hdrColorImageView;
        hdrImageInfo.sampler = m_colorSampler;

        VkWriteDescriptorSet finalWrite{};
        finalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        finalWrite.dstSet = m_finalDescriptorSets[i];
        finalWrite.dstBinding = 0;
        finalWrite.dstArrayElement = 0;
        finalWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        finalWrite.descriptorCount = 1;
        finalWrite.pImageInfo = &hdrImageInfo;

        vkUpdateDescriptorSets(device, 1, &finalWrite, 0, nullptr);
    }
}

void DeferredRenderer::DrawFrame(Camera *camera)
{
    VkDevice device = m_context->GetDevice();

    vkWaitForFences(device, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, m_swapchain->GetSwapchain(), UINT64_MAX, m_imageAvailableSemaphores[m_currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        Recreate();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("failed to acquire swapchain image!");
    }

    vkResetFences(device, 1, &m_inFlightFences[m_currentFrame]);

    // Dear ImGui 프레임 시작
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    // ImGui UI 위젯 생성
    {
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(370, 560), ImGuiCond_FirstUseEver);
        ImGui::Begin("Sponza Render Settings", nullptr, ImGuiWindowFlags_NoCollapse);

        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 4.0f);

        // 카메라 설정
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Camera Parameters");
        ImGui::SliderFloat("Speed", &camera->MovementSpeed, 10.0f, 2000.0f, "%.1f");
        ImGui::SliderFloat("Sensitivity", &camera->MouseSensitivity, 0.01f, 1.0f, "%.2f");

        float pos[3] = {camera->Position.x, camera->Position.y, camera->Position.z};
        if (ImGui::DragFloat3("Position", pos, 1.0f, -5000.0f, 5000.0f, "%.1f"))
        {
            camera->Position = Math::Vec3(pos[0], pos[1], pos[2]);
        }

        ImGui::Dummy(ImVec2(0.0f, 5.0f));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0.0f, 5.0f));

        // 조명 설정
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Lighting Parameters");
        if (m_config)
        {
            ImGui::Checkbox("Auto Rotate Light", &m_config->autoRotateLight);
        }

        static float lightYaw = 0.0f;
        static float lightPitch = -45.0f;
        static bool anglesSynced = false;

        if (m_config)
        {
            if (m_config->autoRotateLight || !anglesSynced)
            {
                lightPitch = std::asin(m_config->lightDir.y) * (180.0f / 3.14159265f);
                float cosPitch = std::cos(lightPitch * (3.14159265f / 180.0f));
                if (std::abs(cosPitch) > 1e-4f)
                {
                    lightYaw = std::atan2(m_config->lightDir.z, m_config->lightDir.x) * (180.0f / 3.14159265f);
                }
                else
                {
                    lightYaw = 0.0f;
                }
                anglesSynced = true;
            }
        }

        if (m_config && !m_config->autoRotateLight)
        {
            bool changed = false;

            float dir[3] = {m_config->lightDir.x, m_config->lightDir.y, m_config->lightDir.z};
            if (ImGui::DragFloat3("Light Dir (X, Y, Z)", dir, 0.01f, -1.0f, 1.0f, "%.3f"))
            {
                m_config->lightDir.x = dir[0];
                m_config->lightDir.y = dir[1];
                m_config->lightDir.z = dir[2];
                float len = std::sqrt(m_config->lightDir.x * m_config->lightDir.x +
                                      m_config->lightDir.y * m_config->lightDir.y +
                                      m_config->lightDir.z * m_config->lightDir.z);
                if (len > 0.00001f)
                {
                    m_config->lightDir = m_config->lightDir.Normalize();
                }
                else
                {
                    m_config->lightDir = Math::Vec3(0.0f, -1.0f, 0.0f);
                }
                anglesSynced = false;
            }

            if (ImGui::SliderFloat("Light Pitch", &lightPitch, -89.0f, 89.0f, "%.1f"))
                changed = true;
            if (ImGui::SliderFloat("Light Yaw", &lightYaw, -180.0f, 180.0f, "%.1f"))
                changed = true;

            if (changed)
            {
                float yawRad = lightYaw * (3.14159265f / 180.0f);
                float pitchRad = lightPitch * (3.14159265f / 180.0f);
                m_config->lightDir.x = std::cos(yawRad) * std::cos(pitchRad);
                m_config->lightDir.y = std::sin(pitchRad);
                m_config->lightDir.z = std::sin(yawRad) * std::cos(pitchRad);
                m_config->lightDir = m_config->lightDir.Normalize();
            }
        }
        else
        {
            ImGui::TextDisabled("Light is rotating automatically...");
            anglesSynced = false;
        }

        if (m_config)
        {
            ImGui::Dummy(ImVec2(0.0f, 5.0f));
            ImGui::Separator();
            ImGui::Dummy(ImVec2(0.0f, 5.0f));
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Light Properties");

            float color[3] = {m_config->lightColor.x, m_config->lightColor.y, m_config->lightColor.z};
            if (ImGui::ColorEdit3("Light Color", color))
            {
                m_config->lightColor.x = color[0];
                m_config->lightColor.y = color[1];
                m_config->lightColor.z = color[2];
            }

            ImGui::SliderFloat("Ambient Intensity", &m_config->ambientStrength, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Specular Intensity", &m_config->specularStrength, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Specular Power", &m_config->specularPower, 1.0f, 256.0f, "%.1f");

            ImGui::Dummy(ImVec2(0.0f, 5.0f));
            ImGui::Separator();
            ImGui::Dummy(ImVec2(0.0f, 5.0f));
            ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1.0f), "HDR Parameters");
            ImGui::SliderFloat("Exposure", &m_config->hdrExposure, 0.1f, 5.0f, "%.2f");
            ImGui::SliderFloat("Bloom Strength", &m_config->bloomStrength, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Bloom Threshold", &m_config->bloomThreshold, 0.1f, 10.0f, "%.2f");
            ImGui::SliderFloat("Bloom Soft Knee", &m_config->bloomSoftKnee, 0.0f, 5.0f, "%.2f");
        }

        ImGui::Dummy(ImVec2(0.0f, 10.0f));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0.0f, 10.0f));

        // 저장 버튼
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.6f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.15f, 0.7f, 0.35f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.08f, 0.5f, 0.25f, 1.0f));
        if (ImGui::Button("Save Settings", ImVec2(-1, 40)))
        {
            if (m_config)
            {
                m_config->cameraSpeed = camera->MovementSpeed;
                m_config->cameraSensitivity = camera->MouseSensitivity;
                m_config->cameraPos = camera->Position;
                m_config->cameraYaw = camera->Yaw;
                m_config->cameraPitch = camera->Pitch;
                m_config->Save("config.txt");
            }
        }
        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar(2);

        ImGui::End();
    }

    // 렌더 목록과 선택된 재질 편집 UI
    {
        ImGui::SetNextWindowPos(ImVec2(390, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(470, 560), ImGuiCond_FirstUseEver);
        ImGui::Begin("Render Object Properties", nullptr, ImGuiWindowFlags_NoCollapse);

        if (m_model)
        {
            const auto &subMeshes = m_model->GetSubMeshes();
            auto &materials = m_model->GetMaterials();

            if (!subMeshes.empty())
            {
                if (m_selectedSubMesh < 0 || m_selectedSubMesh >= static_cast<int>(subMeshes.size()))
                {
                    m_selectedSubMesh = 0;
                }

                ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Render List");
                if (ImGui::BeginListBox("##RenderList", ImVec2(-1.0f, 190.0f)))
                {
                    for (int i = 0; i < static_cast<int>(subMeshes.size()); i++)
                    {
                        const SubMesh &subMesh = subMeshes[i];
                        int matIdx = subMesh.materialIndex;
                        const char *materialName = "Invalid Material";
                        if (matIdx >= 0 && matIdx < static_cast<int>(materials.size()))
                        {
                            materialName = materials[matIdx].name.c_str();
                        }

                        std::string label = std::to_string(i) + ": ";
                        label += subMesh.name.empty() ? "SubMesh" : subMesh.name;
                        label += " / ";
                        label += materialName;

                        if (ImGui::Selectable(label.c_str(), m_selectedSubMesh == i))
                        {
                            m_selectedSubMesh = i;
                        }
                    }
                    ImGui::EndListBox();
                }

                const SubMesh &selected = subMeshes[m_selectedSubMesh];
                int matIdx = selected.materialIndex;
                ImGui::Separator();
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Selected Object");
                ImGui::Text("SubMesh: %s", selected.name.empty() ? "(unnamed)" : selected.name.c_str());
                ImGui::Text("Index Start: %u", selected.indexStart);
                ImGui::Text("Index Count: %u", selected.indexCount);
                ImGui::Text("Material Index: %d", matIdx);

                if (matIdx >= 0 && matIdx < static_cast<int>(materials.size()))
                {
                    Material &material = materials[matIdx];
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1.0f), "Material Properties");
                    ImGui::Text("Name: %s", material.name.c_str());

                    float tint[3] = {material.colorTint.x, material.colorTint.y, material.colorTint.z};
                    if (ImGui::ColorEdit3("Color Tint", tint))
                    {
                        material.colorTint.x = tint[0];
                        material.colorTint.y = tint[1];
                        material.colorTint.z = tint[2];
                    }

                    ImGui::SliderFloat("Material HDR", &material.hdrIntensity, 0.0f, 20.0f, "%.2f");

                    if (ImGui::Button("Reset Material Color/HDR", ImVec2(-1.0f, 0.0f)))
                    {
                        material.colorTint = Math::Vec3(1.0f, 1.0f, 1.0f);
                        material.hdrIntensity = 1.0f;
                    }

                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Textures");
                    ImGui::TextWrapped("Diffuse: %s", material.diffuseTexturePath.empty() ? "(fallback)" : material.diffuseTexturePath.c_str());
                    ImGui::TextWrapped("Normal: %s", material.normalTexturePath.empty() ? "(fallback)" : material.normalTexturePath.c_str());
                    ImGui::TextWrapped("Alpha: %s", material.alphaTexturePath.empty() ? "(fallback)" : material.alphaTexturePath.c_str());
                    ImGui::TextWrapped("Specular: %s", material.specularTexturePath.empty() ? "(fallback)" : material.specularTexturePath.c_str());
                }
                else
                {
                    ImGui::TextDisabled("No editable material is bound to this SubMesh.");
                }
            }
            else
            {
                ImGui::TextDisabled("No renderable SubMesh was loaded.");
            }
        }
        else
        {
            ImGui::TextDisabled("No model is loaded.");
        }

        ImGui::End();
    }

    if (m_model)
    {
        const auto &subMeshes = m_model->GetSubMeshes();
        if (m_selectedSubMesh >= 0 && m_selectedSubMesh < static_cast<int>(subMeshes.size()))
        {
            const SubMesh &selected = subMeshes[m_selectedSubMesh];
            Math::Vec3 min = selected.boundsMin;
            Math::Vec3 max = selected.boundsMax;
            std::array<Math::Vec3, 8> corners = {
                Math::Vec3(min.x, min.y, min.z),
                Math::Vec3(max.x, min.y, min.z),
                Math::Vec3(max.x, max.y, min.z),
                Math::Vec3(min.x, max.y, min.z),
                Math::Vec3(min.x, min.y, max.z),
                Math::Vec3(max.x, min.y, max.z),
                Math::Vec3(max.x, max.y, max.z),
                Math::Vec3(min.x, max.y, max.z)};

            VkExtent2D extent = m_swapchain->GetExtent();
            float aspect = (float)extent.width / (float)extent.height;
            Math::Mat4 viewProjection = camera->GetProjectionMatrix(aspect) * camera->GetViewMatrix();
            std::array<Math::Vec4, 8> clipCorners{};
            for (size_t i = 0; i < corners.size(); i++)
            {
                clipCorners[i] = multiply(viewProjection, Math::Vec4(corners[i].x, corners[i].y, corners[i].z, 1.0f));
            }

            static const std::array<std::array<int, 2>, 12> edges = {{
                {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
                {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
                {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}}}};

            ImDrawList *drawList = ImGui::GetBackgroundDrawList();
            ImU32 green = IM_COL32(0, 255, 0, 255);
            for (const auto &edge : edges)
            {
                Math::Vec4 a = clipCorners[edge[0]];
                Math::Vec4 b = clipCorners[edge[1]];
                if (clipLineToClipSpace(a, b))
                {
                    drawList->AddLine(clipToScreen(a, extent), clipToScreen(b, extent), green, 2.0f);
                }
            }
        }
    }

    ImGui::Render();

    updateUniformBuffer(m_currentFrame, camera);

    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    vkResetCommandBuffer(cmd, 0);
    recordCommandBuffer(cmd, imageIndex, camera);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {m_imageAvailableSemaphores[m_currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkSemaphore signalSemaphores[] = {m_renderFinishedSemaphores[m_currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(m_context->GetGraphicsQueue(), 1, &submitInfo, m_inFlightFences[m_currentFrame]) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapchains[] = {m_swapchain->GetSwapchain()};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(m_context->GetPresentQueue(), &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    {
        Recreate();
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("failed to present swapchain image!");
    }

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void DeferredRenderer::updateUniformBuffer(uint32_t currentImage, Camera *camera)
{
    VkDevice device = m_context->GetDevice();

    UniformBufferObject ubo{};
    ubo.model = Math::Mat4::Identity();
    ubo.view = camera->GetViewMatrix();

    float aspect = (float)m_swapchain->GetExtent().width / (float)m_swapchain->GetExtent().height;
    ubo.proj = camera->GetProjectionMatrix(aspect);

    void *data;
    vkMapMemory(device, m_uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, m_uniformBuffersMemory[currentImage]);

    static float time = 0.0f;
    if (m_config && m_config->autoRotateLight)
    {
        time += 0.005f;
    }

    LightBufferObject lbo{};
    Math::Vec3 activeLightDir;

    if (m_config && !m_config->autoRotateLight)
    {
        activeLightDir = m_config->lightDir.Normalize();
    }
    else
    {
        float lightX = std::cos(time) * 1000.0f;
        float lightZ = std::sin(time) * 1000.0f;
        activeLightDir = Math::Vec3(lightX, -1500.0f, lightZ).Normalize();
        if (m_config)
        {
            m_config->lightDir = activeLightDir;
        }
    }

    Math::Vec3 lightPos = Math::Vec3(-activeLightDir.x * 1800.0f, -activeLightDir.y * 1800.0f, -activeLightDir.z * 1800.0f);
    Math::Mat4 lightView = Math::Mat4::LookAt(lightPos, Math::Vec3(0.0f, 0.0f, 0.0f), Math::Vec3(0.0f, 1.0f, 0.0f));
    Math::Mat4 lightProj = Math::Mat4::Ortho(-2500.0f, 2500.0f, -2500.0f, 2500.0f, 1.0f, 5000.0f);
    lbo.lightSpaceMatrix = lightProj * lightView;

    float ambientStrength = 0.1f;
    float specularStrength = 0.5f;
    Math::Vec3 lightColor = Math::Vec3(1.0f, 1.0f, 1.0f);
    float specularPower = 32.0f;

    if (m_config)
    {
        ambientStrength = m_config->ambientStrength;
        specularStrength = m_config->specularStrength;
        lightColor = m_config->lightColor;
        specularPower = m_config->specularPower;
    }

    lbo.lightDirAndAmbient = Math::Vec4(activeLightDir.x, activeLightDir.y, activeLightDir.z, ambientStrength);
    lbo.viewPosAndSpecular = Math::Vec4(camera->Position.x, camera->Position.y, camera->Position.z, specularStrength);
    lbo.lightColorAndSpecPower = Math::Vec4(lightColor.x, lightColor.y, lightColor.z, specularPower);

    vkMapMemory(device, m_lightBuffersMemory[currentImage], 0, sizeof(lbo), 0, &data);
    memcpy(data, &lbo, sizeof(lbo));
    vkUnmapMemory(device, m_lightBuffersMemory[currentImage]);

    HdrSettingsBufferObject hdrSettings{};
    hdrSettings.params = Math::Vec4(1.0f, 0.35f, 1.0f, 1.0f);

    if (m_config)
    {
        hdrSettings.params = Math::Vec4(m_config->hdrExposure, m_config->bloomStrength, m_config->bloomThreshold, m_config->bloomSoftKnee);
    }

    vkMapMemory(device, m_hdrSettingsBuffersMemory[currentImage], 0, sizeof(hdrSettings), 0, &data);
    memcpy(data, &hdrSettings, sizeof(hdrSettings));
    vkUnmapMemory(device, m_hdrSettingsBuffersMemory[currentImage]);

    if (m_model && currentImage < m_materialSettingsBuffersMemory.size())
    {
        auto &materials = m_model->GetMaterials();
        size_t materialCount = m_materialSettingsBuffersMemory[currentImage].size();
        for (size_t i = 0; i < materialCount; i++)
        {
            MaterialSettingsBufferObject materialSettings{};
            materialSettings.colorAndHdr = Math::Vec4(1.0f, 1.0f, 1.0f, 1.0f);

            if (i < materials.size())
            {
                const Material &material = materials[i];
                materialSettings.colorAndHdr = Math::Vec4(material.colorTint.x, material.colorTint.y, material.colorTint.z, material.hdrIntensity);
            }

            vkMapMemory(device, m_materialSettingsBuffersMemory[currentImage][i], 0, sizeof(materialSettings), 0, &data);
            memcpy(data, &materialSettings, sizeof(materialSettings));
            vkUnmapMemory(device, m_materialSettingsBuffersMemory[currentImage][i]);
        }
    }
}

void DeferredRenderer::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, Camera *camera)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkExtent2D extent = m_swapchain->GetExtent();

    VkRenderPassBeginInfo shadowPassInfo{};
    shadowPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    shadowPassInfo.renderPass = m_shadowPass;
    shadowPassInfo.framebuffer = m_shadowFramebuffer;
    shadowPassInfo.renderArea.offset = {0, 0};
    shadowPassInfo.renderArea.extent = {2048, 2048};

    VkClearValue shadowClearValue{};
    shadowClearValue.depthStencil = {1.0f, 0};
    shadowPassInfo.clearValueCount = 1;
    shadowPassInfo.pClearValues = &shadowClearValue;

    vkCmdBeginRenderPass(commandBuffer, &shadowPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    {
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = 2048.0f;
        viewport.height = 2048.0f;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = {2048, 2048};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);

        if (m_model)
        {
            VkBuffer vertexBuffers[] = {m_model->GetVertexBuffer()};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, m_model->GetIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);

            const auto &subMeshes = m_model->GetSubMeshes();
            const auto &descSets = m_shadowDescriptorSets[m_currentFrame];

            for (const auto &subMesh : subMeshes)
            {
                int matIdx = subMesh.materialIndex;
                if (matIdx < 0 || matIdx >= (int)descSets.size())
                {
                    matIdx = 0;
                }
                if (!descSets.empty())
                {
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipelineLayout, 0, 1, &descSets[matIdx], 0, nullptr);
                }
                vkCmdDrawIndexed(commandBuffer, subMesh.indexCount, 1, subMesh.indexStart, 0, 0);
            }
        }
    }
    vkCmdEndRenderPass(commandBuffer);

    VkRenderPassBeginInfo geomPassInfo{};
    geomPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    geomPassInfo.renderPass = m_geometryPass;
    geomPassInfo.framebuffer = m_geometryFramebuffers[imageIndex];
    geomPassInfo.renderArea.offset = {0, 0};
    geomPassInfo.renderArea.extent = extent;

    std::array<VkClearValue, 5> geomClearValues{};
    geomClearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    geomClearValues[1].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    geomClearValues[2].color = {{0.1f, 0.1f, 0.15f, 1.0f}};
    geomClearValues[3].color = {{1.0f, 1.0f, 1.0f, 1.0f}};
    geomClearValues[4].depthStencil = {1.0f, 0};

    geomPassInfo.clearValueCount = static_cast<uint32_t>(geomClearValues.size());
    geomPassInfo.pClearValues = geomClearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &geomPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    {
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)extent.width;
        viewport.height = (float)extent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = extent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_geometryPipeline);

        if (m_model)
        {
            VkBuffer vertexBuffers[] = {m_model->GetVertexBuffer()};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, m_model->GetIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);

            const auto &subMeshes = m_model->GetSubMeshes();
            const auto &descSets = m_geometryDescriptorSets[m_currentFrame];

            for (const auto &subMesh : subMeshes)
            {
                int matIdx = subMesh.materialIndex;
                if (matIdx < 0 || matIdx >= (int)descSets.size())
                {
                    matIdx = 0;
                }
                if (!descSets.empty())
                {
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_geometryPipelineLayout, 0, 1, &descSets[matIdx], 0, nullptr);
                }
                vkCmdDrawIndexed(commandBuffer, subMesh.indexCount, 1, subMesh.indexStart, 0, 0);
            }
        }
    }
    vkCmdEndRenderPass(commandBuffer);

    VkRenderPassBeginInfo compPassInfo{};
    compPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    compPassInfo.renderPass = m_compositionPass;
    compPassInfo.framebuffer = m_hdrFramebuffer;
    compPassInfo.renderArea.offset = {0, 0};
    compPassInfo.renderArea.extent = extent;

    VkClearValue compClearValue{};
    compClearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    compPassInfo.clearValueCount = 1;
    compPassInfo.pClearValues = &compClearValue;

    VkViewport compViewport{};
    compViewport.x = 0.0f;
    compViewport.y = 0.0f;
    compViewport.width = (float)extent.width;
    compViewport.height = (float)extent.height;
    compViewport.minDepth = 0.0f;
    compViewport.maxDepth = 1.0f;

    VkRect2D compScissor{};
    compScissor.offset = {0, 0};
    compScissor.extent = extent;

    vkCmdBeginRenderPass(commandBuffer, &compPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositionPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_compositionPipelineLayout, 0, 1, &m_compositionDescriptorSets[m_currentFrame], 0, nullptr);

        vkCmdSetViewport(commandBuffer, 0, 1, &compViewport);

        vkCmdSetScissor(commandBuffer, 0, 1, &compScissor);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    vkCmdEndRenderPass(commandBuffer);

    VkRenderPassBeginInfo finalPassInfo{};
    finalPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    finalPassInfo.renderPass = m_finalPass;
    finalPassInfo.framebuffer = m_compositionFramebuffers[imageIndex];
    finalPassInfo.renderArea.offset = {0, 0};
    finalPassInfo.renderArea.extent = extent;

    VkClearValue finalClearValue{};
    finalClearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    finalPassInfo.clearValueCount = 1;
    finalPassInfo.pClearValues = &finalClearValue;

    vkCmdBeginRenderPass(commandBuffer, &finalPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_finalPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_finalPipelineLayout, 0, 1, &m_finalDescriptorSets[m_currentFrame], 0, nullptr);

        vkCmdSetViewport(commandBuffer, 0, 1, &compViewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &compScissor);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    vkCmdEndRenderPass(commandBuffer);

    VkRenderPassBeginInfo imguiPassInfo{};
    imguiPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    imguiPassInfo.renderPass = m_imguiPass;
    imguiPassInfo.framebuffer = m_imguiFramebuffers[imageIndex];
    imguiPassInfo.renderArea.offset = {0, 0};
    imguiPassInfo.renderArea.extent = extent;
    imguiPassInfo.clearValueCount = 0;
    imguiPassInfo.pClearValues = nullptr;

    vkCmdBeginRenderPass(commandBuffer, &imguiPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    {
        // UI는 3D 후처리 패스 밖에서 swapchain 위에 직접 렌더링한다.
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    }
    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to record command buffer!");
    }
}

VkShaderModule DeferredRenderer::createShaderModule(const std::vector<char> &code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_context->GetDevice(), &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}
