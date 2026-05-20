#pragma once
#include <vulkan/vulkan.h>

class VulkanContext;

namespace VulkanUtils
{
uint32_t FindMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
void CreateBuffer(VulkanContext *context, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory);
VkCommandBuffer BeginSingleTimeCommands(VulkanContext *context);
void EndSingleTimeCommands(VulkanContext *context, VkCommandBuffer commandBuffer);
void CopyBuffer(VulkanContext *context, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
void CreateImage(VulkanContext *context, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image, VkDeviceMemory &imageMemory);
VkImageView CreateImageView(VulkanContext *context, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
} // VulkanUtils 네임스페이스 끝
