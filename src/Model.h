#pragma once
#include "Math.h"
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class VulkanContext;

namespace tinyobj
{
struct material_t;
}

struct Vertex
{
    Math::Vec3 pos;
    Math::Vec3 normal;
    Math::Vec2 texCoord; // Math.h에 Vec2 필요

    static VkVertexInputBindingDescription getBindingDescription();
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
};

struct Material
{
    VkImage diffuseImage;
    VkDeviceMemory diffuseImageMemory;
    VkImageView diffuseImageView;
    VkSampler diffuseSampler;
    // ... 노멀 맵 등을 추가할 수 있다.
};

struct SubMesh
{
    uint32_t indexStart;
    uint32_t indexCount;
    int materialIndex;
};

class Model
{
  public:
    Model(VulkanContext *context, const std::string &path);
    ~Model();

    void Draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout);

    VkBuffer GetVertexBuffer() const
    {
        return m_vertexBuffer;
    }
    VkBuffer GetIndexBuffer() const
    {
        return m_indexBuffer;
    }
    const std::vector<SubMesh> &GetSubMeshes() const
    {
        return m_subMeshes;
    }
    const std::vector<Material> &GetMaterials() const
    {
        return m_materials;
    }
    VkSampler GetTextureSampler() const
    {
        return m_textureSampler;
    }

  private:
    void loadModel(const std::string &path);
    void createVertexBuffer();
    void createIndexBuffer();
    void createTextureSampler();

    void loadMaterials(const std::string &baseDir, const std::vector<tinyobj::material_t> &materials);
    void loadTexture(const std::string &filename, VkImage &textureImage, VkDeviceMemory &textureImageMemory);
    void createFallbackTexture(VkImage &textureImage, VkDeviceMemory &textureImageMemory, VkImageView &textureImageView);
    static void transitionImageLayout(VulkanContext *context, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);

    VulkanContext *m_context;

    std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;

    VkBuffer m_vertexBuffer;
    VkDeviceMemory m_vertexBufferMemory;
    VkBuffer m_indexBuffer;
    VkDeviceMemory m_indexBufferMemory;

    std::vector<SubMesh> m_subMeshes;
    std::vector<Material> m_materials;
    VkSampler m_textureSampler;
};
