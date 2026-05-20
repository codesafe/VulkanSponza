#include "Model.h"
#include "VulkanContext.h"
#include "VulkanUtils.h"
#define TINYOBJLOADER_USE_DOUBLE
#define TINYOBJLOADER_IMPLEMENTATION
#include "../vendor/tiny_obj_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../vendor/stb_image.h"
#include <filesystem>
#include <iostream>
#include <unordered_map>

VkVertexInputBindingDescription Vertex::getBindingDescription()
{
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

std::vector<VkVertexInputAttributeDescription> Vertex::getAttributeDescriptions()
{
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions(3);

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, normal);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

    return attributeDescriptions;
}

Model::Model(VulkanContext *context, const std::string &path)
    : m_context(context)
{
    createTextureSampler();
    loadModel(path);
    createVertexBuffer();
    createIndexBuffer();
}

Model::~Model()
{
    vkDestroySampler(m_context->GetDevice(), m_textureSampler, nullptr);
    vkDestroyBuffer(m_context->GetDevice(), m_indexBuffer, nullptr);
    vkFreeMemory(m_context->GetDevice(), m_indexBufferMemory, nullptr);
    vkDestroyBuffer(m_context->GetDevice(), m_vertexBuffer, nullptr);
    vkFreeMemory(m_context->GetDevice(), m_vertexBufferMemory, nullptr);

    for (auto &material : m_materials)
    {
        vkDestroyImageView(m_context->GetDevice(), material.diffuseImageView, nullptr);
        vkDestroyImage(m_context->GetDevice(), material.diffuseImage, nullptr);
        vkFreeMemory(m_context->GetDevice(), material.diffuseImageMemory, nullptr);
    }
}

void Model::loadModel(const std::string &path)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::string base_dir = path.substr(0, path.find_last_of('/') + 1);
    if (base_dir.empty())
        base_dir = path.substr(0, path.find_last_of('\\') + 1);

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str(), base_dir.c_str()))
    {
        throw std::runtime_error(warn + err);
    }

    std::cout << "Loaded Shapes: " << shapes.size() << ", Materials: " << materials.size() << ", Vertices: " << (attrib.vertices.size() / 3) << std::endl;

    // 모든 머티리얼 텍스처 로드
    loadMaterials(base_dir, materials);

    m_vertices.clear();
    m_indices.clear();
    m_subMeshes.clear();

    for (const auto &shape : shapes)
    {
        SubMesh subMesh{};
        subMesh.indexStart = static_cast<uint32_t>(m_indices.size());

        int matId = -1;
        if (!shape.mesh.material_ids.empty())
        {
            matId = shape.mesh.material_ids[0];
        }
        subMesh.materialIndex = matId;

        for (const auto &index : shape.mesh.indices)
        {
            Vertex vertex{};
            vertex.pos = {
                static_cast<float>(attrib.vertices[3 * index.vertex_index + 0]),
                static_cast<float>(attrib.vertices[3 * index.vertex_index + 1]),
                static_cast<float>(attrib.vertices[3 * index.vertex_index + 2])};

            if (index.normal_index >= 0)
            {
                vertex.normal = {
                    static_cast<float>(attrib.normals[3 * index.normal_index + 0]),
                    static_cast<float>(attrib.normals[3 * index.normal_index + 1]),
                    static_cast<float>(attrib.normals[3 * index.normal_index + 2])};
            }

            if (index.texcoord_index >= 0)
            {
                vertex.texCoord = {
                    static_cast<float>(attrib.texcoords[2 * index.texcoord_index + 0]),
                    static_cast<float>(1.0 - attrib.texcoords[2 * index.texcoord_index + 1])};
            }

            m_vertices.push_back(vertex);
            m_indices.push_back(static_cast<uint32_t>(m_indices.size()));
        }

        subMesh.indexCount = static_cast<uint32_t>(m_indices.size()) - subMesh.indexStart;
        if (subMesh.indexCount > 0)
        {
            m_subMeshes.push_back(subMesh);
        }
    }
}

void Model::createVertexBuffer()
{
    VkDeviceSize bufferSize = sizeof(m_vertices[0]) * m_vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtils::CreateBuffer(m_context, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(m_context->GetDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, m_vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(m_context->GetDevice(), stagingBufferMemory);

    VulkanUtils::CreateBuffer(m_context, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vertexBuffer, m_vertexBufferMemory);
    VulkanUtils::CopyBuffer(m_context, stagingBuffer, m_vertexBuffer, bufferSize);

    vkDestroyBuffer(m_context->GetDevice(), stagingBuffer, nullptr);
    vkFreeMemory(m_context->GetDevice(), stagingBufferMemory, nullptr);
}

void Model::createIndexBuffer()
{
    VkDeviceSize bufferSize = sizeof(m_indices[0]) * m_indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtils::CreateBuffer(m_context, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(m_context->GetDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, m_indices.data(), (size_t)bufferSize);
    vkUnmapMemory(m_context->GetDevice(), stagingBufferMemory);

    VulkanUtils::CreateBuffer(m_context, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_indexBuffer, m_indexBufferMemory);
    VulkanUtils::CopyBuffer(m_context, stagingBuffer, m_indexBuffer, bufferSize);

    vkDestroyBuffer(m_context->GetDevice(), stagingBuffer, nullptr);
    vkFreeMemory(m_context->GetDevice(), stagingBufferMemory, nullptr);
}

void Model::createTextureSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(m_context->GetPhysicalDevice(), &properties);
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(m_context->GetDevice(), &samplerInfo, nullptr, &m_textureSampler) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

void Model::Draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout)
{
    VkBuffer vertexBuffers[] = {m_vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    if (!m_subMeshes.empty())
    {
        for (const auto &subMesh : m_subMeshes)
        {
            vkCmdDrawIndexed(commandBuffer, subMesh.indexCount, 1, subMesh.indexStart, 0, 0);
        }
    }
    else
    {
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(m_indices.size()), 1, 0, 0, 0);
    }
}

void Model::loadMaterials(const std::string &baseDir, const std::vector<tinyobj::material_t> &materials)
{
    m_materials.resize(materials.size());

    for (size_t i = 0; i < materials.size(); i++)
    {
        const auto &mat = materials[i];
        std::string texturePath = "";

        if (!mat.diffuse_texname.empty())
        {
            texturePath = baseDir + mat.diffuse_texname;
            for (auto &c : texturePath)
            {
                if (c == '\\')
                    c = '/';
            }

            if (!std::filesystem::exists(texturePath))
            {
                std::string altPath = texturePath;
                size_t pos = altPath.find("textures/");
                if (pos != std::string::npos)
                {
                    altPath.replace(pos, 9, "texture/");
                    if (std::filesystem::exists(altPath))
                    {
                        texturePath = altPath;
                    }
                }
                else
                {
                    pos = altPath.find("texture/");
                    if (pos != std::string::npos)
                    {
                        altPath.replace(pos, 8, "textures/");
                        if (std::filesystem::exists(altPath))
                        {
                            texturePath = altPath;
                        }
                    }
                }
            }
        }

        bool loaded = false;
        if (!texturePath.empty())
        {
            try
            {
                loadTexture(texturePath, m_materials[i].diffuseImage, m_materials[i].diffuseImageMemory);
                m_materials[i].diffuseImageView = VulkanUtils::CreateImageView(m_context, m_materials[i].diffuseImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
                loaded = true;
                std::cout << "Loaded material texture [" << i << "]: " << texturePath << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Failed to load texture " << texturePath << ": " << e.what() << ". Using fallback white texture." << std::endl;
            }
        }

        if (!loaded)
        {
            createFallbackTexture(m_materials[i].diffuseImage, m_materials[i].diffuseImageMemory, m_materials[i].diffuseImageView);
        }
    }

    if (m_materials.empty())
    {
        Material fallback{};
        createFallbackTexture(fallback.diffuseImage, fallback.diffuseImageMemory, fallback.diffuseImageView);
        m_materials.push_back(fallback);
    }
}

void Model::loadTexture(const std::string &filename, VkImage &textureImage, VkDeviceMemory &textureImageMemory)
{
    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load(filename.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels)
    {
        throw std::runtime_error("failed to load texture image!");
    }

    VkDeviceSize imageSize = texWidth * texHeight * 4;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtils::CreateBuffer(m_context, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(m_context->GetDevice(), stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(m_context->GetDevice(), stagingBufferMemory);

    stbi_image_free(pixels);

    VulkanUtils::CreateImage(m_context, texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    transitionImageLayout(m_context, textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

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
    region.imageExtent = {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1};

    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VulkanUtils::EndSingleTimeCommands(m_context, commandBuffer);

    transitionImageLayout(m_context, textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(m_context->GetDevice(), stagingBuffer, nullptr);
    vkFreeMemory(m_context->GetDevice(), stagingBufferMemory, nullptr);
}

void Model::createFallbackTexture(VkImage &textureImage, VkDeviceMemory &textureImageMemory, VkImageView &textureImageView)
{
    uint32_t whitePixel = 0xFFFFFFFF;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtils::CreateBuffer(m_context, 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(m_context->GetDevice(), stagingBufferMemory, 0, 4, 0, &data);
    memcpy(data, &whitePixel, 4);
    vkUnmapMemory(m_context->GetDevice(), stagingBufferMemory);

    VulkanUtils::CreateImage(m_context, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    transitionImageLayout(m_context, textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

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

    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VulkanUtils::EndSingleTimeCommands(m_context, commandBuffer);

    transitionImageLayout(m_context, textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(m_context->GetDevice(), stagingBuffer, nullptr);
    vkFreeMemory(m_context->GetDevice(), stagingBufferMemory, nullptr);

    textureImageView = VulkanUtils::CreateImageView(m_context, textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
}

void Model::transitionImageLayout(VulkanContext *context, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
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
