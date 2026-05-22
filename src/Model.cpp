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
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions(5);

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

    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[3].offset = offsetof(Vertex, tangent);

    attributeDescriptions[4].binding = 0;
    attributeDescriptions[4].location = 4;
    attributeDescriptions[4].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[4].offset = offsetof(Vertex, bitangent);

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
        vkDestroyImageView(m_context->GetDevice(), material.normalImageView, nullptr);
        vkDestroyImage(m_context->GetDevice(), material.normalImage, nullptr);
        vkFreeMemory(m_context->GetDevice(), material.normalImageMemory, nullptr);
        vkDestroyImageView(m_context->GetDevice(), material.alphaImageView, nullptr);
        vkDestroyImage(m_context->GetDevice(), material.alphaImage, nullptr);
        vkFreeMemory(m_context->GetDevice(), material.alphaImageMemory, nullptr);
        vkDestroyImageView(m_context->GetDevice(), material.specularImageView, nullptr);
        vkDestroyImage(m_context->GetDevice(), material.specularImage, nullptr);
        vkFreeMemory(m_context->GetDevice(), material.specularImageMemory, nullptr);
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
        subMesh.name = shape.name;
        subMesh.indexStart = static_cast<uint32_t>(m_indices.size());
        bool hasBounds = false;

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

            if (!hasBounds)
            {
                subMesh.boundsMin = vertex.pos;
                subMesh.boundsMax = vertex.pos;
                hasBounds = true;
            }
            else
            {
                subMesh.boundsMin.x = std::min(subMesh.boundsMin.x, vertex.pos.x);
                subMesh.boundsMin.y = std::min(subMesh.boundsMin.y, vertex.pos.y);
                subMesh.boundsMin.z = std::min(subMesh.boundsMin.z, vertex.pos.z);
                subMesh.boundsMax.x = std::max(subMesh.boundsMax.x, vertex.pos.x);
                subMesh.boundsMax.y = std::max(subMesh.boundsMax.y, vertex.pos.y);
                subMesh.boundsMax.z = std::max(subMesh.boundsMax.z, vertex.pos.z);
            }
        }

        subMesh.indexCount = static_cast<uint32_t>(m_indices.size()) - subMesh.indexStart;
        if (subMesh.indexCount > 0)
        {
            m_subMeshes.push_back(subMesh);
        }
    }

    calculateTangents();
}

void Model::calculateTangents()
{
    for (auto &vertex : m_vertices)
    {
        vertex.tangent = Math::Vec3(0.0f);
        vertex.bitangent = Math::Vec3(0.0f);
    }

    for (size_t i = 0; i + 2 < m_indices.size(); i += 3)
    {
        Vertex &v0 = m_vertices[m_indices[i]];
        Vertex &v1 = m_vertices[m_indices[i + 1]];
        Vertex &v2 = m_vertices[m_indices[i + 2]];

        Math::Vec3 edge1 = v1.pos - v0.pos;
        Math::Vec3 edge2 = v2.pos - v0.pos;
        float deltaU1 = v1.texCoord.x - v0.texCoord.x;
        float deltaV1 = v1.texCoord.y - v0.texCoord.y;
        float deltaU2 = v2.texCoord.x - v0.texCoord.x;
        float deltaV2 = v2.texCoord.y - v0.texCoord.y;
        float determinant = deltaU1 * deltaV2 - deltaU2 * deltaV1;

        if (std::abs(determinant) < 0.000001f)
        {
            continue;
        }

        float invDeterminant = 1.0f / determinant;
        Math::Vec3 tangent = (edge1 * deltaV2 - edge2 * deltaV1) * invDeterminant;
        Math::Vec3 bitangent = (edge2 * deltaU1 - edge1 * deltaU2) * invDeterminant;

        v0.tangent += tangent;
        v1.tangent += tangent;
        v2.tangent += tangent;
        v0.bitangent += bitangent;
        v1.bitangent += bitangent;
        v2.bitangent += bitangent;
    }

    for (auto &vertex : m_vertices)
    {
        Math::Vec3 normal = vertex.normal.Normalize();
        Math::Vec3 tangent = vertex.tangent - normal * Math::Vec3::Dot(normal, vertex.tangent);
        if (Math::Vec3::Dot(tangent, tangent) < 0.000001f)
        {
            tangent = Math::Vec3(1.0f, 0.0f, 0.0f);
        }

        tangent = tangent.Normalize();
        Math::Vec3 bitangent = Math::Vec3::Cross(normal, tangent);
        if (Math::Vec3::Dot(bitangent, vertex.bitangent) < 0.0f)
        {
            bitangent = bitangent * -1.0f;
        }

        vertex.tangent = tangent;
        vertex.bitangent = bitangent.Normalize();
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
        m_materials[i].name = mat.name.empty() ? ("Material " + std::to_string(i)) : mat.name;
        auto resolveTexturePath = [&](const std::string &textureName) -> std::string
        {
            if (textureName.empty())
            {
                return "";
            }

            std::string texturePath = baseDir + textureName;
            for (auto &c : texturePath)
            {
                if (c == '\\')
                {
                    c = '/';
                }
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

            return texturePath;
        };

        auto inferSpecularTexturePath = [&](const std::string &diffusePath) -> std::string
        {
            if (diffusePath.empty())
            {
                return "";
            }

            std::vector<std::string> candidates;
            auto addReplacementCandidate = [&](const std::string &token)
            {
                size_t pos = diffusePath.rfind(token);
                if (pos != std::string::npos)
                {
                    std::string candidate = diffusePath;
                    candidate.replace(pos, token.size(), "_spec");
                    candidates.push_back(candidate);
                }
            };

            addReplacementCandidate("_diff");
            addReplacementCandidate("_dif");

            size_t extensionPos = diffusePath.find_last_of('.');
            if (extensionPos != std::string::npos)
            {
                candidates.push_back(diffusePath.substr(0, extensionPos) + "_spec" + diffusePath.substr(extensionPos));
            }

            for (const auto &candidate : candidates)
            {
                if (std::filesystem::exists(candidate))
                {
                    return candidate;
                }
            }

            return "";
        };

        bool loaded = false;
        std::string texturePath = resolveTexturePath(mat.diffuse_texname);
        if (!texturePath.empty())
        {
            try
            {
                loadTexture(texturePath, VK_FORMAT_R8G8B8A8_SRGB, m_materials[i].diffuseImage, m_materials[i].diffuseImageMemory);
                m_materials[i].diffuseImageView = VulkanUtils::CreateImageView(m_context, m_materials[i].diffuseImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
                m_materials[i].diffuseTexturePath = texturePath;
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
            createFallbackTexture(0xFFFFFFFF, VK_FORMAT_R8G8B8A8_SRGB, m_materials[i].diffuseImage, m_materials[i].diffuseImageMemory, m_materials[i].diffuseImageView);
        }

        bool normalLoaded = false;
        std::string normalTexturePath = resolveTexturePath(!mat.normal_texname.empty() ? mat.normal_texname : (!mat.bump_texname.empty() ? mat.bump_texname : mat.displacement_texname));
        if (!normalTexturePath.empty())
        {
            try
            {
                loadTexture(normalTexturePath, VK_FORMAT_R8G8B8A8_UNORM, m_materials[i].normalImage, m_materials[i].normalImageMemory);
                m_materials[i].normalImageView = VulkanUtils::CreateImageView(m_context, m_materials[i].normalImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
                m_materials[i].normalTexturePath = normalTexturePath;
                normalLoaded = true;
                std::cout << "Loaded normal texture [" << i << "]: " << normalTexturePath << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Failed to load normal texture " << normalTexturePath << ": " << e.what() << ". Using fallback flat normal texture." << std::endl;
            }
        }

        if (!normalLoaded)
        {
            createFallbackTexture(0xFFFF8080, VK_FORMAT_R8G8B8A8_UNORM, m_materials[i].normalImage, m_materials[i].normalImageMemory, m_materials[i].normalImageView);
        }

        bool alphaLoaded = false;
        std::string alphaTexturePath = resolveTexturePath(mat.alpha_texname);
        if (!alphaTexturePath.empty())
        {
            try
            {
                loadTexture(alphaTexturePath, VK_FORMAT_R8G8B8A8_UNORM, m_materials[i].alphaImage, m_materials[i].alphaImageMemory);
                m_materials[i].alphaImageView = VulkanUtils::CreateImageView(m_context, m_materials[i].alphaImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
                m_materials[i].alphaTexturePath = alphaTexturePath;
                alphaLoaded = true;
                std::cout << "Loaded alpha texture [" << i << "]: " << alphaTexturePath << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Failed to load alpha texture " << alphaTexturePath << ": " << e.what() << ". Using fallback opaque texture." << std::endl;
            }
        }

        if (!alphaLoaded)
        {
            createFallbackTexture(0xFFFFFFFF, VK_FORMAT_R8G8B8A8_UNORM, m_materials[i].alphaImage, m_materials[i].alphaImageMemory, m_materials[i].alphaImageView);
        }

        bool specularLoaded = false;
        std::string specularTexturePath = resolveTexturePath(mat.specular_texname);
        if (specularTexturePath.empty())
        {
            specularTexturePath = inferSpecularTexturePath(texturePath);
        }

        if (!specularTexturePath.empty())
        {
            try
            {
                loadTexture(specularTexturePath, VK_FORMAT_R8G8B8A8_UNORM, m_materials[i].specularImage, m_materials[i].specularImageMemory);
                m_materials[i].specularImageView = VulkanUtils::CreateImageView(m_context, m_materials[i].specularImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
                m_materials[i].specularTexturePath = specularTexturePath;
                specularLoaded = true;
                std::cout << "Loaded specular texture [" << i << "]: " << specularTexturePath << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Failed to load specular texture " << specularTexturePath << ": " << e.what() << ". Using fallback white specular texture." << std::endl;
            }
        }

        if (!specularLoaded)
        {
            createFallbackTexture(0xFFFFFFFF, VK_FORMAT_R8G8B8A8_UNORM, m_materials[i].specularImage, m_materials[i].specularImageMemory, m_materials[i].specularImageView);
        }
    }

    if (m_materials.empty())
    {
        Material fallback{};
        fallback.name = "Fallback Material";
        createFallbackTexture(0xFFFFFFFF, VK_FORMAT_R8G8B8A8_SRGB, fallback.diffuseImage, fallback.diffuseImageMemory, fallback.diffuseImageView);
        createFallbackTexture(0xFFFF8080, VK_FORMAT_R8G8B8A8_UNORM, fallback.normalImage, fallback.normalImageMemory, fallback.normalImageView);
        createFallbackTexture(0xFFFFFFFF, VK_FORMAT_R8G8B8A8_UNORM, fallback.alphaImage, fallback.alphaImageMemory, fallback.alphaImageView);
        createFallbackTexture(0xFFFFFFFF, VK_FORMAT_R8G8B8A8_UNORM, fallback.specularImage, fallback.specularImageMemory, fallback.specularImageView);
        m_materials.push_back(fallback);
    }
}

void Model::loadTexture(const std::string &filename, VkFormat format, VkImage &textureImage, VkDeviceMemory &textureImageMemory)
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

    VulkanUtils::CreateImage(m_context, texWidth, texHeight, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    transitionImageLayout(m_context, textureImage, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

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

    transitionImageLayout(m_context, textureImage, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(m_context->GetDevice(), stagingBuffer, nullptr);
    vkFreeMemory(m_context->GetDevice(), stagingBufferMemory, nullptr);
}

void Model::createFallbackTexture(uint32_t pixel, VkFormat format, VkImage &textureImage, VkDeviceMemory &textureImageMemory, VkImageView &textureImageView)
{
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VulkanUtils::CreateBuffer(m_context, 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(m_context->GetDevice(), stagingBufferMemory, 0, 4, 0, &data);
    memcpy(data, &pixel, 4);
    vkUnmapMemory(m_context->GetDevice(), stagingBufferMemory);

    VulkanUtils::CreateImage(m_context, 1, 1, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    transitionImageLayout(m_context, textureImage, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

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

    transitionImageLayout(m_context, textureImage, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(m_context->GetDevice(), stagingBuffer, nullptr);
    vkFreeMemory(m_context->GetDevice(), stagingBufferMemory, nullptr);

    textureImageView = VulkanUtils::CreateImageView(m_context, textureImage, format, VK_IMAGE_ASPECT_COLOR_BIT);
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
