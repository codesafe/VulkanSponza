#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragTangent;
layout(location = 4) in vec3 fragBitangent;

layout(binding = 1) uniform sampler2D diffuseSampler;
layout(binding = 2) uniform sampler2D normalSampler;
layout(binding = 3) uniform sampler2D alphaSampler;
layout(binding = 4) uniform sampler2D specularSampler;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outSpecular;

void main()
{
    if (texture(alphaSampler, fragTexCoord).r < 0.5)
    {
        discard;
    }

    vec3 vertexNormal = normalize(fragNormal);
    vec3 tangent = normalize(fragTangent - vertexNormal * dot(vertexNormal, fragTangent));
    float handedness = dot(cross(vertexNormal, tangent), fragBitangent) < 0.0 ? -1.0 : 1.0;
    vec3 bitangent = normalize(cross(vertexNormal, tangent)) * handedness;
    vec3 tangentNormal = texture(normalSampler, fragTexCoord).xyz * 2.0 - 1.0;
    vec3 finalNormal = normalize(mat3(tangent, bitangent, vertexNormal) * tangentNormal);

    outPosition = vec4(fragPos, 1.0);
    outNormal = vec4(finalNormal, 1.0);
    outAlbedo = texture(diffuseSampler, fragTexCoord);
    outSpecular = texture(specularSampler, fragTexCoord);
}
