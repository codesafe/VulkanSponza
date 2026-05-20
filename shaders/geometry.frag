#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;

// 바인딩 0은 UBO
layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;

void main()
{
    outPosition = vec4(fragPos, 1.0);
    outNormal = vec4(normalize(fragNormal), 1.0);
    outAlbedo = texture(texSampler, fragTexCoord);
}
