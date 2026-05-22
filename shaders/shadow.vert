#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;

layout(binding = 0) uniform UniformBufferObject
{
    mat4 model;
}
ubo;

layout(binding = 1) uniform LightBufferObject
{
    vec4 lightDirAndAmbient;
    vec4 viewPosAndSpecular;
    vec4 lightColorAndSpecPower;
    mat4 lightSpaceMatrix;
}
light;

void main()
{
    fragTexCoord = inTexCoord;
    gl_Position = light.lightSpaceMatrix * ubo.model * vec4(inPosition, 1.0);
}
