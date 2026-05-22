#version 450

layout(location = 0) in vec2 fragTexCoord;

layout(binding = 0) uniform sampler2D positionSampler;
layout(binding = 1) uniform sampler2D normalSampler;
layout(binding = 2) uniform sampler2D albedoSampler;
layout(binding = 3) uniform sampler2D shadowMapSampler;

layout(binding = 4) uniform LightBufferObject
{
    vec4 lightDirAndAmbient;     // xyz = 조명 방향, w = 주변광 강도
    vec4 viewPosAndSpecular;     // xyz = 시점 위치, w = 정반사 강도
    vec4 lightColorAndSpecPower; // xyz = 조명 색상, w = 정반사 지수
    mat4 lightSpaceMatrix;
}
light;

layout(location = 0) out vec4 outColor;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // Vulkan Z는 0에서 1이고, Y는 아래쪽이다.
    projCoords.xy = projCoords.xy * 0.5 + 0.5;

    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z > 1.0 || projCoords.z < 0.0)
    {
        return 0.0;
    }

    float currentDepth = projCoords.z;
    float bias = max(0.0015 * (1.0 - dot(normalize(normal), normalize(lightDir))), 0.0004);
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMapSampler, 0));
    float shadow = 0.0;

    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float closestDepth = texture(shadowMapSampler, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > closestDepth ? 1.0 : 0.0;
        }
    }

    return shadow / 9.0;
}

void main()
{
    vec3 fragPos = texture(positionSampler, fragTexCoord).xyz;
    vec3 normal = texture(normalSampler, fragTexCoord).xyz;
    vec3 albedo = texture(albedoSampler, fragTexCoord).rgb;

    vec3 lightDirVec = light.lightDirAndAmbient.xyz;
    float ambientStrength = light.lightDirAndAmbient.w;

    vec3 viewPosVec = light.viewPosAndSpecular.xyz;
    float specularStrength = light.viewPosAndSpecular.w;

    vec3 lightColor = light.lightColorAndSpecPower.xyz;
    float specularPower = light.lightColorAndSpecPower.w;

    // 주변광
    vec3 ambient = ambientStrength * albedo * lightColor;

    // 난반사
    vec3 lightDir = normalize(-lightDirVec);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * albedo * lightColor;

    // 정반사
    vec3 viewDir = normalize(viewPosVec - fragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), specularPower);
    vec3 specular = lightColor * specularStrength * spec;

    // 그림자
    vec4 fragPosLightSpace = light.lightSpaceMatrix * vec4(fragPos, 1.0);
    float shadow = ShadowCalculation(fragPosLightSpace, normal, lightDir);

    vec3 lighting = ambient + (1.0 - shadow) * (diffuse + specular);
    outColor = vec4(lighting, 1.0);
}
