#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(binding = 0) uniform sampler2D hdrSampler;
layout(binding = 1) uniform HdrSettingsBufferObject
{
    vec4 params;
} hdrSettings;

layout(location = 0) out vec4 outColor;

vec3 TonemapACES(vec3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

vec3 LinearToSrgb(vec3 color)
{
    return pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));
}

void main()
{
    vec2 texelSize = 1.0 / vec2(textureSize(hdrSampler, 0));
    vec3 hdrColor = texture(hdrSampler, fragTexCoord).rgb;
    vec3 bloom = vec3(0.0);
    float weightSum = 0.0;
    float exposure = max(hdrSettings.params.x, 0.0);
    float bloomStrength = max(hdrSettings.params.y, 0.0);
    float bloomThreshold = max(hdrSettings.params.z, 0.0);
    float bloomSoftKnee = max(hdrSettings.params.w, 0.0001);

    for (int x = -4; x <= 4; ++x)
    {
        for (int y = -4; y <= 4; ++y)
        {
            vec2 offset = vec2(x, y) * texelSize * 2.0;
            vec3 sampleColor = texture(hdrSampler, fragTexCoord + offset).rgb;
            float brightness = max(max(sampleColor.r, sampleColor.g), sampleColor.b);
            vec3 brightColor = max(sampleColor - vec3(bloomThreshold), vec3(0.0));
            float weight = exp(-dot(vec2(x, y), vec2(x, y)) / 12.0);
            bloom += brightColor * weight * smoothstep(bloomThreshold, bloomThreshold + bloomSoftKnee, brightness);
            weightSum += weight;
        }
    }

    bloom /= max(weightSum, 0.0001);

    vec3 color = hdrColor * exposure + bloom * bloomStrength;
    color = TonemapACES(color);
    color = LinearToSrgb(color);
    outColor = vec4(color, 1.0);
}
