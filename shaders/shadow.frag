#version 450

layout(location = 0) in vec2 fragTexCoord;

layout(binding = 2) uniform sampler2D alphaSampler;

void main()
{
    if (texture(alphaSampler, fragTexCoord).r < 0.5)
    {
        discard;
    }

    // 깊이 값은 깊이 버퍼에 자동으로 기록된다.
}
