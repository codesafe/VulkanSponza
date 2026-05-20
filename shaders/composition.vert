#version 450

// 내장 정점 인덱스로 전체 화면 사각형 렌더링
layout(location = 0) out vec2 fragTexCoord;

void main()
{
    vec2 outTexCoords[3] = vec2[](
        vec2(0.0, 0.0),
        vec2(2.0, 0.0),
        vec2(0.0, 2.0));
    gl_Position = vec4(outTexCoords[gl_VertexIndex] * 2.0 - 1.0, 0.0, 1.0);
    fragTexCoord = outTexCoords[gl_VertexIndex];
}
