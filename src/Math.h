#pragma once
#include <algorithm>
#include <cmath>

namespace Math
{

struct Vec2
{
    float x, y;
    Vec2()
        : x(0), y(0)
    {
    }
    Vec2(float v)
        : x(v), y(v)
    {
    }
    Vec2(float x, float y)
        : x(x), y(y)
    {
    }
};

struct Vec3
{
    float x, y, z;

    Vec3()
        : x(0), y(0), z(0)
    {
    }
    Vec3(float v)
        : x(v), y(v), z(v)
    {
    }
    Vec3(float x, float y, float z)
        : x(x), y(y), z(z)
    {
    }

    Vec3 operator+(const Vec3 &o) const
    {
        return {x + o.x, y + o.y, z + o.z};
    }
    Vec3 operator-(const Vec3 &o) const
    {
        return {x - o.x, y - o.y, z - o.z};
    }
    Vec3 operator*(float s) const
    {
        return {x * s, y * s, z * s};
    }
    Vec3 &operator+=(const Vec3 &o)
    {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }
    Vec3 &operator-=(const Vec3 &o)
    {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
    }

    static float Dot(const Vec3 &a, const Vec3 &b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static Vec3 Cross(const Vec3 &a, const Vec3 &b)
    {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
    }

    Vec3 Normalize() const
    {
        float len = std::sqrt(x * x + y * y + z * z);
        if (len > 0.00001f)
        {
            return {x / len, y / len, z / len};
        }
        return *this;
    }
};

struct Vec4
{
    float x, y, z, w;
    Vec4()
        : x(0), y(0), z(0), w(0)
    {
    }
    Vec4(float x, float y, float z, float w)
        : x(x), y(y), z(z), w(w)
    {
    }
};

struct Mat4
{
    float m[4][4] = {0}; // 열 우선 형식

    Mat4()
    {
        m[0][0] = 1;
        m[1][1] = 1;
        m[2][2] = 1;
        m[3][3] = 1;
    }

    Mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33)
    {
        m[0][0] = m00;
        m[0][1] = m01;
        m[0][2] = m02;
        m[0][3] = m03;
        m[1][0] = m10;
        m[1][1] = m11;
        m[1][2] = m12;
        m[1][3] = m13;
        m[2][0] = m20;
        m[2][1] = m21;
        m[2][2] = m22;
        m[2][3] = m23;
        m[3][0] = m30;
        m[3][1] = m31;
        m[3][2] = m32;
        m[3][3] = m33;
    }

    static Mat4 Identity()
    {
        return Mat4();
    }

    Mat4 operator*(const Mat4 &o) const
    {
        Mat4 res;
        res.m[0][0] = 0;
        res.m[1][1] = 0;
        res.m[2][2] = 0;
        res.m[3][3] = 0; // 단위 행렬 값 제거
        for (int c = 0; c < 4; ++c)
        {
            for (int r = 0; r < 4; ++r)
            {
                res.m[c][r] = m[0][r] * o.m[c][0] +
                              m[1][r] * o.m[c][1] +
                              m[2][r] * o.m[c][2] +
                              m[3][r] * o.m[c][3];
            }
        }
        return res;
    }

    static Mat4 Translate(const Vec3 &t)
    {
        Mat4 res;
        res.m[3][0] = t.x;
        res.m[3][1] = t.y;
        res.m[3][2] = t.z;
        return res;
    }

    static Mat4 Scale(const Vec3 &s)
    {
        Mat4 res;
        res.m[0][0] = s.x;
        res.m[1][1] = s.y;
        res.m[2][2] = s.z;
        return res;
    }

    static Mat4 RotateY(float angle)
    {
        Mat4 res;
        float c = std::cos(angle);
        float s = std::sin(angle);
        res.m[0][0] = c;
        res.m[0][2] = -s;
        res.m[2][0] = s;
        res.m[2][2] = c;
        return res;
    }

    static Mat4 LookAt(const Vec3 &eye, const Vec3 &center, const Vec3 &up)
    {
        Vec3 f = (center - eye).Normalize();
        Vec3 s = Vec3::Cross(f, up).Normalize();
        Vec3 u = Vec3::Cross(s, f);

        Mat4 res;
        res.m[0][0] = s.x;
        res.m[1][0] = s.y;
        res.m[2][0] = s.z;
        res.m[0][1] = u.x;
        res.m[1][1] = u.y;
        res.m[2][1] = u.z;
        res.m[0][2] = -f.x;
        res.m[1][2] = -f.y;
        res.m[2][2] = -f.z;
        res.m[3][0] = -Vec3::Dot(s, eye);
        res.m[3][1] = -Vec3::Dot(u, eye);
        res.m[3][2] = Vec3::Dot(f, eye);
        return res;
    }

    static Mat4 Perspective(float fovY, float aspect, float zNear, float zFar)
    {
        float tanHalfFov = std::tan(fovY / 2.0f);
        Mat4 res;
        res.m[0][0] = 0;
        res.m[1][1] = 0;
        res.m[2][2] = 0;
        res.m[3][3] = 0;
        res.m[0][0] = 1.0f / (aspect * tanHalfFov);
        res.m[1][1] = 1.0f / tanHalfFov; // Vulkan은 아래쪽이 Y 양수이므로 여기나 뷰포트에서 Y를 반전할 수 있다.
        res.m[2][2] = zFar / (zNear - zFar);
        res.m[2][3] = -1.0f;
        res.m[3][2] = -(zFar * zNear) / (zFar - zNear);

        // Vulkan 클립 공간은 Y가 반전되고 Z 범위가 절반이므로 보정을 적용한다.
        res.m[1][1] *= -1.0f;
        return res;
    }

    static Mat4 Ortho(float left, float right, float bottom, float top, float zNear, float zFar)
    {
        Mat4 res;
        res.m[0][0] = 2.0f / (right - left);
        res.m[1][1] = 2.0f / (bottom - top); // 반전된 Y
        res.m[2][2] = 1.0f / (zNear - zFar);
        res.m[3][0] = -(right + left) / (right - left);
        res.m[3][1] = -(bottom + top) / (bottom - top);
        res.m[3][2] = zNear / (zNear - zFar);
        return res;
    }
};
} // Math 네임스페이스 끝
