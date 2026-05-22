#pragma once
#include "Math.h"
#include <fstream>
#include <sstream>
#include <string>

struct RenderConfig
{
    float cameraSpeed = 500.0f;
    float cameraSensitivity = 0.1f;
    Math::Vec3 cameraPos = Math::Vec3(0.0f, 200.0f, 200.0f);
    float cameraYaw = -90.0f;
    float cameraPitch = 0.0f;

    Math::Vec3 lightDir = Math::Vec3(0.5f, -1.0f, 0.5f).Normalize();
    bool autoRotateLight = false;

    Math::Vec3 lightColor = Math::Vec3(1.0f, 1.0f, 1.0f);
    float ambientStrength = 0.1f;
    float specularStrength = 0.5f;
    float specularPower = 32.0f;
    float hdrExposure = 1.0f;
    float bloomStrength = 0.35f;
    float bloomThreshold = 1.0f;
    float bloomSoftKnee = 1.0f;

    void Load(const std::string &path)
    {
        std::ifstream f(path);
        if (!f.is_open())
            return;
        std::string line;
        while (std::getline(f, line))
        {
            // '#'으로 시작하는 주석 제거
            size_t comment = line.find('#');
            if (comment != std::string::npos)
            {
                line = line.substr(0, comment);
            }

            std::stringstream ss(line);
            std::string key, eq;
            if (ss >> key >> eq)
            {
                if (eq == "=")
                {
                    if (key == "camera_speed")
                        ss >> cameraSpeed;
                    else if (key == "camera_sensitivity")
                        ss >> cameraSensitivity;
                    else if (key == "camera_pos_x")
                        ss >> cameraPos.x;
                    else if (key == "camera_pos_y")
                        ss >> cameraPos.y;
                    else if (key == "camera_pos_z")
                        ss >> cameraPos.z;
                    else if (key == "camera_yaw")
                        ss >> cameraYaw;
                    else if (key == "camera_pitch")
                        ss >> cameraPitch;
                    else if (key == "light_dir_x")
                        ss >> lightDir.x;
                    else if (key == "light_dir_y")
                        ss >> lightDir.y;
                    else if (key == "light_dir_z")
                        ss >> lightDir.z;
                    else if (key == "auto_rotate_light")
                        ss >> autoRotateLight;
                    else if (key == "light_color_r")
                        ss >> lightColor.x;
                    else if (key == "light_color_g")
                        ss >> lightColor.y;
                    else if (key == "light_color_b")
                        ss >> lightColor.z;
                    else if (key == "light_ambient")
                        ss >> ambientStrength;
                    else if (key == "light_specular_strength")
                        ss >> specularStrength;
                    else if (key == "light_specular_power")
                        ss >> specularPower;
                    else if (key == "hdr_exposure")
                        ss >> hdrExposure;
                    else if (key == "bloom_strength")
                        ss >> bloomStrength;
                    else if (key == "bloom_threshold")
                        ss >> bloomThreshold;
                    else if (key == "bloom_soft_knee")
                        ss >> bloomSoftKnee;
                }
            }
        }
        lightDir = lightDir.Normalize();
    }

    void Save(const std::string &path) const
    {
        std::ofstream f(path);
        if (!f.is_open())
            return;
        f << "# Sponza Deferred Renderer Configuration\n\n";
        f << "camera_speed = " << cameraSpeed << "\n";
        f << "camera_sensitivity = " << cameraSensitivity << "\n";
        f << "camera_pos_x = " << cameraPos.x << "\n";
        f << "camera_pos_y = " << cameraPos.y << "\n";
        f << "camera_pos_z = " << cameraPos.z << "\n";
        f << "camera_yaw = " << cameraYaw << "\n";
        f << "camera_pitch = " << cameraPitch << "\n";
        f << "light_dir_x = " << lightDir.x << "\n";
        f << "light_dir_y = " << lightDir.y << "\n";
        f << "light_dir_z = " << lightDir.z << "\n";
        f << "auto_rotate_light = " << (autoRotateLight ? 1 : 0) << "\n";
        f << "light_color_r = " << lightColor.x << "\n";
        f << "light_color_g = " << lightColor.y << "\n";
        f << "light_color_b = " << lightColor.z << "\n";
        f << "light_ambient = " << ambientStrength << "\n";
        f << "light_specular_strength = " << specularStrength << "\n";
        f << "light_specular_power = " << specularPower << "\n";
        f << "hdr_exposure = " << hdrExposure << "\n";
        f << "bloom_strength = " << bloomStrength << "\n";
        f << "bloom_threshold = " << bloomThreshold << "\n";
        f << "bloom_soft_knee = " << bloomSoftKnee << "\n";
    }
};
