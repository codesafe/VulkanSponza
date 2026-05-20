@echo off
set GLSLC="D:\private\vulkansdk\Bin\glslc.exe"

%GLSLC% shaders\geometry.vert -o shaders\geometry.vert.spv
%GLSLC% shaders\geometry.frag -o shaders\geometry.frag.spv

%GLSLC% shaders\composition.vert -o shaders\composition.vert.spv
%GLSLC% shaders\composition.frag -o shaders\composition.frag.spv

%GLSLC% shaders\shadow.vert -o shaders\shadow.vert.spv
%GLSLC% shaders\shadow.frag -o shaders\shadow.frag.spv

echo Shaders compiled.
