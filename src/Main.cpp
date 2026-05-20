#include "Application.h"
#include <iostream>

int main()
{
    try
    {
        Application app(1280, 720, "Sponza Deferred Vulkan");
        app.Run();
    }
    catch (const std::exception &e)
    {
        MessageBoxA(NULL, e.what(), "Error", MB_ICONERROR);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
