#pragma once
//#include<vulkan/vulkan.h>
class IWindowCore
{
public:
	virtual void Initialize() = 0;
	virtual bool WindowSouldClose() = 0;
	virtual void SwapBufferCore() = 0;
	virtual void MainLoop() = 0;
//	virtual VkSurfaceKHR& SurfaceWindow(const VkInstance& vkInstance) = 0;
	virtual ~IWindowCore() = default;
};