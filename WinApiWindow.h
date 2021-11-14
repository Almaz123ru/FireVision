#pragma once
#include<iostream>
#include<Windows.h>
//#define VK_USE_PLATFORM_WIN32_KHR
#include"IWinApiWindow.h"
class WinApiWindow : public IWindowCore
{
public:
	WinApiWindow(UINT BrushColor, LPCWSTR ClassName, LPCWSTR WindowName, int Width, int Height, LRESULT(WINAPI* WndProc)(HWND, UINT, WPARAM, LPARAM) = DefWindowProc, int x = CW_USEDEFAULT, int y = 0,
		HINSTANCE hInst = GetModuleHandle(nullptr), UINT Style = CS_VREDRAW | CS_HREDRAW, DWORD WindowStyle = WS_OVERLAPPEDWINDOW, HWND hParent = NULL, HMENU hMenu = NULL);

	void Initialize();
	void Destroy();
	void SwapBufferCore();
//	VkSurfaceKHR& SurfaceWindow(const VkInstance& vkInstance);
	bool WindowSouldClose();
	void MainLoop();
protected:
	HINSTANCE hInst;
	HWND hWnd;
	UINT BrushColor;
	LRESULT(WINAPI* WndProc)(HWND, UINT, WPARAM, LPARAM);
	UINT Style;
	LPCWSTR ClassName;
	LPCWSTR WindowName;
	LPCWSTR MenuName;
	DWORD WindowStyle;
	int postion_x;
	int postion_y;
	int Width;
	int Height;
	HWND WndChild;
	HMENU hMenu;
	bool Quit;
	
	WNDCLASSEX* wcex;
	MSG* msg;
	
	void WindowClassEx();
	void WindowCreate();
};