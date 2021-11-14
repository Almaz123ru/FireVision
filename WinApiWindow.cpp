#include"WinApiWindow.h"

WinApiWindow::WinApiWindow(UINT BrushColor, LPCWSTR ClassName, LPCWSTR WindowName, int Width, int Height,
	LRESULT(__stdcall* WndProc)(HWND, UINT, WPARAM, LPARAM), int postion_x, int position_y, HINSTANCE hInstance, UINT Style, DWORD WindowStyle, HWND WndChild, HMENU hMenu)
{
	this->BrushColor = BrushColor;
	this->WndProc = WndProc;
	this->hInst = hInstance;
	this->Style = Style;
	this->ClassName = ClassName;
	this->WindowName = WindowName;
	this->WindowStyle = WindowStyle;
	this->postion_x = postion_x;
	this->postion_y = position_y;
	this->Width = Width;
	this->Height = Height;
	this->WndChild = WndChild;
	this->hMenu = hMenu;
	wcex = new WNDCLASSEX();
	msg = new MSG();
	Quit = true;
}

void WinApiWindow::Initialize()
{
	WindowClassEx();
	WindowCreate();
}

void WinApiWindow::Destroy()
{
	if (hWnd != NULL)
		DestroyWindow(hWnd);
}

void WinApiWindow::SwapBufferCore()
{
	if (PeekMessage(msg, NULL, 0, 0, PM_REMOVE))
	{
		TranslateMessage(msg);
		DispatchMessage(msg);
		if (msg->message == WM_QUIT)
			Quit = false;
	}
}

/*VkSurfaceKHR& WinApiWindow::SurfaceWindow(const VkInstance& vkInstance)
{
	VkWin32SurfaceCreateInfoKHR surfaceInfoKHR{};
	surfaceInfoKHR.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
	surfaceInfoKHR.pNext = nullptr;
	surfaceInfoKHR.flags = 0;
	surfaceInfoKHR.hinstance = hInst;
	surfaceInfoKHR.hwnd = hWnd;
	
	PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(vkInstance,
		"vkCreateWin32SurfaceKHR");
	VkSurfaceKHR vkSurface;
	if (vkCreateWin32SurfaceKHR(vkInstance, &surfaceInfoKHR, nullptr, &vkSurface) != VK_SUCCESS)
		std::runtime_error("failend initialize surface");
	// TODO: вставьте здесь оператор return
	return vkSurface;
}*/

bool WinApiWindow::WindowSouldClose()
{
	return Quit;
}

void WinApiWindow::WindowClassEx()
{
	UnregisterClass(ClassName, hInst);
	wcex->cbSize = sizeof(WNDCLASSEX);
	wcex->cbClsExtra = 0;
	wcex->cbWndExtra = 0;
	wcex->hbrBackground = reinterpret_cast<HBRUSH>(GetStockObject(BrushColor));
	wcex->lpfnWndProc = WndProc;
	wcex->hInstance = hInst;
	wcex->style = Style;
	wcex->hCursor = LoadCursor(NULL, IDC_ARROW);
	wcex->hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wcex->hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wcex->lpszClassName = ClassName;
	wcex->lpszMenuName = MenuName = L"Menu name";

	if (!RegisterClassEx(wcex))
		throw std::runtime_error("Error Register Class Ex!");
}

void WinApiWindow::WindowCreate()
{
	hWnd = nullptr;
	hWnd = CreateWindow(ClassName, WindowName, WindowStyle, postion_x, postion_y, Width,
		Height, WndChild, hMenu, hInst, NULL);

	if (!hWnd)
		throw std::runtime_error("Error Create Window");

	ShowWindow(hWnd, SW_SHOWNORMAL);
	UpdateWindow(hWnd);
}

void WinApiWindow::MainLoop()
{
	while (true) {
		if (PeekMessage(msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(msg);
			DispatchMessage(msg);
			if (msg->message == WM_QUIT)
			{
				break;
			}
		}

	}
}
