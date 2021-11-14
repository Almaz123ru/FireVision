#include<iostream>
#include"WinApiWindow.h"
#include"DecoderAnimation.h"
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK WndVision(HWND, UINT, WPARAM, LPARAM);
#define BUTTON_OK 1
#define STATIC_POS_X 2
#define STATIC_POS_Y 3	
#define PATH_OPEN_DIRECTORY 4
#define EDIT_PATH_DIRECTORY 5
#define EXIT 6
IWindowCore* windowVision = nullptr;
IWindowCore* windowChildPath = nullptr;
bool flagVsiont = false;
int main()
{
	IWindowCore* windowCore = new WinApiWindow(WHITE_BRUSH, L"Main class", L"Eбашь монтаж", 1500, 1000, WndProc);
		windowCore->Initialize();
		windowCore->MainLoop();
	return 0;
}
HWND ButtonOk;
HWND EditPathDirectory;
HWND ClassificationNeuralNetworkStatus;
HWND ClassificationNeuralNetwork;
HWND StaticNamePos_x;
HWND StaticNamePos_y;
HWND StaticPos_x;
HWND StaticPos_y;
HWND VisionChild;
HMENU hMenuBar;
HMENU hFile;
HMENU hExit;
RECT clientRect;
HGDIOBJ oldPen;
DecoderAnimation decoderAnimation;
int t = 0;
HDC hDC; PAINTSTRUCT ps;
char textEditPath[300] = { 0 };

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wp, LPARAM lp)
{
	switch (uMsg)
	{
	case WM_CREATE:
	///	windowVision = new WinApiWindow(BLACK_BRUSH, L"vision class", L"", 512, 400, DefWindowProc, 0, 0, GetModuleHandle(nullptr),  CS_VREDRAW | CS_HREDRAW | CS_OWNDC, WS_CHILD | WS_VISIBLE | WS_BORDER,hwnd);
	//	windowVision->Initialize();
	//	ButtonOk = CreateWindow(L"button", L"ok", WS_VISIBLE | WS_CHILD | WS_BORDER, 150, 550, 50, 20, hwnd, reinterpret_cast<HMENU>(BUTTON_OK), GetModuleHandle(nullptr), nullptr);
		ClassificationNeuralNetwork = CreateWindow(L"static", L"Статус!", WS_VISIBLE | WS_CHILD | WS_BORDER, 150, 730, 150, 25, hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
		ClassificationNeuralNetworkStatus = CreateWindow(L"static", L"пожара нет", WS_VISIBLE | WS_CHILD | WS_BORDER, 150, 755, 150, 25, hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
		StaticNamePos_x = CreateWindow(L"static", L"Позиция х", WS_VISIBLE | WS_CHILD | WS_BORDER, 400, 730, 100, 25, hwnd, nullptr,GetModuleHandle(nullptr), nullptr);
		StaticNamePos_y = CreateWindow(L"static", L"Позиция y", WS_VISIBLE | WS_CHILD | WS_BORDER, 600, 730, 100, 25, hwnd, nullptr, GetModuleHandle(nullptr), nullptr);
		StaticPos_x = CreateWindow(L"static", L"", WS_VISIBLE | WS_CHILD | WS_BORDER, 400, 755, 100, 25, hwnd, reinterpret_cast<HMENU>(STATIC_POS_X), GetModuleHandle(nullptr), nullptr);
		StaticPos_y = CreateWindow(L"static", L"", WS_VISIBLE | WS_CHILD | WS_BORDER, 600, 755, 100, 25, hwnd, reinterpret_cast<HMENU>(STATIC_POS_Y), GetModuleHandle(nullptr), nullptr);
	
		hMenuBar = CreateMenu();
		hFile = CreateMenu();
		AppendMenu(hMenuBar, MF_POPUP, (UINT_PTR)hFile, L"файл");
		AppendMenu(hFile, MF_STRING, PATH_OPEN_DIRECTORY, L"Сохранение файла");
		AppendMenu(hFile, MF_STRING, EXIT, L"Выход из программы");
		SetMenu(hwnd, hMenuBar);
		SetTimer(hwnd, 1, 16, NULL);
		return 0;
	case WM_COMMAND:
		switch (wp)
		{
		case PATH_OPEN_DIRECTORY:
			windowChildPath = new WinApiWindow(WHITE_BRUSH, L"vision class", L"", 512, 400, WndVision, 0, 0, GetModuleHandle(nullptr), CS_VREDRAW | CS_HREDRAW | CS_OWNDC, WS_SYSMENU, hwnd);
			windowChildPath->Initialize();
			if (textEditPath != 0)
			{
			}
			break;
	
		case EXIT:
			PostQuitMessage(0);
			break;
		default:
			break;
		}
		return 0;
	case WM_PAINT:

		if (flagVsiont)
		{
			decoderAnimation.animation(t, hDC, hwnd, ps, 0, 0);
			
			t = t + 1;
		}

		else
		{
			GetClientRect(hwnd, &clientRect);
			hDC = BeginPaint(hwnd, &ps);
			oldPen = SelectObject(hDC, GetStockObject(BLACK_BRUSH));
			Rectangle(hDC, 0, 0, 1280, 720);
			SelectObject(hDC, oldPen);
			EndPaint(hwnd, &ps);
		}
		if (t > decoderAnimation.vectorHBitmap.size() - 1)

		{
			decoderAnimation.pathFile.clear();
			decoderAnimation.vectorHBitmap.clear();
			t = 0;
			flagVsiont = false;
		}
		return 0;
	case WM_TIMER:
		InvalidateRect(hwnd, NULL, false);
		return 0;

	case WM_DESTROY:
		KillTimer(hwnd, 101);
		PostQuitMessage(0);
		return 0;
	default:
		DefWindowProc(hwnd, uMsg, wp, lp);
	}
}

LRESULT CALLBACK WndVision(HWND hwnd, UINT uMsg, WPARAM wp, LPARAM lp) 
{
	switch (uMsg)
	{
	case WM_CREATE:
		EditPathDirectory = CreateWindow(L"edit", L"", WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL | ES_AUTOVSCROLL | ES_MULTILINE  | WS_VSCROLL 
			| WS_HSCROLL | WS_BORDER, 10, 10, 450, 250, hwnd, reinterpret_cast<HMENU>(EDIT_PATH_DIRECTORY), GetModuleHandle(nullptr), nullptr);
		ButtonOk = CreateWindow(L"button", L"ок", WS_VISIBLE | WS_CHILD | WS_BORDER, 50, 300, 50, 20, hwnd, reinterpret_cast<HMENU>(BUTTON_OK), GetModuleHandle(nullptr), nullptr);
		return 0;
	case WM_COMMAND:
		switch (wp)
		{
		case EDIT_PATH_DIRECTORY:
			
			break;
		case BUTTON_OK:
			GetWindowTextA(EditPathDirectory, textEditPath, 300);
			std::cout << textEditPath << std::endl;
			decoderAnimation.setPathDirectory(textEditPath);
			decoderAnimation.searchFile();
			decoderAnimation.initialize(1280, 720);
			flagVsiont = true;
			return 0;
		default:
			break;
		}
		return 0;
	default:
		return DefWindowProc(hwnd, uMsg, wp, lp);
	}
	}

	