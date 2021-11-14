#ifndef DECODER_ANIMATION_H
#define DECODER_ANIMATION_H
#include"WinApiWindow.h"
#include<vector>
#include<string>
class DecoderAnimation
{

public:
	
	void setPathDirectory(const std::string& pathDirectory);
	
	void searchFile();
	void initialize(int width, int height);
	
	void animation(int t, HDC& hDC, HWND& hWnd, PAINTSTRUCT& ps, int width, int height);
	void DrawBitmap(HDC hDC, int x, int y, HBITMAP hBitmap);
	std::vector<std::string> pathFile;
	std::vector<HBITMAP> vectorHBitmap;
private:

	//std::vector<HBITMAP> vectorHBitmap;

	std::string pathDirectory;

};
#endif // !DECODER_ANIMATION_H
