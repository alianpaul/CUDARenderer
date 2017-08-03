#include "console.h"
#include "ref-renderer.h"

void startRendererWithDisplay(Renderer* renderer);


int main(int argc, char** argv)
{
	int imageSize = 720;
	SceneName scene = CIRCLE_RGBY;

	Renderer* renderer = new RefRenderer();
	renderer->allocOutputImage(imageSize, imageSize);
	renderer->loadScene(scene);
	renderer->setup();

	startRendererWithDisplay(renderer);
}