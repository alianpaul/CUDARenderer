
#include "ref-renderer.h"
#include "cuda-renderer.h"

void startRendererWithDisplay(Renderer* renderer);

int main(int argc, char** argv)
{
	int imageSize = 1024;
	//SceneName scene = CIRCLE_RGB;
	//SceneName scene = CIRCLE_TEST_10K;
	//SceneName scene = CIRCLE_TEST_100K;
	//SceneName scene = PATTERN;
	//SceneName scene = SNOWFLAKES_SINGLE_FRAME;
	SceneName scene = HYPNOSIS;
	//SceneName scene = FIREWORKS;
	//SceneName scene = BOUNCING_BALLS;
	//SceneName scene = BIG_LITTLE;

	Renderer* renderer = new CudaRenderer();
	//Renderer* renderer = new RefRenderer();
	renderer->allocOutputImage(imageSize, imageSize);
	renderer->loadScene(scene);
	renderer->setup();

	startRendererWithDisplay(renderer);
}