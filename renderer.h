#ifndef RENDERER_H
#define RENDERER_H

#include <algorithm>

#define NUM_FIREWORKS 15
#define NUM_SPARKS 20
#define CLAMP(x, minimum, maximum) std::max(minimum, std::min(x, maximum))

enum SceneName
{
	CIRCLE_RGB,
	CIRCLE_RGBY,
	CIRCLE_TEST_10K,
	CIRCLE_TEST_100K,
	PATTERN,
	SNOWFLAKES,
	FIREWORKS,
	HYPNOSIS,
	BOUNCING_BALLS,
	SNOWFLAKES_SINGLE_FRAME,
	BIG_LITTLE,
	LITTLE_BIG
};

struct Image;

class Renderer
{
public:
	virtual ~Renderer() { };

	virtual const Image* getImage() = 0;

	virtual void setup() = 0;

	virtual void loadScene(SceneName name) = 0;

	virtual void allocOutputImage(int width, int height) = 0;

	virtual void clearImage() = 0;

	virtual void advanceAnimation() = 0;

	virtual void render() = 0;
};

#endif