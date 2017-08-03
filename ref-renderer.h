#ifndef REFRENDERER_H
#define REFRENDERER_H

#include "renderer.h"

class RefRenderer : public Renderer {

private:

	Image*		image;
	SceneName	sceneName;

	int		numCircles;
	float*	position;
	float*	velocity;
	float*	color;
	float*	radius;

public:

	RefRenderer();
	virtual ~RefRenderer();

	const Image* getImage();

	void setup() {};

	void loadScene(SceneName name);

	void allocOutputImage(int width, int height);

	void clearImage();

	void advanceAnimation();

	void render();

	void shadePixel(
		int circleIndex,
		float pixelCenterX, float pixelCenterY,
		float px, float py, float pz,
		float* pixelData);
};

#endif