#ifndef SCENE_LOAD_H
#define SCENE_LOAD_H

#include "renderer.h"

void
loadCircleScene(
	SceneName sceneName,
	int&	numCircles,
	float*& position,
	float*& velocity,
	float*& color,
	float*& radius);

#endif