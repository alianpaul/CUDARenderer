#include "renderer.h"
#include "image.h"
#include "console.h"

#include <GLFW\glfw3.h>

#include "perf.h"

static struct {
	int width;
	int height;
	bool updateSim;
	bool printStats;
	bool pauseSim;
	double lastFrameTime;

	Renderer* renderer;

} gDisplay;

static GLFWwindow * window;

void display();
void renderPicture();

void startRendererWithDisplay(Renderer* renderer) 
{
	// setup the display

	const Image* img = renderer->getImage();

	gDisplay.renderer = renderer;
	gDisplay.updateSim = true;
	gDisplay.pauseSim = false;
	gDisplay.printStats = true;
	//gDisplay.lastFrameTime = CycleTimer::currentSeconds();
	gDisplay.width = img->width;
	gDisplay.height = img->height;

	// configure GLUT
	if (!glfwInit())
	{
		out_err("Counld not init GLFW");
		exit(-1);
	}

	window = glfwCreateWindow(gDisplay.width, gDisplay.height, 
							  "CUDA Circle Renderer", NULL, NULL);

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	while (!glfwWindowShouldClose(window))
	{
		display();
	}
}

void display()
{
	renderPicture();


	const Image* img = gDisplay.renderer->getImage();

	int width = gDisplay.width;
	int height = gDisplay.height;

	glDisable(GL_DEPTH_TEST);
	glClearColor(0.f, 0.f, 0.f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.f, gDisplay.width, 0.f, gDisplay.height, -1.f, 1.f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// copy image data from the renderer to the OpenGL
	// frame-buffer.  This is inefficient solution is the processing
	// to generate the image is done in CUDA.  An improved solution
	// would render to a CUDA surface object (stored in GPU memory),
	// and then bind this surface as a texture enabling it's use in
	// normal openGL rendering
	glRasterPos2i(0, 0);
	glDrawPixels(width, height, GL_RGBA, GL_FLOAT, img->data);

	glfwSwapBuffers(window);
	glfwPollEvents();
}

void renderPicture()
{
	QUERY_PERFORMANCE_ENTER;
	gDisplay.renderer->clearImage();
	QUERY_PERFORMANCE_EXIT(ClearImage);

	//update particle positions and state
	QUERY_PERFORMANCE_ENTER;
	if (gDisplay.updateSim) {
		gDisplay.renderer->advanceAnimation();
	}
	if (gDisplay.pauseSim)
		gDisplay.updateSim = false;
	QUERY_PERFORMANCE_EXIT(Animate);

	QUERY_PERFORMANCE_ENTER
	gDisplay.renderer->render();
	QUERY_PERFORMANCE_EXIT(Render);
}