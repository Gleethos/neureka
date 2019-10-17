/*
 * JOCL - Java bindings for OpenCL
 *
 * Copyright 2009 Marco Hutter - http://www.jocl.org/
 */

// A very simple OpenCL kernel for computing the mandelbrot set
//
// output        : A buffer with sizeX*sizeY elements, storing
//                 the colors as RGB ints
// sizeX, sizeX  : The width and height of the buffer
// x0,y0,x1,y1   : The rectangle in which the mandelbrot
//                 set will be computed
// maxIterations : The maximum number of iterations
// colorMap      : A buffer with colorMapSize elements,
//                 containing the pixel colors

__kernel void computeMandelbrot(
    __global uint *output,
    int sizeX, int sizeY,
    float x0, float y0,
    float x1, float y1,
    int maxIterations,
    __global uint *colorMap,
    int colorMapSize
    )
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);

    float r = x0 + ix * (x1-x0) / sizeX;
    float i = y0 + iy * (y1-y0) / sizeY;

    float x = 0;
    float y = 0;

    float magnitudeSquared = 0;
    int iteration = 0;
    while (iteration<maxIterations && magnitudeSquared<4)
    {
        float xx = x*x;
        float yy = y*y;
        y = 2*x*y+i;
        x = xx-yy+r;
        magnitudeSquared=xx+yy;
        iteration++;
    }
    if (iteration == maxIterations)
    {
        output[iy*sizeX+ix] = 0;
    }
    else
    {
        float alpha = (float)iteration/maxIterations;
        int colorIndex = (int)(alpha * colorMapSize);
        output[iy*sizeX+ix] = colorMap[colorIndex];
	}
}