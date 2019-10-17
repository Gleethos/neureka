/*
 * JOCL - Java bindings for OpenCL
 *
 * Copyright 2009 Marco Hutter - http://www.jocl.org/
 */

// A mandelbrot kernel using QuadFloat functions

inline int iterate(
    float2 x0, float2 y0,
    float2 dx, float2 dy,
    float relX, float relY,
    int maxIterations)
{
    float4 qx0 = qfAssign2(x0);
    float4 qy0 = qfAssign2(y0);
    float4 qdx = qfAssign2(dx);
    float4 qdy = qfAssign2(dy);

    float4 qr = qfAssign(0);
    float4 qi = qfAssign(0);

    float4 qx = qfAssign(0);
    float4 qy = qfAssign(0);

    float4 qxx = qfAssign(0);
    float4 qyy = qfAssign(0);

    float4 qfTemp = qfAssign(0);
    float4 magnitudeSquared = qfAssign(0);

    //float r = x0 + ((float)ix / sizeX) * dx;
    //float i = y0 + ((float)iy / sizeY) * dy;
    qfMulFloat(&qfTemp, qdx, relX);
    qfAdd(&qr, qx0, qfTemp);

    qfMulFloat(&qfTemp, qdy, relY);
    qfAdd(&qi, qy0, qfTemp);

    int iteration = 0;
    while (iteration<maxIterations && qfLessThan(&magnitudeSquared, 4))
    {

        // float xx = x*x;
        qfMul(&qxx, qx,qx);

        // float yy = y*y;
        qfMul(&qyy, qy,qy);

        //y = 2*x*y+i;
        qfMulFloat(&qfTemp, qx,2);
        qfMul(&qfTemp, qfTemp,qy);
        qfAdd(&qy, qfTemp,qi);

        //x = xx-yy+r;
        qfTemp.x = -qyy.x;
        qfTemp.y = -qyy.y;
        qfTemp.z = -qyy.z;
        qfTemp.w = -qyy.w;
        qfAdd(&qfTemp,qxx,qfTemp);
        qfAdd(&qx, qfTemp,qr);


        qfAdd(&magnitudeSquared, qxx, qyy);
        iteration++;
    }
    return iteration;

}



__kernel void computeMyMandelbrot(
    __global uint *output,
    int sizeX, int sizeY,
    int tileX, int tileY,
    int tileSizeX, int tileSizeY,
    float2 x0, float2 y0,
    float2 dx, float2 dy,
    int maxIterations)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);

    int indexX = ix + tileX * tileSizeX;
    int indexY = iy + tileY * tileSizeY;

    float relX = (float)indexX / sizeX;
    float relY = (float)indexY / sizeY;

    int iteration = iterate(x0, y0, dx, dy, relX, relY, maxIterations);
    output[mul24((int)iy, tileSizeX)+ix] = iteration;
}

