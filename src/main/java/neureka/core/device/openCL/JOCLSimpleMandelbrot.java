package neureka.core.device.openCL;
/*
 * JOCL - Java bindings for OpenCL
 *
 * Copyright 2009 Marco Hutter - http://www.jocl.org/
 */

import static org.jocl.CL.*;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import java.io.*;

import javax.swing.*;

import org.jocl.*;

/**
 * A class that uses a simple OpenCL kernel to compute the
 * Mandelbrot set and displays it in an image
 */
public class JOCLSimpleMandelbrot
{
    /**
     * Entry point for this sample.
     *
     * @param args not used
     */
    public static void start(String args[])
    {
        SwingUtilities.invokeLater(new Runnable()
        {
            public void run()
            {
                new JOCLSimpleMandelbrot(800,600);
            }
        });
    }

    /**
     * The image which will contain the Mandelbrot pixel data
     */
    private BufferedImage image;

    /**
     * The width of the image
     */
    private int sizeX = 0;

    /**
     * The height of the image
     */
    private int sizeY = 0;

    /**
     * The component which is used for rendering the image
     */
    private JComponent imageComponent;

    /**
     * The OpenCL context
     */
    private cl_context context;

    /**
     * The OpenCL command queue
     */
    private cl_command_queue commandQueue;

    /**
     * The OpenCL kernel which will actually compute the Mandelbrot
     * set and store the pixel data in a CL memory object
     */
    private cl_kernel kernel;

    /**
     * The OpenCL memory object which stores the pixel data
     */
    private cl_mem pixelMem;

    /**
     * An OpenCL memory object which stores a nifty color map,
     * encoded as integers combining the RGB components of
     * the colors.
     */
    private cl_mem colorMapMem;

    /**
     * The color map which will be copied to OpenCL for filling
     * the PBO.
     */
    private int colorMap[];

    /**
     * The minimum x-value of the area in which the Mandelbrot
     * set should be computed
     */
    private float x0 = -2f;

    /**
     * The minimum y-value of the area in which the Mandelbrot
     * set should be computed
     */
    private float y0 = -1.3f;

    /**
     * The maximum x-value of the area in which the Mandelbrot
     * set should be computed
     */
    private float x1 = 0.6f;

    /**
     * The maximum y-value of the area in which the Mandelbrot
     * set should be computed
     */
    private float y1 = 1.3f;


    /**
     * Creates the JOCLSimpleMandelbrot sample with the given
     * width and height
     */
    public JOCLSimpleMandelbrot(int width, int height)
    {
        this.sizeX = width;
        this.sizeY = height;

        // Create the image and the component that will paint the image
        image = new BufferedImage(sizeX, sizeY, BufferedImage.TYPE_INT_RGB);
        imageComponent = new JPanel()
        {
            private static final long serialVersionUID = 1L;
            public void paintComponent(Graphics g)
            {
                super.paintComponent(g);
                g.drawImage(image, 0,0,this);
            }
        };

        // Initialize the mouse interaction
        initInteraction();

        // Initialize OpenCL
        initCL();

        // Initial image update
        updateImage();

        // Create the main frame
        JFrame frame = new JFrame("JOCL Simple Mandelbrot");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        imageComponent.setPreferredSize(new Dimension(width, height));
        frame.add(imageComponent, BorderLayout.CENTER);
        frame.pack();

        frame.setVisible(true);
    }

    /**
     * Initialize OpenCL: Create the context, the command queue
     * and the kernel.
     */
    private void initCL()
    {
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        // Create a context for the selected device
        context = clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null);

        // Create a command-queue for the selected device
        commandQueue =
                clCreateCommandQueue(context, device, 0, null);

        // Program Setup
        String source = readFile("build/resources/main/kernels/SimpleMandelbrot.cl");

        // Create the program
        cl_program cpProgram = clCreateProgramWithSource(context, 1,
                new String[]{ source }, null, null);

        // Build the program
        clBuildProgram(cpProgram, 0, null, "-cl-mad-enable", null, null);

        // Create the kernel
        kernel = clCreateKernel(cpProgram, "computeMandelbrot", null);

        // Create the memory object which will be filled with the
        // pixel data
        pixelMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                sizeX * sizeY * Sizeof.cl_uint, null, null);

        // Create and fill the memory object containing the color map
        initColorMap(32, Color.RED, Color.GREEN, Color.BLUE);
        colorMapMem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                colorMap.length * Sizeof.cl_uint, null, null);
        clEnqueueWriteBuffer(commandQueue, colorMapMem, true, 0,
                colorMap.length * Sizeof.cl_uint, Pointer.to(colorMap), 0, null, null);

        //for(int i=0; i<100000; i++){
        //    colorMapMem = clCreateBuffer(context, CL_MEM_READ_WRITE,
        //            colorMap.length * Sizeof.cl_uint, null, null);
        //    clEnqueueWriteBuffer(commandQueue, colorMapMem, true, 0,
        //            colorMap.length * Sizeof.cl_uint, Pointer.to(new int[colorMap.length]), 0, null, null);
        //}

    }

    /**
     * Helper function which reads the file with the given name and returns
     * the contents of this file as a String. Will exit the application
     * if the file can not be read.
     *
     * @param fileName The name of the file to read.
     * @return The contents of the file
     */
    private String readFile(String fileName)
    {
        File curDir = new File(".");
        File[] filesList = curDir.listFiles();
        for(File f : filesList){
            if(f.isDirectory())
                System.out.println(f.getName());
            if(f.isFile()){
                System.out.println(f.getName());
            }
        }

        try
        {

            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));
            StringBuffer sb = new StringBuffer();
            String line = null;
            while (true)
            {
                line = br.readLine();
                if (line == null)
                {
                    break;
                }
                sb.append(line).append("\n");
            }
            return sb.toString();
        }
        catch (IOException e)
        {
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    /**
     * Creates the colorMap array which contains RGB colors as integers,
     * interpolated through the given colors with colors.length * stepSize
     * steps
     *
     * @param stepSize The number of interpolation steps between two colors
     * @param colors The colors for the map
     */
    private void initColorMap(int stepSize, Color ... colors)
    {
        colorMap = new int[stepSize*colors.length];
        int index = 0;
        for (int i=0; i<colors.length-1; i++)
        {
            Color c0 = colors[i];
            int r0 = c0.getRed();
            int g0 = c0.getGreen();
            int b0 = c0.getBlue();

            Color c1 = colors[i+1];
            int r1 = c1.getRed();
            int g1 = c1.getGreen();
            int b1 = c1.getBlue();

            int dr = r1-r0;
            int dg = g1-g0;
            int db = b1-b0;

            for (int j=0; j<stepSize; j++)
            {
                float alpha = (float)j / (stepSize-1);
                int r = (int)(r0 + alpha * dr);
                int g = (int)(g0 + alpha * dg);
                int b = (int)(b0 + alpha * db);
                int rgb =
                        (r << 16) |
                                (g <<  8) |
                                (b <<  0);
                colorMap[index++] = rgb;
            }
        }
    }


    /**
     * Attach the mouse- and mouse wheel listeners to the glComponent
     * which allow zooming and panning the fractal
     */
    private void initInteraction()
    {
        final Point previousPoint = new Point();

        imageComponent.addMouseMotionListener(new MouseMotionListener()
        {
            @Override
            public void mouseDragged(MouseEvent e)
            {
                int dx = previousPoint.x - e.getX();
                int dy = previousPoint.y - e.getY();

                float wdx = x1-x0;
                float wdy = y1-y0;

                x0 += (dx / 150.0f) * wdx;
                x1 += (dx / 150.0f) * wdx;

                y0 += (dy / 150.0f) * wdy;
                y1 += (dy / 150.0f) * wdy;

                previousPoint.setLocation(e.getX(), e.getY());

                updateImage();
            }

            @Override
            public void mouseMoved(MouseEvent e)
            {
                previousPoint.setLocation(e.getX(), e.getY());
            }

        });

        imageComponent.addMouseWheelListener(new MouseWheelListener()
        {
            @Override
            public void mouseWheelMoved(MouseWheelEvent e)
            {
                float dx = x1-x0;
                float dy = y1-y0;
                float delta = e.getWheelRotation() / 20.0f;
                x0 += delta * dx;
                x1 -= delta * dx;
                y0 += delta * dy;
                y1 -= delta * dy;

                updateImage();
            }
        });
    }


    /**
     * Execute the kernel function and read the resulting pixel data
     * into the BufferedImage
     */
    private void updateImage()
    {
        // Set work size and execute the kernel
        long globalWorkSize[] = new long[2];
        globalWorkSize[0] = sizeX;
        globalWorkSize[1] = sizeY;

        int maxIterations = 250;
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(pixelMem));
        clSetKernelArg(kernel, 1, Sizeof.cl_uint, Pointer.to(new int[]{sizeX}));
        clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{sizeY}));
        clSetKernelArg(kernel, 3, Sizeof.cl_float, Pointer.to(new float[]{ x0 }));
        clSetKernelArg(kernel, 4, Sizeof.cl_float, Pointer.to(new float[]{ y0 }));
        clSetKernelArg(kernel, 5, Sizeof.cl_float, Pointer.to(new float[]{ x1 }));
        clSetKernelArg(kernel, 6, Sizeof.cl_float, Pointer.to(new float[]{ y1 }));
        clSetKernelArg(kernel, 7, Sizeof.cl_int, Pointer.to(new int[]{ maxIterations }));
        clSetKernelArg(kernel, 8, Sizeof.cl_mem, Pointer.to(colorMapMem));
        clSetKernelArg(kernel, 9, Sizeof.cl_int, Pointer.to(new int[]{ colorMap.length }));

        clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
                globalWorkSize, null, 0, null, null);

        // Read the pixel data into the BufferedImage
        DataBufferInt dataBuffer = (DataBufferInt)image.getRaster().getDataBuffer();
        int data[] = dataBuffer.getData();
        clEnqueueReadBuffer(commandQueue, pixelMem, CL_TRUE, 0,
                Sizeof.cl_int * sizeY * sizeX, Pointer.to(data), 0, null, null);

        imageComponent.repaint();
    }


}
