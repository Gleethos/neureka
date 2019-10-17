/*
 * JOCL - Java bindings for OpenCL
 *
 * Copyright 2009 Marco Hutter - http://www.jocl.org/
 */

package org.jocl.samples;

import static org.jocl.CL.*;

import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;
import java.awt.image.*;
import java.io.*;
import java.nio.IntBuffer;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.jocl.*;


/**
 * A sample application that computes the Mandelbrot set. The area of
 * the Mandelbrot set which is to be computed is divided into tiles,
 * and the tiles are scheduled for execution as individual tasks.
 * The tasks are processed concurrently by all available devices. <br />
 * <br />
 * The actual computation of the Mandelbrot set is done by an OpenCL
 * kernel which uses the "Quad-Float" data type: A high-precision
 * floating point data type that is represented as a single float4.
 */
public class MyJOCLMandelbrot
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
                new org.jocl.samples.MyJOCLMandelbrot();
            }
        });
    }

    /**
     * Constant for the default image size in x-direction
     */
    private static final int DEFAULT_SIZE_X = 256;

    /**
     * Constant for the default image size in y-direction
     */
    private static final int DEFAULT_SIZE_Y = 256;

    /**
     * Constant for the default tile size
     */
    private static final int DEFAULT_TILE_SIZE = 16;

    /**
     * Constant for the default number of iterations
     */
    private static final int DEFAULT_MAX_ITERATIONS = 100;

    /**
     * A flag indicating whether some sort of "benchmarking"
     * information should be printed for each tile
     */
    private static final boolean BENCHMARK = false;

    /**
     * Abstract base class for tasks that refer to a region of the Mandelbrot
     * set. The processColors method of this class may be implemented to
     * either fill the whole image with the preview, or to fill a small
     * region of the image with a single tile.
     */
    private abstract class Task
    {
        /**
         * The OpenCL memory object that the output should be
         * written to
         */
        protected cl_mem outputMem;

        /**
         * The total number of pixels in x-direction
         */
        protected int sizeX;

        /**
         * The total number of pixels in y-direction
         */
        protected int sizeY;

        /**
         * The x-coordinate of the tile which is computed by this task
         */
        protected int tileX;

        /**
         * The y-coordinate of the tile which is computed by this task
         */
        protected int tileY;

        /**
         * The size in x-direction of the tile which is computed by this task
         */
        protected int tileSizeX;

        /**
         * The size in y-direction of the tile which is computed by this task
         */
        protected int tileSizeY;

        /**
         * The total area of the Mandelbrot set that the tile this task
         * computes belongs to
         */
        protected Rectangle2D.Double area;

        /**
         * The maximum number of iterations to perform
         */
        protected int maxIterations;

        /**
         * Creates a new Task that computes the specified tile.
         *
         * @param outputMem The target memory object
         * @param sizeX The total number of pixels in x-direction
         * @param sizeY The total number of pixels in y-direction
         * @param tileX The x-coordinate of the tile
         * @param tileY The y-coordinate of the tile
         * @param tileSizeX The tile size in x-direction
         * @param tileSizeY The tile size in y-direction
         * @param area The Mandelbrot area
         * @param maxIterations The maximum number of iterations
         */
        Task(cl_mem outputMem,
             int sizeX, int sizeY,
             int tileX, int tileY,
             int tileSizeX, int tileSizeY,
             Rectangle2D.Double area, int maxIterations)
        {
            this.outputMem = outputMem;
            this.sizeX = sizeX;
            this.sizeY = sizeY;
            this.tileX = tileX;
            this.tileY = tileY;
            this.tileSizeX = tileSizeX;
            this.tileSizeY = tileSizeY;
            this.area = area;
            this.maxIterations = maxIterations;
        }

        /**
         * Set up the OpenCL arguments for this task for the given kernel
         *
         * @param kernel The OpenCL kernel for which the arguments will be set
         */
        protected void setupArguments(cl_kernel kernel)
        {
            clSetKernelArg(kernel,  0,
                    Sizeof.cl_mem, Pointer.to(outputMem));
            clSetKernelArg(kernel,  1,
                    Sizeof.cl_uint, Pointer.to(new int[]{sizeX}));
            clSetKernelArg(kernel,  2,
                    Sizeof.cl_uint, Pointer.to(new int[]{sizeY}));
            clSetKernelArg(kernel,  3,
                    Sizeof.cl_uint, Pointer.to(new int[]{tileX}));
            clSetKernelArg(kernel,  4,
                    Sizeof.cl_uint, Pointer.to(new int[]{tileY}));
            clSetKernelArg(kernel,  5,
                    Sizeof.cl_uint, Pointer.to(new int[]{tileSizeX}));
            clSetKernelArg(kernel,  6,
                    Sizeof.cl_uint, Pointer.to(new int[]{tileSizeY}));
            clSetKernelArg(kernel,  7,
                    Sizeof.cl_float2, Pointer.to(toDoubleFloat(area.x)));
            clSetKernelArg(kernel,  8,
                    Sizeof.cl_float2, Pointer.to(toDoubleFloat(area.y)));
            clSetKernelArg(kernel,  9,
                    Sizeof.cl_float2, Pointer.to(toDoubleFloat(area.width)));
            clSetKernelArg(kernel, 10,
                    Sizeof.cl_float2, Pointer.to(toDoubleFloat(area.height)));
            clSetKernelArg(kernel, 11,
                    Sizeof.cl_int, Pointer.to(new int[]{ maxIterations }));

        }

        /**
         * Compute the "low word" of the given double value
         * and return it as a float
         *
         * @param a The value
         * @return The low word
         */
        private float computeLo(double a)
        {
            double temp = ((1<<27)+1) * a;
            double hi = temp - (temp - a);
            double lo = a - hi;
            return (float)lo;
        }

        /**
         * Compute the "high word" of the given double value
         * and return it as a float
         *
         * @param a The value
         * @return The high word
         */
        private float computeHi(double a)
        {
            double temp = ((1<<27)+1) * a;
            double hi = temp - (temp - a);
            return (float)hi;
        }

        /**
         * Convert the given double value to an array containing
         * the "low- and high word" as float values
         *
         * @param d The value
         * @return The low- and high word of the value
         */
        private float[] toDoubleFloat(double d)
        {
            return new float[]{computeHi(d),computeLo(d)};
        }

        /**
         * Will execute this task with the given kernel on the given
         * command queue
         *
         * @param kernel The kernel
         * @param commandQueue The command queue
         */
        public void execute(cl_kernel kernel, cl_command_queue commandQueue)
        {
            setupArguments(kernel);

            long globalWorkSize[] = new long[2];
            globalWorkSize[0] = tileSizeX;
            globalWorkSize[1] = tileSizeY;

            cl_event event = new cl_event();

            clEnqueueNDRangeKernel(
                    commandQueue,
                    kernel, 2, null,
                    globalWorkSize, null, 0, null, event);

            clWaitForEvents(1, new cl_event[]{event});

            if (BENCHMARK && (this instanceof TileTask))
            {
                printBenchmarkInfo(
                        event, commandQueue, outputMem, tileX, tileY);
            }

            // Read the contents of the iterations memory object
            int size = tileSizeX * tileSizeY;
            int result[] = new int[size];
            Pointer target = Pointer.to(result);
            clEnqueueReadBuffer(
                    commandQueue, outputMem,
                    CL_TRUE, 0, size * Sizeof.cl_int,
                    target, 0, null, null);

            convertIterationsToColors(result);
            processColors(result);
        }

        /**
         * Process the given RGB colors
         *
         * @param rgbColors The RGB colors
         */
        protected abstract void processColors(int rgbColors[]);

    }

    /**
     * An implementation of a Task that computes one
     * tile of the Mandelbrot set
     */
    private class TileTask extends Task
    {
        /**
         * Creates a new Task that computes the specified tile.
         *
         * @param outputMem The target memory object
         * @param sizeX The total number of pixels in x-direction
         * @param sizeY The total number of pixels in y-direction
         * @param tileX The x-coordinate of the tile
         * @param tileY The y-coordinate of the tile
         * @param tileSizeX The tile size in x-direction
         * @param tileSizeY The tile size in y-direction
         * @param area The Mandelbrot area
         * @param maxIterations The maximum number of iterations
         */
        TileTask(cl_mem outputMem,
                 int sizeX, int sizeY,
                 int tileX, int tileY,
                 int tileSizeX, int tileSizeY,
                 Rectangle2D.Double area, int maxIterations)
        {
            super(outputMem, sizeX, sizeY, tileX, tileY,
                    tileSizeX, tileSizeY, area, maxIterations);
        }

        @Override
        protected void processColors(int rgbColors[])
        {
            // Write the colors into the bufferedImage
            int globalX = tileX * tileSizeX;
            int globalY = tileY * tileSizeY;
            int index = globalY * sizeX + globalX;
            DataBufferInt dataBuffer =
                    (DataBufferInt)image.getRaster().getDataBuffer();
            int data[] = dataBuffer.getData();
            for (int y=0; y<tileSizeY; y++)
            {
                System.arraycopy(
                        rgbColors, y*tileSizeX, data, index+y*sizeX, tileSizeX);
            }
        }

    }


    /**
     * An implementation of a Task that computes the Mandelbrot
     * set at a coarse resolution. It will process the computed
     * colors by filling the output image with rectangles of
     * the specified tile size.
     */
    private class PreviewTask extends Task
    {
        /**
         * The size in which the "pixels" will be painted for
         * the preview, in x-direction
         */
        private int paintedTileSizeX;

        /**
         * The size in which the "pixels" will be painted for
         * the preview, in y-direction
         */
        private int paintedTileSizeY;

        /**
         * Creates a new Task that computes the specified tile.
         *
         * @param outputMem The target memory object
         * @param sizeX The total number of pixels in x-direction
         * @param sizeY The total number of pixels in y-direction
         * @param tileSizeX The tile size in x-direction
         * @param tileSizeY The tile size in y-direction
         * @param area The Mandelbrot area
         * @param maxIterations The maximum number of iterations
         */
        PreviewTask(
                cl_mem outputMem,
                int sizeX, int sizeY,
                int tileSizeX, int tileSizeY,
                Rectangle2D.Double area, int maxIterations)
        {
            super(outputMem, sizeX, sizeY, 0, 0,
                    sizeX, sizeY, area, maxIterations);

            paintedTileSizeX = tileSizeX;
            paintedTileSizeY = tileSizeY;

        }

        @Override
        protected void processColors(int rgbColors[])
        {
            Graphics2D g = image.createGraphics();
            for (int x=0; x<numTilesX; x++)
            {
                for (int y=0; y<numTilesY; y++)
                {
                    int rgb = rgbColors[y*numTilesX+x];
                    Color color = new Color(rgb);
                    g.setColor(color);
                    g.fillRect(
                            x*paintedTileSizeX, y*paintedTileSizeX,
                            paintedTileSizeY, paintedTileSizeY);
                }
            }
            g.dispose();
        }
    }

    /**
     * A class that takes {@link Task}s from the taskQueue, executes
     * them and repaints the image component.
     */
    private class TaskProcessor implements Runnable
    {
        /**
         * The kernel which will be executed
         */
        protected cl_kernel kernel;

        /**
         * The OpenCL command queue
         */
        protected cl_command_queue commandQueue;

        /**
         * The list of tasks which are currently active
         */
        private List<Task> activeTasks =
                Collections.synchronizedList(new ArrayList<Task>());

        /**
         * Creates a new TaskProcessor which will execute the
         * given kernel on the given command queue
         *
         * @param kernel The kernel
         * @param commandQueue The command queue
         */
        public TaskProcessor(cl_kernel kernel, cl_command_queue commandQueue)
        {
            this.kernel = kernel;
            this.commandQueue = commandQueue;
        }

        @Override
        public void run()
        {
            while (true)
            {
                Task task = null;
                try
                {
                    task = taskQueue.take();
                }
                catch (InterruptedException e)
                {
                    Thread.currentThread().interrupt();
                    break;
                }
                activeTasks.add(task);
                task.execute(kernel, commandQueue);
                activeTasks.remove(task);
                synchronized (activeTasks)
                {
                    activeTasks.notifyAll();
                }
                imageComponent.repaint();

                // Tasks occupy the graphics card -
                // give it some time to breathe
                try
                {
                    Thread.sleep(5);
                }
                catch (InterruptedException e)
                {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }

        /**
         * Wait until all tasks are finished.
         */
        public void finish()
        {
            synchronized (activeTasks)
            {
                while (!activeTasks.isEmpty())
                {
                    try
                    {
                        activeTasks.wait();
                    }
                    catch (InterruptedException e)
                    {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }
    }


    /**
     * Converts the given array (which contains numbers of iterations)
     * into an array of colors using the colorMap
     *
     * @param array The array to convert
     */
    private void convertIterationsToColors(int array[])
    {
        for (int i=0; i<array.length; i++)
        {
            if (array[i] >= maxIterations)
            {
                array[i] = 0;
            }
            else
            {
                double alpha = (double)array[i] / maxIterations;
                int colorIndex = (int)(alpha * colorMap.length);
                array[i] = colorMap[colorIndex];
            }
        }
    }






    /**
     * The component which will paint the image of the
     * Mandelbrot set and the selection rectangle
     */
    private class ImageComponent extends JComponent
    {
        private static final long serialVersionUID = 1L;

        @Override
        public void paintComponent(Graphics gr)
        {
            super.paintComponent(gr);
            Graphics2D g = (Graphics2D)gr;

            g.drawImage(image, 0,0,this);

            // Paint the selection rectangle
            if (selectionRectangle != null)
            {
                int x = (int)(selectionRectangle.getX() * getWidth());
                int y = (int)(selectionRectangle.getY() * getHeight());
                int w = (int)(selectionRectangle.getWidth() * getWidth());
                int h = (int)(selectionRectangle.getHeight() * getWidth());
                g.setXORMode(Color.GRAY);
                g.drawRect(x,y,w,h);
            }
        }

        /**
         * Computes the relative position of the given point inside
         * the image component, i.e. the position in [0,1]x[0,1]
         *
         * @param point The point
         * @return The relative position
         */
        private Point2D.Double toRelative(Point point)
        {
            double x = (double)point.x / getWidth();
            double y = (double)point.y / getHeight();
            return new Point2D.Double(x,y);
        }

    }



    /**
     * Helper function which reads the file with the given name and returns
     * the contents of this file as a String. Will exit the application
     * if the file can not be read.
     *
     * @param fileName The name of the file to read.
     * @return The contents of the file
     */
    private static String readFile(String fileName)
    {
        try
        {
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(new FileInputStream(fileName)));
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
     * The OpenCL context
     */
    private cl_context context;

    /**
     * The number of OpenCL devices that may be used
     */
    private int numDevices;

    /**
     * The OpenCL command queues
     */
    private cl_command_queue commandQueues[];

    /**
     * The OpenCL kernels which will actually compute the Mandelbrot
     * set and store the numbers of iterations in the
     * memory object
     */
    private cl_kernel kernels[];

    /**
     * The OpenCL memory objects which store the number of
     * iterations for each pixel. One memory object for
     * each tile.
     */
    private cl_mem iterationsMem[][];

    /**
     * The OpenCL memory object which stores the number of
     * iterations for the preview
     */
    private cl_mem previewIterationsMem;

    /**
     * The maximum number of iterations for the Mandelbrot computation
     */
    private int maxIterations = DEFAULT_MAX_ITERATIONS;

    /**
     * The area in which the Mandelbrot set should be computed
     */
    private Rectangle2D.Double area =
            new Rectangle2D.Double(-2, -1.3, 2.6, 2.6);

    /**
     * The number of pixels in x-direction
     */
    private int sizeX = DEFAULT_SIZE_X;

    /**
     * The number of pixels in y-direction
     */
    private int sizeY = DEFAULT_SIZE_Y;

    /**
     * The horizontal tile size
     */
    private int tileSizeX = DEFAULT_TILE_SIZE;

    /**
     * The vertical tile size
     */
    private int tileSizeY = DEFAULT_TILE_SIZE;

    /**
     * The number of tiles in x-direction
     */
    private int numTilesX;

    /**
     * The number of tiles in y-direction
     */
    private int numTilesY;

    /**
     * The {@link TaskProcessor}s which will execute the tasks
     * for computing the preview- and tile kernels
     */
    private TaskProcessor taskProcessors[];

    /**
     * The queue of Tasks that are about to be executed by
     * the {@link TaskProcessor}.
     */
    private BlockingQueue<Task> taskQueue =
            new ArrayBlockingQueue<Task>(32768, true);

    /**
     * The image which will be used to display the pixels
     */
    private BufferedImage image;

    /**
     * The component which is used for painting the image
     */
    private ImageComponent imageComponent;

    /**
     * An array which contains RGB colors. The number of iterations
     * that are computed by the Mandelbrot OpenCL kernels will be
     * translated into colors using this map.
     */
    private int colorMap[] = null;

    /**
     * The area selection rectangle, in "world" (Mandelbrot) coordinates
     */
    private Rectangle2D.Double selectionRectangle = null;

    /**
     * The point where mouse dragging started
     */
    private Point dragStart = null;

    /**
     * The stack of views (areas) which have been pushed
     */
    private Stack<Rectangle2D.Double> viewStack =
            new Stack<Rectangle2D.Double>();

    /**
     * The list model for the views (areas)
     */
    private DefaultListModel viewListModel;

    /**
     * The list displaying the views (areas)
     */
    private JList viewList;

    /**
     * Creates a new MyJOCLMandelbrot
     */
    public MyJOCLMandelbrot()
    {
        // Initialize OpenCL
        initCL();

        // Create the Swing component which paints the image
        imageComponent = new ImageComponent();
        imageComponent.setPreferredSize(new Dimension(sizeX, sizeY));

        // Initialize the mouse listeners for the interaction
        initInteraction();

        // Create the main frame containing the imageComponent
        // and the control panel
        JFrame frame = new JFrame("JOCL Mandelbrot");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        JPanel p = new JPanel(new FlowLayout());
        p.setPreferredSize(new Dimension(1024, 1024));
        p.add(imageComponent);
        frame.add(p, BorderLayout.CENTER);
        frame.add(createControlPanel(), BorderLayout.EAST);
        frame.pack();
        frame.setVisible(true);

        // Trigger the initial rendering
        pushView();
        doRender();

    }


    /**
     * Initialize OpenCL
     */
    private void initCL()
    {
        // The platform and device type that will be used
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;

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

        System.out.println("Using plaform "+
                getPlatformInfoString(platform, CL.CL_PLATFORM_NAME));

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        numDevices = numDevicesArray[0];

        // Obtain the device IDs
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);

        for (int i=0; i<numDevices; i++)
        {
            System.out.println("Device "+i+": "+
                    getDeviceInfoString(devices[i], CL.CL_DEVICE_NAME));
        }

        // Create a context for the selected devices
        context = clCreateContext(
                contextProperties, numDevices, devices,
                null, null, null);

        // Read the kernel files and set up the OpenCL program
        String source0 = readFile("build/resources/main/kernels/QuadFloat.cl");
        String source1 = readFile("build/resources/main/kernels/QuadFloatMandelbrot.cl");
        cl_program program = clCreateProgramWithSource(context, 2,
                new String[]{ source0, source1 }, null, null);
        clBuildProgram(program, 0, null, "-cl-mad-enable", null, null);

        // Create a the command-queues and kernels
        commandQueues = new cl_command_queue[numDevices];
        kernels = new cl_kernel[numDevices];
        long properties = 0;
        if (BENCHMARK)
        {
            properties |= CL_QUEUE_PROFILING_ENABLE;
        }
        for (int i=0; i<numDevices; i++)
        {
            commandQueues[i] =
                    clCreateCommandQueue(context, devices[i], properties, null);
            kernels[i] = clCreateKernel(program, "computeMandelbrot", null);
        }
        // Create the color map
        colorMap = createColorMap(2048,
                Color.RED, Color.YELLOW, Color.GREEN,
                Color.CYAN, Color.BLUE, Color.MAGENTA);

        // Start the task processors
        taskProcessors = new TaskProcessor[numDevices];
        for (int i=0; i<numDevices; i++)
        {
            taskProcessors[i] =
                    new TaskProcessor(kernels[i], commandQueues[i]);
            Thread thread =
                    new Thread(taskProcessors[i], "taskProcessorThread"+i);
            thread.setDaemon(true);
            thread.start();
        }

        // Initialize the BufferedImage and the OpenCL memory
        initImage(
                DEFAULT_SIZE_X, DEFAULT_SIZE_Y,
                DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE);
    }


    /**
     * Initializes the OpenCL memory object and the BufferedImage which
     * will later receive the pixels
     */
    private void initImage(
            int newSizeX, int newSizeY, int newTileSizeX, int newTileSizeY)
    {

        // Flush all pending tasks
        flush();

        // Release all existing memory objects
        if (iterationsMem != null)
        {
            for (int x=0; x<numTilesX; x++)
            {
                for (int y=0; y<numTilesY; y++)
                {
                    clReleaseMemObject(iterationsMem[x][y]);
                    iterationsMem[x][y] = null;
                }
            }
            clReleaseMemObject(previewIterationsMem);
            iterationsMem = null;
        }

        // Set the new image size parameters
        this.sizeX = newSizeX;
        this.sizeY = newSizeY;
        this.tileSizeX = newTileSizeX;
        this.tileSizeY = newTileSizeY;
        numTilesX = sizeX / tileSizeX;
        numTilesY = sizeY / tileSizeY;

        // Create the new memory objects and the image
        iterationsMem = new cl_mem[numTilesX][numTilesY];
        for (int x=0; x<numTilesX; x++)
        {
            for (int y=0; y<numTilesY; y++)
            {
                iterationsMem[x][y] =
                        clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                tileSizeX * tileSizeY * Sizeof.cl_uint, null, null);
            }
        }
        previewIterationsMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                numTilesX * numTilesY * Sizeof.cl_uint, null, null);

        image = new BufferedImage(sizeX, sizeY, BufferedImage.TYPE_INT_RGB);
        if (imageComponent != null)
        {
            imageComponent.setPreferredSize(new Dimension(sizeX, sizeY));
        }
    }

    /**
     * Flush all pending tasks and finish all running tasks
     */
    private void flush()
    {
        taskQueue.drainTo(new ArrayList<Task>());
        for (int i=0; i<numDevices; i++)
        {
            clFinish(commandQueues[i]);
            taskProcessors[i].finish();
        }
    }


    /**
     * Attach the mouse- and mouse wheel listeners to the imageComponent
     * which allow zooming and panning and selecting regions to zoom in
     */
    private void initInteraction()
    {
        final Point previousPoint = new Point();

        imageComponent.addMouseListener(new MouseAdapter()
        {
            @Override
            public void mousePressed(MouseEvent e)
            {
                if ((e.getModifiersEx() & MouseEvent.BUTTON1_DOWN_MASK) ==
                        MouseEvent.BUTTON1_DOWN_MASK)
                {
                    dragStart = e.getPoint();
                    imageComponent.repaint();
                }
            }
            @Override
            public void mouseReleased(MouseEvent e)
            {
                if (selectionRectangle != null)
                {
                    double rx0 = selectionRectangle.x;
                    double ry0 = selectionRectangle.y;
                    double rdx = selectionRectangle.width;
                    double rdy = selectionRectangle.height;

                    if (rdx > rdy)
                    {
                        rdy = rdx;
                    }
                    else
                    {
                        rdx = rdy;
                    }
                    area.x += rx0 * area.width;
                    area.y += ry0 * area.height;
                    area.width *= rdx;
                    area.height *= rdy;

                    pushView();
                    doRender();
                }
                selectionRectangle = null;
                imageComponent.repaint();
            }
        });

        imageComponent.addMouseMotionListener(new MouseMotionListener()
        {
            @Override
            public void mouseDragged(MouseEvent e)
            {
                if ((e.getModifiersEx() & MouseEvent.BUTTON3_DOWN_MASK) ==
                        MouseEvent.BUTTON3_DOWN_MASK)
                {
                    int deltaX = previousPoint.x - e.getX();
                    int deltaY = previousPoint.y - e.getY();

                    area.x += (deltaX / 150.0f) * area.width;
                    area.y += (deltaY / 150.0f) * area.height;

                    previousPoint.setLocation(e.getX(), e.getY());

                    doRender();
                }
                else
                {
                    if (e.getPoint().distance(dragStart) < 5)
                    {
                        selectionRectangle = null;
                        imageComponent.repaint();
                        return;
                    }
                    Point2D.Double rStart =
                            imageComponent.toRelative(dragStart);
                    Point2D.Double rEnd =
                            imageComponent.toRelative(e.getPoint());

                    Point2D.Double rMin = new Point2D.Double(
                            Math.min(rStart.x, rEnd.x),
                            Math.min(rStart.y, rEnd.y));

                    Point2D.Double rMax = new Point2D.Double(
                            Math.max(rStart.x, rEnd.x),
                            Math.max(rStart.y, rEnd.y));

                    selectionRectangle = new Rectangle2D.Double(
                            rMin.x, rMin.y, rMax.x-rMin.x, rMax.y-rMin.y);

                    imageComponent.repaint();
                }
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
                double delta = e.getWheelRotation() / 20.0f;
                area.x += delta * area.width / 2;
                area.width -= delta * area.width;
                area.y += delta * area.height / 2;
                area.height -= delta * area.height;

                doRender();
            }
        });
    }


    /**
     * Creates and returns the control panel
     *
     * @return The control panel
     */
    private JPanel createControlPanel()
    {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        panel.add(createParameterPanel());
        panel.add(createViewPanel());

        JButton b = new JButton("Save image...");
        b.addActionListener(new ActionListener()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                JFileChooser fileChooser = new JFileChooser();
                fileChooser.setFileFilter(
                        new FileNameExtensionFilter("PNG files", "png"));
                int result = fileChooser.showSaveDialog(null);
                if (result == JFileChooser.APPROVE_OPTION)
                {
                    File file = fileChooser.getSelectedFile();
                    if (!file.getName().toLowerCase().endsWith("png"))
                    {
                        String extension = "png";
                        if (!file.getName().endsWith("."))
                        {
                            extension = ".png";
                        }
                        file = new File(file.getPath()+extension);
                    }
                    try
                    {
                        ImageIO.write(image, "png", file);
                    }
                    catch (IOException ex)
                    {
                        System.err.println("Error while saving image:");
                        ex.printStackTrace();
                    }
                }
            }
        });
        panel.add(b);

        JPanel controlPanel = new JPanel(new BorderLayout());
        controlPanel.add(panel, BorderLayout.NORTH);
        return controlPanel;

    }

    /**
     * Creates the panel containing the controls for the image
     * computation parameters, like image and tile size and
     * the number of iterations.
     *
     * @return The parameter panel
     */
    private JComponent createParameterPanel()
    {
        JPanel parameterPanel = new JPanel(new GridLayout(0,2));
        parameterPanel.setBorder(
                BorderFactory.createTitledBorder("Parameters"));

        parameterPanel.add(new JLabel("Tile size:"));
        final JSpinner tileSizeSpinner =
                new JSpinner(new SpinnerNumberModel(tileSizeX,1,1024,1));
        parameterPanel.add(tileSizeSpinner);

        parameterPanel.add(new JLabel("Size X:"));
        final JSpinner sizeXSpinner =
                new JSpinner(new SpinnerNumberModel(sizeX,1,1024,1));
        parameterPanel.add(sizeXSpinner);

        parameterPanel.add(new JLabel("Size Y:"));
        final JSpinner sizeYSpinner =
                new JSpinner(new SpinnerNumberModel(sizeY,1,1024,1));
        parameterPanel.add(sizeYSpinner);

        parameterPanel.add(new JLabel("Internal size:"));
        final JLabel sizeLabel = new JLabel(sizeX+"x"+sizeY);
        parameterPanel.add(sizeLabel);

        parameterPanel.add(new JLabel("Max. Iterations"));
        final JTextField maxIterationsTextField =
                new JTextField(String.valueOf(maxIterations),6);
        parameterPanel.add(maxIterationsTextField);


        ChangeListener changeListener = new ChangeListener()
        {
            @Override
            public void stateChanged(ChangeEvent e)
            {
                int ts = (Integer)tileSizeSpinner.getValue();
                int sx = (Integer)sizeXSpinner.getValue();
                int sy = (Integer)sizeYSpinner.getValue();
                int nx = (int)Math.ceil((double)sx / ts);
                int ny = (int)Math.ceil((double)sy / ts);
                sizeLabel.setText((nx*ts)+"x"+(ny*ts));
            }
        };
        tileSizeSpinner.addChangeListener(changeListener);
        sizeXSpinner.addChangeListener(changeListener);
        sizeYSpinner.addChangeListener(changeListener);

        JButton applyButton = new JButton("Apply");
        applyButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                int ts = (Integer)tileSizeSpinner.getValue();
                int sx = (Integer)sizeXSpinner.getValue();
                int sy = (Integer)sizeYSpinner.getValue();
                int nx = (int)Math.ceil((double)sx / ts);
                int ny = (int)Math.ceil((double)sy / ts);

                try
                {
                    maxIterations =
                            Integer.parseInt(maxIterationsTextField.getText());
                }
                catch (NumberFormatException ex)
                {
                    maxIterationsTextField.setText(
                            String.valueOf(maxIterations));
                }

                initImage(nx*ts, ny*ts, ts, ts);
                doRender();

                imageComponent.invalidate();
                imageComponent.getParent().validate();
                imageComponent.repaint();
            }
        });
        parameterPanel.add(applyButton);
        return parameterPanel;
    }

    /**
     * Creates and returns the panel containing the "view stack"
     * list and the control buttons for the views.
     *
     * @return The view panel
     */
    private JComponent createViewPanel()
    {
        JPanel viewPanel = new JPanel(new BorderLayout());
        viewPanel.setBorder(
                BorderFactory.createTitledBorder("Views"));

        JPanel buttonsPanel = new JPanel(new GridLayout(1,2));

        JButton b = null;

        b = new JButton("Push");
        b.addActionListener(new ActionListener()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                pushView();
            }
        });
        buttonsPanel.add(b);

        b = new JButton("Peek");
        b.addActionListener(new ActionListener()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                peekView();
            }
        });
        buttonsPanel.add(b);

        b = new JButton("Pop");
        b.addActionListener(new ActionListener()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                popView();
            }
        });
        buttonsPanel.add(b);

        viewPanel.add(buttonsPanel, BorderLayout.NORTH);

        viewListModel = new DefaultListModel();
        viewList = new JList(viewListModel);
        viewList.setCellRenderer(new DefaultListCellRenderer()
        {
            private static final long serialVersionUID = 1L;

            @Override
            public Component getListCellRendererComponent(
                    JList list, Object value, int index,
                    boolean isSelected, boolean cellHasFocus)
            {
                super.getListCellRendererComponent(
                        list, value, index,
                        isSelected, cellHasFocus);
                Rectangle2D.Double r = (Rectangle2D.Double)value;
                String s =
                        "<html>" +
                                "x: "+r.x+"<br>" +
                                "y: "+r.y+"<br>" +
                                "dx: "+r.width+"<br>" +
                                "dy: "+r.height+"<hr></html>";
                setText(s);
                return this;
            }

        });
        viewPanel.add(new JScrollPane(viewList), BorderLayout.CENTER);
        return viewPanel;
    }



    /**
     * Push the current view (area) on the stack
     */
    private void pushView()
    {
        Rectangle2D.Double view =
                new Rectangle2D.Double(
                        area.x, area.y, area.width, area.height);
        viewStack.push(view);
        viewListModel.add(0, view);
        viewList.clearSelection();
    }

    /**
     * Peek the current view (area) on the stack
     */
    private void peekView()
    {
        Rectangle2D.Double view = viewStack.peek();
        area = new Rectangle2D.Double(
                view.x, view.y, view.width, view.height);
        viewList.clearSelection();
        doRender();
    }

    /**
     * Pop the current view (area) from the stack and render
     */
    private void popView()
    {
        if (viewStack.size() > 1)
        {
            viewStack.pop();
            viewListModel.remove(0);
        }
        peekView();
    }



    /**
     * Creates the colorMap array which contains RGB colors as integers,
     * interpolated through the given colors with colors.length * stepSize
     * steps
     *
     * @param stepSize The number of interpolation steps between two colors
     * @param colors The colors for the map
     * @return The color map
     */
    private int[] createColorMap(int stepSize, Color ... colors)
    {
        int colorMap[] = new int[stepSize*colors.length];
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
        return colorMap;
    }


    /**
     * Schedules tasks for computing the preview and the actual
     * image of the Mandelbrot for the current region.
     */
    private void doRender()
    {
        flush();

        Rectangle2D.Double currentArea =
                new Rectangle2D.Double(area.x, area.y, area.width, area.height);

        // Schedule the preview task
        Task previewTask = new PreviewTask(
                previewIterationsMem, numTilesX, numTilesY,
                tileSizeX, tileSizeY, currentArea, maxIterations);
        try
        {
            taskQueue.put(previewTask);
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
        }

        // Schedule the tasks for computing the tiles
        for (int x=0; x<numTilesX; x++)
        {
            for (int y=0; y<numTilesY; y++)
            {
                Task task = new TileTask(
                        iterationsMem[x][y], sizeX, sizeY, x, y,
                        tileSizeX, tileSizeY, currentArea, maxIterations);
                try
                {
                    taskQueue.put(task);
                }
                catch (InterruptedException e)
                {
                    x = numTilesX;
                    y = numTilesY;
                    Thread.currentThread().interrupt();
                }
            }
        }
    }


    /*
     * Print some "benchmarking" information
     */
    private void printBenchmarkInfo(
            cl_event event, cl_command_queue commandQueue,
            cl_mem iterationsMem, int tileX, int tileY)
    {
        IntBuffer buffer = IntBuffer.allocate(tileSizeX * tileSizeY);
        clEnqueueReadBuffer(
                commandQueue, iterationsMem,
                CL_TRUE, 0, tileSizeX * tileSizeY * Sizeof.cl_int,
                Pointer.to(buffer), 0, null, null);

        long sum = 0;
        for (int x=0; x<tileSizeX; x++)
        {
            for (int y=0; y<tileSizeY; y++)
            {
                int index = y * tileSizeX + x;
                int value = buffer.get(index);
                sum += value;
            }
        }

        double executionTime = computeExecutionTimeMs(event);
        System.out.println(
                "Tile "+tileX+" "+tileY+": "+
                        sum+" iterations, "+
                        executionTime+" ms, "+
                        (long)((1000/executionTime)*sum)+" i/s");
    }

    /*
     * Compute the execution time for the given event, in milliseconds
     */
    private static double computeExecutionTimeMs(cl_event event)
    {
        long startTime[] = new long[1];
        long endTime[] = new long[1];
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                Sizeof.cl_ulong, Pointer.to(endTime), null);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                Sizeof.cl_ulong, Pointer.to(startTime), null);
        return (endTime[0]-startTime[0]) / 1e6;
    }


    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private static String getDeviceInfoString(cl_device_id device, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }


    /**
     * Returns the value of the platform info parameter with the given name
     *
     * @param platform The platform
     * @param paramName The parameter name
     * @return The value
     */
    private static String getPlatformInfoString(cl_platform_id platform, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        clGetPlatformInfo(platform, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        clGetPlatformInfo(platform, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }

}
