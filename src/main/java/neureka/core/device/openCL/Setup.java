package neureka.core.device.openCL;

import org.jocl.*;

import java.awt.*;
import java.io.*;

import static org.jocl.CL.*;
import static org.jocl.CL.clEnqueueWriteBuffer;

public class Setup {

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
     * Initialize OpenCL: Create the context, the command queue
     * and the kernel.
     */
    public void initCL()
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


        cl_mem[] memos = new cl_mem[100];
        for(int i=0; i<memos.length; i++){
            int size = 1000 * 1000;
            memos[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * Sizeof.cl_uint, null, null);
            int[] data = new int[size];
            for(int ii=0; ii<size; ii++){
                data[ii] = ii;
            }
            clEnqueueWriteBuffer(
                    commandQueue, memos[i], true, 0,
                            size * Sizeof.cl_uint, Pointer.to(data), 0, null, null);
        }
        int err = clEnqueueMigrateMemObjects(
                commandQueue,
                memos.length,
                memos
                ,
                CL_MIGRATE_MEM_OBJECT_HOST,
        0,
                null,
                null);
        try {
            Thread.sleep(6000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // Create the memory object which will be filled with the
        // pixel data
        //pixelMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeX * sizeY * Sizeof.cl_uint, null, null);
//
        //// Create and fill the memory object containing the color map
        //initColorMap(32, Color.RED, Color.GREEN, Color.BLUE);
        //colorMapMem = clCreateBuffer(context, CL_MEM_READ_WRITE,
        //        colorMap.length * Sizeof.cl_uint, null, null);
        //clEnqueueWriteBuffer(commandQueue, colorMapMem, true, 0,
        //        colorMap.length * Sizeof.cl_uint, Pointer.to(colorMap), 0, null, null);
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


}
