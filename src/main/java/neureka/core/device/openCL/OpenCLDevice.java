package neureka.core.device.openCL;

import static org.jocl.CL.*;

import java.io.*;
import java.nio.*;
import java.util.*;

import neureka.core.Tsr;
import neureka.core.device.IDevice;
import org.jocl.*;


public class OpenCLDevice implements IDevice
{
    private static List<OpenCLDevice> DEVICES;
    static{
        DEVICES = findAllDevices();
    }
    private static List<OpenCLDevice> findAllDevices(){
        // Obtain the number of platforms
        int numPlatforms[] = new int[1];
        clGetPlatformIDs(0, null, numPlatforms);

        System.out.println("Number of platforms: "+numPlatforms[0]);

        // Obtain the platform IDs
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms[0]];
        clGetPlatformIDs(platforms.length, platforms, null);

        List<OpenCLDevice> myDevices = new ArrayList<>();

        // Collect all devices of all platforms
        List<cl_device_id> devices = new ArrayList<cl_device_id>();
        for (int i=0; i<platforms.length; i++)
        {
            String platformName = DeviceQuery.getString(platforms[i], CL_PLATFORM_NAME);

            // Obtain the number of devices for the current platform
            int numDevices[] = new int[1];
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, null, numDevices);

            System.out.println("Number of devices in platform "+platformName+": "+numDevices[0]);

            cl_device_id devicesArray[] = new cl_device_id[numDevices[0]];
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices[0], devicesArray, null);

            devices.addAll(Arrays.asList(devicesArray));
            for(cl_device_id did : devicesArray){
                OpenCLDevice clDevice = new OpenCLDevice(platforms[i] , did);
                myDevices.add(clDevice);
            }
        }
        return myDevices;
    }

    class cl_tsr{
        public int fp = 1;
        public cl_mem shape;
        public cl_mem translation;
        public cl_mem value;
    }

    private Map<Tsr, cl_tsr> _mapping = new HashMap<>();

    private cl_platform_id _pid;
    private cl_device_id _did;

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
    private Map<String, cl_kernel> _kernels;

    /**==============================================================================================================**/

    public static List<OpenCLDevice> DEVICES(){
        return DEVICES;
    }

    public OpenCLDevice(cl_platform_id pid, cl_device_id did) {
        _pid = pid;
        _did = did;
        _kernels = new HashMap<>();

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
        commandQueue = clCreateCommandQueue(context, device, 0, null);

        //####Reading all kernels!
        File curDir = new File("build/resources/main/kernels/");
        File[] filesList = curDir.listFiles();
        for(File f : filesList){
            if(f.isDirectory())
                System.out.println(f.getName());
            if(f.isFile()){
                System.out.println(f.getName());
            }
        }
        String[] sources = new String[filesList.length];
        for(int i=0; i<sources.length; i++) {
            sources[i] = readFile(filesList[i].toString());
        }
        //####
        // Program Setup
        //String source = readFile("build/resources/main/kernels/SimpleMandelbrot.cl");

        // Create the program
        cl_program cpProgram = clCreateProgramWithSource(context, sources.length,
                sources, null, null);//new String[]{ source }

        // Build the program
        int err = clBuildProgram(cpProgram, 0, null, "-cl-mad-enable", null, null);
        //TODO: check compilation errors!

        // Create the kernel
        _kernels.put(
                "computeMandelbrot",
                clCreateKernel(cpProgram, "computeMandelbrot", null)
        );//TODO: name kernels after file!

        //--- ########### ---\\
        //Testing:
        cl_mem[] memos = new cl_mem[300];
        for(int i=0; i<memos.length; i++){
            int size = 1000 * 1000;//CL_MEM_WRITE_ONLY
            memos[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, size * Sizeof.cl_uint, null, null);
            int[] data = new int[size];
            for(int ii=0; ii<size; ii++){
                data[ii] = ii;//new Random().nextInt();
            }
            clEnqueueWriteBuffer(
                    commandQueue, memos[i], true, 0,
                    size * Sizeof.cl_uint, Pointer.to(data), 0, null, null);
        }
        err = clEnqueueMigrateMemObjects(
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



    //---

    @Override
    public void dispose() {

    }

    @Override
    public IDevice add(Tsr tensor) {
        add(tensor, 1);
        return this;
    }

    public IDevice add(Tsr tensor, int fp) {
        cl_tsr newClTsr = new cl_tsr();


        double[] hostData = tensor.value();
        //VALUE TRANSFER:
        newClTsr.value = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,
                hostData.length * Sizeof.cl_float*fp,
                null,
                null
        );
        clEnqueueWriteBuffer(
                commandQueue,
                newClTsr.value,
                true, 0,
                hostData.length * Sizeof.cl_float2,
                Pointer.to(hostData),
                0, null, null
        );
        //SHAPE TRANSFER:
        newClTsr.shape = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,
                hostData.length * Sizeof.cl_int,
                null, null
        );
        clEnqueueWriteBuffer(
                commandQueue,
                newClTsr.shape,
                true, 0,
                tensor.shape().length * Sizeof.cl_int,
                Pointer.to(tensor.shape()),
                0, null, null
        );
        //TRANSLATION TRANSFER:
        newClTsr.translation = clCreateBuffer(
                context,
                CL_MEM_READ_WRITE,
                tensor.translation().length * Sizeof.cl_int,
                null, null
        );
        clEnqueueWriteBuffer(
                commandQueue,
                newClTsr.translation,
                true, 0,
                tensor.translation().length * Sizeof.cl_int,
                Pointer.to(tensor.shape()),
                0, null, null
        );
        _mapping.put(tensor, newClTsr);
        tensor.add(this);
        tensor.setIsOutsourced(true);
        return this;
    }

    @Override
    public IDevice rmv(Tsr tensor) {

        cl_tsr clTsr = _mapping.get(tensor);
        clEnqueueWriteBuffer(commandQueue, clTsr.value, true, 0,
                tensor.size() * Sizeof.cl_uint, Pointer.to(new double[666]), 0, null, null);
        //remove translations/shapes from device!
        _mapping.remove(tensor);
        return this;
    }

    @Override
    public IDevice overwrite(Tsr tensor, double[] value) {
        return this;
    }

    @Override
    public IDevice swap(Tsr former, Tsr replacement) {
        return this;
    }

    //---

    public String name(){
        return DeviceQuery.getString(_did, CL_DEVICE_NAME);
    }

    public String vendor(){
        return DeviceQuery.getString(_did, CL_DEVICE_VENDOR);
    }

    public String version(){
        return DeviceQuery.getString(_did, CL_DRIVER_VERSION);
    }

    public String type(){
        // CL_DEVICE_TYPE
        long deviceType = DeviceQuery.getLong(_did, CL_DEVICE_TYPE);
        if( (deviceType & CL_DEVICE_TYPE_CPU) != 0)
            return "CPU";
        if( (deviceType & CL_DEVICE_TYPE_GPU) != 0)
            return "GPU";
        if( (deviceType & CL_DEVICE_TYPE_ACCELERATOR) != 0)
            return "ACCELERATOR";
        if( (deviceType & CL_DEVICE_TYPE_DEFAULT) != 0)
            return "DEFAULT";
        return "UNKNOWN";
    }

    public int maxComputeUnits(){
        return DeviceQuery.getInt(_did, CL_DEVICE_MAX_COMPUTE_UNITS);
    }

    public long maxWorkItemSimensions(){
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    }

    public long[] maxWorkItemSizes(){
        return DeviceQuery.getSizes(_did, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
    }

    public long maxWorkGroupSize(){
        return DeviceQuery.getSize(_did, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    }

    public long maxClockFrequenzy(){
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_CLOCK_FREQUENCY);
    }

    public int maxAddressBits(){
        return DeviceQuery.getInt(_did, CL_DEVICE_ADDRESS_BITS);
    }

    public long maxMemAllocSize(){
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    }

    public long globalMemSize(){
        return DeviceQuery.getLong(_did, CL_DEVICE_GLOBAL_MEM_SIZE);
    }

    public int errorCorrectionSupport(){
        return DeviceQuery.getInt(_did, CL_DEVICE_ERROR_CORRECTION_SUPPORT);
    }

    public int localMemType(){
        return DeviceQuery.getInt(_did, CL_DEVICE_LOCAL_MEM_TYPE);
    }

    public long localMemSize(){
        return DeviceQuery.getLong(_did, CL_DEVICE_LOCAL_MEM_SIZE);
    }

    public long maxConstantBufferSize(){
        return DeviceQuery.getLong(_did, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    }

    public long maxConstantBufferSizeKB(){
        return (int)(DeviceQuery.getLong(_did, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)/1024);
    }

    public boolean queueExecIsOrdered(){
        long queueProperties = DeviceQuery.getLong(_did, CL_DEVICE_QUEUE_PROPERTIES);
        return (( queueProperties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ) != 0);
    }

    public boolean queueProfilingIsEnabled(){
        long queueProperties = DeviceQuery.getLong(_did, CL_DEVICE_QUEUE_PROPERTIES);
        return (( queueProperties & CL_QUEUE_PROFILING_ENABLE ) != 0);
    }

    public int imageSupport(){
        return DeviceQuery.getInt(_did, CL_DEVICE_IMAGE_SUPPORT);
    }

    public int maxReadImageArgs(){
        return DeviceQuery.getInt(_did, CL_DEVICE_MAX_READ_IMAGE_ARGS);
    }

    public int maxWriteImageArgs(){
       return DeviceQuery.getInt(_did, CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
    }

    public long singleFPConfig(){
        return DeviceQuery.getLong(_did, CL_DEVICE_SINGLE_FP_CONFIG);
    }

    public long image2DMaxWidth(){
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE2D_MAX_WIDTH);
    }

    public long image2DMaxHeight(){
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
    }

    public long image3DMaxWidth(){
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE3D_MAX_WIDTH);
    }

    public long image3DMaxHeight(){
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE3D_MAX_HEIGHT);
    }

    public long image3DMaxDepth(){
        return DeviceQuery.getSize(_did, CL_DEVICE_IMAGE3D_MAX_DEPTH);
    }

    public int prefVecWidthChar(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
    }

    public int prefVecWidthShort(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
    }

    public int prefVecWidthInt(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
    }

    public int prefVecWidthLong(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
    }

    public int prefVecWidthFloat(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
    }

    public int prefVecWidthDouble(){
        return DeviceQuery.getInt(_did, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
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





    public static class DeviceQuery
    {
        /**
         * Returns the value of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @return The value
         */
        public static int getInt(cl_device_id device, int paramName)
        {
            return getInts(device, paramName, 1)[0];
        }

        /**
         * Returns the values of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value
         */
        public static int[] getInts(cl_device_id device, int paramName, int numValues)
        {
            int values[] = new int[numValues];
            clGetDeviceInfo(device, paramName, Sizeof.cl_int * numValues, Pointer.to(values), null);
            return values;
        }

        /**
         * Returns the value of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @return The value
         */
        public static long getLong(cl_device_id device, int paramName)
        {
            return getLongs(device, paramName, 1)[0];
        }

        /**
         * Returns the values of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value
         */
        public static long[] getLongs(cl_device_id device, int paramName, int numValues)
        {
            long values[] = new long[numValues];
            clGetDeviceInfo(device, paramName, Sizeof.cl_long * numValues, Pointer.to(values), null);
            return values;
        }

        /**
         * Returns the value of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @return The value
         */
        public static String getString(cl_device_id device, int paramName)
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
        public static String getString(cl_platform_id platform, int paramName)
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

        /**
         * Returns the value of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @return The value
         */
        public static long getSize(cl_device_id device, int paramName)
        {
            return getSizes(device, paramName, 1)[0];
        }

        /**
         * Returns the values of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @param numValues The number of values
         * @return The value
         */
        public static long[] getSizes(cl_device_id device, int paramName, int numValues)
        {
            // The size of the returned data has to depend on
            // the size of a size_t, which is handled here
            ByteBuffer buffer = ByteBuffer.allocate(
                    numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
            clGetDeviceInfo(device, paramName, Sizeof.size_t * numValues,
                    Pointer.to(buffer), null);
            long values[] = new long[numValues];
            if (Sizeof.size_t == 4)
            {
                for (int i=0; i<numValues; i++)
                {
                    values[i] = buffer.getInt(i * Sizeof.size_t);
                }
            }
            else
            {
                for (int i=0; i<numValues; i++)
                {
                    values[i] = buffer.getLong(i * Sizeof.size_t);
                }
            }
            return values;
        }

    }


}
