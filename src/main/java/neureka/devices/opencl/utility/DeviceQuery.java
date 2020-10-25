package neureka.devices.opencl.utility;

import static org.jocl.CL.*;

import java.nio.*;
import java.util.*;

import org.jocl.*;

/**
 * A program that queries and prints information about all
 * available devices.
 */
public class DeviceQuery
{
    /**
     * The entry point of this program
     *
     * @return
     */
    public static String query()
    {
        String result = "[DEVICE QUERY]:\n========================================================\n";
        // Obtain the number of platforms
        int[] numPlatforms = new int[ 1 ];
        clGetPlatformIDs(0, null, numPlatforms);

        result+=("Number of platforms: "+numPlatforms[ 0 ]+"\n");

        // Obtain the platform IDs
        cl_platform_id[] platforms = new cl_platform_id[numPlatforms[ 0 ]];
        clGetPlatformIDs(platforms.length, platforms, null);

        // Collect all devices of all platforms
        List<cl_device_id> devices = new ArrayList<cl_device_id>();
        for (cl_platform_id platform : platforms)
        {
            String platformName = getString(platform, CL_PLATFORM_NAME);

            // Obtain the number of devices for the current platform
            int[] numDevices = new int[ 1 ];
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevices);

            result += ("Number of devices in platform " + platformName + ": " + numDevices[ 0 ] + "\n");

            cl_device_id[] devicesArray = new cl_device_id[numDevices[ 0 ]];
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices[ 0 ], devicesArray, null);

            devices.addAll(Arrays.asList(devicesArray));
        }
        result += "========================================================\n";

        // Print the infos about all devices
        for (cl_device_id device : devices)
        {
            // CL_DEVICE_NAME
            String deviceName = getString(device, CL_DEVICE_NAME);
            result+=("\n[Info for device "+deviceName+"]: \n--------------------------------------------------------\n");
            result+=("CL_DEVICE_NAME: "+deviceName+"\n");

            // CL_DEVICE_VENDOR
            String deviceVendor = getString(device, CL_DEVICE_VENDOR);
            result+=("CL_DEVICE_VENDOR: "+deviceVendor+"\n");

            // CL_DRIVER_VERSION
            String driverVersion = getString(device, CL_DRIVER_VERSION);
            result+=("CL_DRIVER_VERSION: "+driverVersion+"\n");

            // CL_DEVICE_TYPE
            long deviceType = getLong(device, CL_DEVICE_TYPE);
            if( (deviceType & CL_DEVICE_TYPE_CPU) != 0) result+=("CL_DEVICE_TYPE: CL_DEVICE_TYPE_CPU\n");
            if( (deviceType & CL_DEVICE_TYPE_GPU) != 0) result+=("CL_DEVICE_TYPE: CL_DEVICE_TYPE_GPU\n");
            if( (deviceType & CL_DEVICE_TYPE_ACCELERATOR) != 0) result+=("CL_DEVICE_TYPE: CL_DEVICE_TYPE_ACCELERATOR\n");
            if( (deviceType & CL_DEVICE_TYPE_DEFAULT) != 0) result+=("CL_DEVICE_TYPE: CL_DEVICE_TYPE_DEFAULT\n");

            // CL_DEVICE_MAX_COMPUTE_UNITS
            int maxComputeUnits = getInt(device, CL_DEVICE_MAX_COMPUTE_UNITS);
            result += ("CL_DEVICE_MAX_COMPUTE_UNITS: "+ maxComputeUnits+"\n");

            // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
            long maxWorkItemDimensions = getLong(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
            result += ("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: "+ maxWorkItemDimensions+"\n");

            // CL_DEVICE_MAX_WORK_ITEM_SIZES
            long[] maxWorkItemSizes = getSizes(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
            result += ("CL_DEVICE_MAX_WORK_ITEM_SIZES: "+maxWorkItemSizes[ 0 ]+", "+ maxWorkItemSizes[ 1 ]+", "+maxWorkItemSizes[ 2 ]+"\n");

            // CL_DEVICE_MAX_WORK_GROUP_SIZE
            long maxWorkGroupSize = getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
            result += ("CL_DEVICE_MAX_WORK_GROUP_SIZE: "+ maxWorkGroupSize+"\n");

            // CL_DEVICE_MAX_CLOCK_FREQUENCY
            long maxClockFrequency = getLong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
            result += ("CL_DEVICE_MAX_CLOCK_FREQUENCY: "+ maxClockFrequency+" MHz\n");

            // CL_DEVICE_ADDRESS_BITS
            int addressBits = getInt(device, CL_DEVICE_ADDRESS_BITS);
            result += ("CL_DEVICE_ADDRESS_BITS: "+ addressBits+"\n");

            // CL_DEVICE_MAX_MEM_ALLOC_SIZE
            long maxMemAllocSize = getLong(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            result+=("CL_DEVICE_MAX_MEM_ALLOC_SIZE: "+ (int)(maxMemAllocSize / (1024 * 1024))+" MByte\n");

            // CL_DEVICE_GLOBAL_MEM_SIZE
            long globalMemSize = getLong(device, CL_DEVICE_GLOBAL_MEM_SIZE);
            result += ("CL_DEVICE_GLOBAL_MEM_SIZE: "+(int)(globalMemSize / (1024 * 1024))+" MByte\n");

            // CL_DEVICE_ERROR_CORRECTION_SUPPORT
            int errorCorrectionSupport = getInt(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT);
            result += ("CL_DEVICE_ERROR_CORRECTION_SUPPORT: "+(errorCorrectionSupport != 0 ? "yes" : "no")+"\n");

            // CL_DEVICE_LOCAL_MEM_TYPE
            int localMemType = getInt(device, CL_DEVICE_LOCAL_MEM_TYPE);
            result += ("CL_DEVICE_LOCAL_MEM_TYPE: "+(localMemType == 1 ? "local" : "global")+"\n");

            // CL_DEVICE_LOCAL_MEM_SIZE
            long localMemSize = getLong(device, CL_DEVICE_LOCAL_MEM_SIZE);
            result += ("CL_DEVICE_LOCAL_MEM_SIZE: "+(int)(localMemSize / 1024)+" KByte\n");

            // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
            long maxConstantBufferSize = getLong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
            result += ("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: "+(int)(maxConstantBufferSize / 1024)+" KByte\n");

            // CL_DEVICE_QUEUE_PROPERTIES
            long queueProperties = getLong(device, CL_DEVICE_QUEUE_PROPERTIES);
            if(( queueProperties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ) != 0)
                result += ("CL_DEVICE_QUEUE_PROPERTIES: CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE\n");
            if(( queueProperties & CL_QUEUE_PROFILING_ENABLE ) != 0)
                result += ("CL_DEVICE_QUEUE_PROPERTIES: CL_QUEUE_PROFILING_ENABLE\n");

            // CL_DEVICE_IMAGE_SUPPORT
            int imageSupport = getInt(device, CL_DEVICE_IMAGE_SUPPORT);
            result += ("CL_DEVICE_IMAGE_SUPPORT: "+imageSupport+"\n");

            // CL_DEVICE_MAX_READ_IMAGE_ARGS
            int maxReadImageArgs = getInt(device, CL_DEVICE_MAX_READ_IMAGE_ARGS);
            result += ("CL_DEVICE_MAX_READ_IMAGE_ARGS: "+maxReadImageArgs+"\n");

            // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
            int maxWriteImageArgs = getInt(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
            result += ("CL_DEVICE_MAX_WRITE_IMAGE_ARGS:  "+maxWriteImageArgs+"\n");

            // CL_DEVICE_SINGLE_FP_CONFIG
            long singleFpConfig = getLong(device, CL_DEVICE_SINGLE_FP_CONFIG);
            result+=("CL_DEVICE_SINGLE_FP_CONFIG: "+stringFor_cl_device_fp_config(singleFpConfig)+"\n");

            // CL_DEVICE_IMAGE2D_MAX_WIDTH
            long image2dMaxWidth = getSize(device, CL_DEVICE_IMAGE2D_MAX_WIDTH);
            result += ("CL_DEVICE_2D_MAX_WIDTH "+image2dMaxWidth+"\n");

            // CL_DEVICE_IMAGE2D_MAX_HEIGHT
            long image2dMaxHeight = getSize(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
            result += ("CL_DEVICE_2D_MAX_HEIGHT "+image2dMaxHeight+"\n");

            // CL_DEVICE_IMAGE3D_MAX_WIDTH
            long image3dMaxWidth = getSize(device, CL_DEVICE_IMAGE3D_MAX_WIDTH);
            result += ("CL_DEVICE_3D_MAX_WIDTH "+image3dMaxWidth+"\n");

            // CL_DEVICE_IMAGE3D_MAX_HEIGHT
            long image3dMaxHeight = getSize(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT);
            result+=("CL_DEVICE_3D_MAX_HEIGHT "+image3dMaxHeight+"\n");

            // CL_DEVICE_IMAGE3D_MAX_DEPTH
            long image3dMaxDepth = getSize(device, CL_DEVICE_IMAGE3D_MAX_DEPTH);
            result += ("CL_DEVICE_3D_MAX_DEPTH "+image3dMaxDepth+"\n");

            // CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
            result += ("CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\n");
            int preferredVectorWidthChar = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
            int preferredVectorWidthShort = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
            int preferredVectorWidthInt = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
            int preferredVectorWidthLong = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
            int preferredVectorWidthFloat = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
            int preferredVectorWidthDouble = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
            result += ("CHAR "+preferredVectorWidthChar+
                            ", SHORT "+preferredVectorWidthShort+
                            ", INT "+preferredVectorWidthInt +
                            ", LONG "+preferredVectorWidthLong+
                            ", FLOAT "+preferredVectorWidthFloat+
                            ", DOUBLE "+ preferredVectorWidthDouble+"\n");
        }
        return result;
    }

    /**
     * Returns the value64 of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value64
     */
    private static int getInt(cl_device_id device, int paramName)
    {
        return getInts(device, paramName, 1)[ 0 ];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    private static int[] getInts(cl_device_id device, int paramName, int numValues)
    {
        int values[] = new int[numValues];
        clGetDeviceInfo(device, paramName, Sizeof.cl_int * numValues, Pointer.to(values), null);
        return values;
    }

    /**
     * Returns the value64 of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private static long getLong(cl_device_id device, int paramName)
    {
        return getLongs(device, paramName, 1)[ 0 ];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    private static long[] getLongs(cl_device_id device, int paramName, int numValues)
    {
        long values[] = new long[numValues];
        clGetDeviceInfo(device, paramName, Sizeof.cl_long * numValues, Pointer.to(values), null);
        return values;
    }

    /**
     * Returns the value64 of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private static String getString(cl_device_id device, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[ 1 ];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[ 0 ]];
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
    private static String getString(cl_platform_id platform, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[ 1 ];
        clGetPlatformInfo(platform, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[ 0 ]];
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
    private static long getSize(cl_device_id device, int paramName)
    {
        return getSizes(device, paramName, 1)[ 0 ];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    static long[] getSizes(cl_device_id device, int paramName, int numValues)
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
                values[ i ] = buffer.getInt(i * Sizeof.size_t);
            }
        }
        else
        {
            for (int i=0; i<numValues; i++)
            {
                values[ i ] = buffer.getLong(i * Sizeof.size_t);
            }
        }
        return values;
    }

}
