package neureka.core.device.openCL;

import org.jocl.*;

import java.io.*;
import java.util.*;

import static org.jocl.CL.*;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;

public class OpenCLPlatform {

    private cl_platform_id _pid;
    private List<OpenCLDevice> _devices;
    private cl_context _context;

    private Map<String, cl_kernel> _kernels;

    OpenCLPlatform(cl_platform_id pid){
        _pid = pid;
        String platformName = OpenCLDevice.DeviceQuery.getString(pid, CL_PLATFORM_NAME);
        // Obtain the number of devices for the current platform
        int numDevices[] = new int[1];
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 0, null, numDevices);
        System.out.println("Number of devices in platform "+platformName+": "+numDevices[0]);
        cl_device_id devicesArray[] = new cl_device_id[numDevices[0]];
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, numDevices[0], devicesArray, null);

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, pid);

        // Create a context for the selected device
        _context = clCreateContext(
                contextProperties, devicesArray.length, devicesArray,
                null, null, null
        );
        // Collect all devices of this platform
        List<OpenCLDevice> myDevices = new ArrayList<>();

        for(cl_device_id did : devicesArray) {
            OpenCLDevice clDevice = new OpenCLDevice(this , did);
            myDevices.add(clDevice);
        }
        _devices = myDevices;
        _kernels = new HashMap<>();

        //Reading all kernels!
        File curDir = new File("build/resources/main/kernels/");
        File[] filesList = curDir.listFiles();
        //for(File f : filesList){
        //    if(f.isDirectory())
        //        System.out.println(f.getName());
        //    if(f.isFile()){
        //        System.out.println(f.getName());
        //    }
        //}
        String[] sources = new String[filesList.length];
        String[] kernelNames = new String[filesList.length];
        for(int i=0; i<sources.length; i++) {
            sources[i] = _setup.readFile(filesList[i].toString());
            if(sources[i].contains("__kernel")) {
                String[] parts = sources[i].split("__kernel")[1].split("\\(")[0].split(" ");
                kernelNames[i] = parts[parts.length-1];
            }
        }
        // Create the program
        cl_program cpProgram = clCreateProgramWithSource(
                _context,
                sources.length,
                sources,
                null,
                null
        );

        // Build the program
        int err = clBuildProgram(
                cpProgram,
                devicesArray.length,
                devicesArray,
                "-cl-mad-enable", null, null
        );
        //TODO: check compilation errors!

        // Create the kernel
        for(String name : kernelNames){
            if(name!=null){
                _kernels.put(
                        name, clCreateKernel(cpProgram, name, null)
                );
            }
        }
    }

    public cl_platform_id getID(){
        return _pid;
    }

    public List<OpenCLDevice> getDevices(){
        return _devices;
    }

    public Map<String, cl_kernel> getKernels(){
        return _kernels;
    }

    public cl_context getContext(){
        return _context;
    }

    public static List<OpenCLPlatform> PLATFORMS(){

        return _setup.PLATFORMS;
    }

    private static class _setup
    {
        public static List<OpenCLPlatform> PLATFORMS = findAllPlatforms();

        public  static List<OpenCLPlatform> findAllPlatforms()
        {
            // Obtain the number of platforms
            int numPlatforms[] = new int[1];
            clGetPlatformIDs(0, null, numPlatforms);

            // Obtain the platform IDs
            cl_platform_id platforms[] = new cl_platform_id[numPlatforms[0]];
            clGetPlatformIDs(platforms.length, platforms, null);

            List<OpenCLPlatform> list = new ArrayList<>();
            for(cl_platform_id id : platforms){
                list.add(new OpenCLPlatform(id));
            }
            return list;
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





}
