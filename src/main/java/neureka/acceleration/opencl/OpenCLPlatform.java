package neureka.acceleration.opencl;

import neureka.Neureka;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.calculus.environment.OperationType;
import org.jocl.*;
import java.util.*;
import neureka.calculus.environment.implementations.*;

import static org.jocl.CL.*;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;

public class OpenCLPlatform {

    private final cl_platform_id _pid;
    private final Map<cl_device_id, OpenCLDevice> _id_device;
    private final cl_context _context;
    private final Map<String, cl_kernel> _kernels;

    private static final Map<String, String> OPERATION_TO_KERNEL_MAPPING = new HashMap<String, String>();

    static {
        OPERATION_TO_KERNEL_MAPPING.put("relu", "activate_relu");
        OPERATION_TO_KERNEL_MAPPING.put("sig", "activate_sigmoid");
        OPERATION_TO_KERNEL_MAPPING.put("quad", "activate_quadratic");
        OPERATION_TO_KERNEL_MAPPING.put("lig", "activate_ligmoid");
        OPERATION_TO_KERNEL_MAPPING.put("idy", "activate_identity");
        OPERATION_TO_KERNEL_MAPPING.put("gaus", "activate_gaussian");
        OPERATION_TO_KERNEL_MAPPING.put("abs", "activate_absolute");
        OPERATION_TO_KERNEL_MAPPING.put("sin", "activate_sinus");
        OPERATION_TO_KERNEL_MAPPING.put("cos", "activate_cosinus");
        //---
        OPERATION_TO_KERNEL_MAPPING.put("sum", "operate_add");
        OPERATION_TO_KERNEL_MAPPING.put("prod", "operate_multiply");
        OPERATION_TO_KERNEL_MAPPING.put("^", "operate_power");
        OPERATION_TO_KERNEL_MAPPING.put("/", "operate_divide");
        OPERATION_TO_KERNEL_MAPPING.put("*", "operate_multiply");
        OPERATION_TO_KERNEL_MAPPING.put("%", "operate_modulo");
        OPERATION_TO_KERNEL_MAPPING.put("-", "operate_subtract");
        OPERATION_TO_KERNEL_MAPPING.put("+", "operate_add");

        OPERATION_TO_KERNEL_MAPPING.put("x", "convolve_multiply");
        //OPERATION_TO_KERNEL_MAPPING.put("x", "convolve_convolve_mul");
        OPERATION_TO_KERNEL_MAPPING.put("d", "convolve_divide");
        OPERATION_TO_KERNEL_MAPPING.put("p", "convolve_power");
        OPERATION_TO_KERNEL_MAPPING.put("a", "convolve_add");
        OPERATION_TO_KERNEL_MAPPING.put("s", "convolve_subtract");

        //"x", ((char)171)+"x", "x"+((char)187),
        //"d", ((char)171)+"d", "d"+((char)187),
        //"p", ((char)171)+"p", "p"+((char)187),
        //"a", ((char)171)+"a", "a"+((char)187),
        //"s", ((char)171)+"s", "s"+((char)187),
        //OPERATION_TO_KERNEL_MAPPING.put((""+((char)171)), "inv_conv_left");
        //OPERATION_TO_KERNEL_MAPPING.put((""+((char)187)), "inv_conv_right");
        OPERATION_TO_KERNEL_MAPPING.put(",", "reshape");

        OPERATION_TO_KERNEL_MAPPING.put("<", "operate_identity");
        OPERATION_TO_KERNEL_MAPPING.put(">", "inject_right");
    }

    public String kernelNameOf(OperationType type) {
        return OPERATION_TO_KERNEL_MAPPING.get(type.identifier());
    }


    private OpenCLPlatform(cl_platform_id pid)
    {
        _id_device = new TreeMap<>(Comparator.comparingInt(NativePointerObject::hashCode));
        _pid = pid;
        // Obtain the number of devices for the current platform
        int[] numDevices = new int[1];
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 0, null, numDevices);
        cl_device_id[] devicesArray = new cl_device_id[numDevices[0]];
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
        for (cl_device_id did : devicesArray) {
            OpenCLDevice clDevice = OpenCLDevice.instance(this, did);
            _id_device.put(did, clDevice);
        }
        _kernels = new HashMap<>();
        _compile(devicesArray);
    }

    public void recompile() {
        List<OpenCLDevice> devices = getDevices();
        cl_device_id[] devicesArray = new cl_device_id[devices.size()];
        for (int i = 0; i < devicesArray.length; i++) devicesArray[i] = devices.get(i).CLDeviceID();
        _compile(devicesArray);
    }

    private void _compile(cl_device_id[] devicesArray)
    {
        //Reading all kernels!
        List<String> templateSources = new ArrayList<>();

        String[] fileNames = {
                "activate_template.cl",
                "broadcast_template.cl",
                "convolve_template.cl",
                "operate_template.cl",
                "scalar_template.cl",
                "utility.cl"
        };
        for(String name : fileNames) {
            templateSources.add(Neureka.instance().utility().readResource("kernels/"+name));
        }
        ArrayList<String> names = new ArrayList<>();
        ArrayList<String> sources = new ArrayList<>();
        for (int i = 0; i < fileNames.length; i++)
        {
            String kernelSource = templateSources.get(i);
            kernelSource = kernelSource.replace(
                    "Neureka.instance().settings().indexing().REVERSE_INDEX_TRANSLATION",
                    (Neureka.instance().settings().indexing().isUsingLegacyIndexing()) ? "true" : "false"
            );
            boolean templateFound = false;
            if (kernelSource.contains("__kernel")) {
                String[] parts = kernelSource.split("__kernel")[1].split("\\(")[0].split(" ");

                templateFound = parts[parts.length - 1].contains("template");
                if (!templateFound) {
                    names.add(parts[parts.length - 1]);
                } else {

                    String preName = parts[parts.length - 1].replace("template", "");

                    // Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
                    // Tsr t0_drain,  Tsr t1_src1,   Tsr t2_src2
                    // drn[di], src1[_i_of_idx_on_tln(prv_src1_cfg, rank)], src2[_i_of_idx_on_tln(prv_src2_cfg, rank)]
                    // default:  src1 o src2 -> drain
                    // inverse:  src1/fdrn <- src2 <- drain
                    //===========================================================================
                    Map<String, String> code = new HashMap<>();
                    for(OperationType type : OperationType.ALL()) {
                        if (preName.contains("activate") && type.supportsImplementation(Activation.class)) {
                            CLExecutor exec = type.getImplementation(Activation.class).getExecutor(CLExecutor.class);
                            if(exec!=null && exec.getSource()!=null) code.put(exec.getName(), exec.getSource());
                        } else if (preName.contains("operate") && type.supportsImplementation(Operation.class)) {
                            CLExecutor exec = type.getImplementation(Operation.class).getExecutor(CLExecutor.class);
                            if(exec!=null && exec.getSource()!=null) code.put(exec.getName(), exec.getSource());
                        } else if (preName.contains("scalar") && type.supportsImplementation(Scalarization.class)) {
                            CLExecutor exec = type.getImplementation(Scalarization.class).getExecutor(CLExecutor.class);
                            if(exec!=null && exec.getSource()!=null) code.put(exec.getName(), exec.getSource());
                        } else if(preName.contains("broadcast") && type.supportsImplementation(Broadcast.class)){//broadcast
                            CLExecutor exec = type.getImplementation(Broadcast.class).getExecutor(CLExecutor.class);
                            if(exec!=null && exec.getSource()!=null) code.put(exec.getName(), exec.getSource());
                        } else if(preName.contains("convolve") && type.supportsImplementation(Convolution.class)) {
                            CLExecutor exec = type.getImplementation(Convolution.class).getExecutor(CLExecutor.class);
                            if(exec!=null && exec.getSource()!=null) code.put(exec.getName(), exec.getSource());
                        }
                    }
                    code.forEach((n, s) -> {
                                names.add(n);
                                sources.add(s);
                            }
                    );
                }
            }
            if (!templateFound) sources.add(kernelSource);
        }

        // Create the program
        cl_program cpProgram = clCreateProgramWithSource(
                _context,
                sources.size(),
                sources.toArray(new String[0]),
                null,
                null
        );

        // Build the program
        int err = clBuildProgram(
                cpProgram,
                devicesArray.length,
                devicesArray,
                "-cl-mad-enable",
                null,
                null
        );
        //TODO: check compilation errors!

        // Create the kernels
        for (String name : names) {
            if (name != null) _kernels.put(name, clCreateKernel(cpProgram, name, null));
        }
    }

    private interface Parser {
        String apply(String name, String first, String second);
    }

    public cl_platform_id getID() {
        return _pid;
    }

    public List<OpenCLDevice> getDevices() {
        List<OpenCLDevice> devices = new ArrayList<>();
        _id_device.forEach((k, v)-> devices.add(v));
        return devices;
    }

    public boolean has(cl_device_id did){
        return _id_device.containsKey(did);
    }
    public OpenCLDevice get(cl_device_id did){
        return _id_device.get(did);
    }
    public void put(cl_device_id did, OpenCLDevice device){
       _id_device.put(did, device);
    }


    public Map<String, cl_kernel> getKernels() {
        return _kernels;
    }

    public cl_context getContext() {
        return _context;
    }

    public static List<OpenCLPlatform> PLATFORMS() {
        return _setup.PLATFORMS;
    }

    private static class _setup
    {
        public static List<OpenCLPlatform> PLATFORMS = findAllPlatforms();

        public static List<OpenCLPlatform> findAllPlatforms() {
            // Obtain the number of platforms
            int[] numPlatforms = new int[1];
            clGetPlatformIDs(0, null, numPlatforms);

            // Obtain the platform IDs
            cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
            clGetPlatformIDs(platforms.length, platforms, null);

            List<OpenCLPlatform> list = new ArrayList<>();
            for (cl_platform_id id : platforms) {
                list.add(new OpenCLPlatform(id));
            }
            return list;
        }

    }


}
