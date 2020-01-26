package neureka.acceleration.opencl;

import neureka.Neureka;
import neureka.calculus.Function;
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

    private static Map<String, String> OPERATION_TO_KERNEL_MAPPING = new HashMap<String, String>();

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

    public String kernelNameOf(int f_id) {
        String name = OPERATION_TO_KERNEL_MAPPING.get(Function.TYPES.REGISTER[f_id]);
        //System.out.println("Kernel needed: "+name);
        return OPERATION_TO_KERNEL_MAPPING.get(Function.TYPES.REGISTER[f_id]);
    }

    OpenCLPlatform(cl_platform_id pid) {
        _pid = pid;
        // Obtain the number of devices for the current platform
        int numDevices[] = new int[1];
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 0, null, numDevices);
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

        for (cl_device_id did : devicesArray) {
            OpenCLDevice clDevice = new OpenCLDevice(this, did);
            myDevices.add(clDevice);
        }
        _devices = myDevices;
        _kernels = new HashMap<>();

        _compile(devicesArray);

    }

    public void recompile() {
        cl_device_id[] devicesArray = new cl_device_id[_devices.size()];
        for (int i = 0; i < devicesArray.length; i++) {
            devicesArray[i] = _devices.get(i).CLDeviceID();
        }
        _compile(devicesArray);
    }

    private void _compile(cl_device_id[] devicesArray)
    {
        //Reading all kernels!
        List<String> templateSources = new ArrayList<>();

        String[] fileNames  = {
                "activate_template.cl",
                "broadcast_template.cl",
                "convolve_template.cl",
                "operate_template.cl",
                "scalar_template.cl",
                "utility.cl"
        };
        for(String name : fileNames){
            InputStream inputStream = getClass().getClassLoader().getResourceAsStream("kernels/"+name);
            templateSources.add(_setup.readFile(inputStream));
        }
        ArrayList<String> names = new ArrayList<>();
        ArrayList<String> sources = new ArrayList<>();
        for (int i = 0; i < fileNames.length; i++) {
            String kernelSource = templateSources.get(i);//_setup.readFile(filesList[i]);
            kernelSource = kernelSource.replace(
                    "Neureka.Settings.Indexing.REVERSE_INDEX_TRANSLATION",
                    (Neureka.Settings.Indexing.legacy()) ? "true" : "false"
            );
            boolean templateFound = false;
            if (kernelSource.contains("__kernel")) {
                String[] parts = kernelSource.split("__kernel")[1].split("\\(")[0].split(" ");

                templateFound = parts[parts.length - 1].contains("template");
                if (!templateFound) {
                    names.add(parts[parts.length - 1]);
                } else {
                    convolve_kernels_of(parts[parts.length - 1], kernelSource)
                            .forEach((n, s) -> {
                                names.add(n);
                                sources.add(s);
                            }
                    );
                }
            }
            if (!templateFound) {
                sources.add(kernelSource);
            }
        }

        // Create the program
        cl_program cpProgram = clCreateProgramWithSource(
                _context,
                sources.size(),//kernelSources.length,
                sources.toArray(new String[sources.size()]),
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

        // Create the kernels
        for (String name : names) {
            if (name != null) {
                _kernels.put(name, clCreateKernel(cpProgram, name, null));
            }
        }
    }

    private interface Parser {
        void apply(String name, String first, String second, boolean advanced);
    }

    private Map<String, String> convolve_kernels_of(String name, String source) {
        Map<String, String> code = new HashMap<>();
        String newName = name.replace("template", "");
        source = source.replace("template", "");
        String[] parts = source.split("//-=<OPERATION>=-//");

        java.util.function.Function<String, String> correct = (s) ->
                "//-=<PARSED>=-//\n" +
                        s.replace("src1", "src1[_i_of_idx_on_tln(prv_src1_cfg, rank)]")
                                .replace("src2", "src2[_i_of_idx_on_tln(prv_src2_cfg, rank)]")
                                .replace("input1", "src1[_i_of_i(i, prv_src1_cfg, rank)]")
                                .replace("input2", "src2[_i_of_i(i, prv_src2_cfg, rank)]")
                                .replace("input", "src1[_i_of_i(i, prv_src1_cfg, rank)]")
                                .replace("output", "drn[_i_of_i(i, prv_drn_cfg, rank)]")
                                .replace("handle", "src1[_i_of_idx_on_tln(prv_src1_cfg, rank)]")
                                .replace("drain", "src2[_i_of_idx_on_tln(prv_src2_cfg, rank)]")
                                .replace("origin", "drn[di]")
                                .replace("target", "frn[_i_of_idx_on_tln(prv_frn_cfg, rank)]") +
                        "\n//-=<PARSED>=-//";

        java.util.function.Function<String, String> asAdvanced = (s) ->
                s.replace("target", "frn[_i_of_idx_on_tln(prv_frn2_cfg, rank)]")
                        .replace("//-=<ARGUMENT>=-//", "")
                        .replace("//-=<CONFIGURATION>=-//", "");

        Parser parser = (n, f, s, advanced) -> {
            String convcode =
                    parts[0].replace(newName, newName + n) +
                            correct.apply(f) +
                            parts[2] +
                            correct.apply(s) +
                            parts[4];
            convcode = (advanced) ? asAdvanced.apply(convcode) : convcode;
            code.put(newName + n, convcode);
        };
        //Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        //Tsr t0_drain,  Tsr t1_src1,   Tsr t2_src2
        //drn[di], src1[_i_of_idx_on_tln(prv_src1_cfg, rank)], src2[_i_of_idx_on_tln(prv_src2_cfg, rank)]
        //default:  src1 o src2 -> drain
        //inverse:  src1/fdrn <-src2 <- drain
        //===========================================================================
        if (newName.contains("activate")) {
            parser.apply(
                    "cosinus",
                    "drn[_i_of_i(i, prv_drn_cfg, rank)] = cos(input);\n",
                    "drn[_i_of_i(i, prv_drn_cfg, rank)] = -sin(input);\n",
                    false
            );
            parser.apply(
                    "gaussian",
                    "output =\n" +
                            "                (float)pow(\n" +
                            "                    (float)M_E,\n" +
                            "                    -(float)pow(\n" +
                            "                        (float)input,\n" +
                            "                        (float)2\n" +
                            "                    )\n" +
                            "                );\n",
                    "output = 1 / (1 + (float)pow((float)M_E, -input));\n",
                    false
            );
            parser.apply(
                    "ligmoid",
                    "output = (\n" +
                            "                    (float) log(\n" +
                            "                        1+pow(\n" +
                            "                            (float)\n" +
                            "                            M_E,\n" +
                            "                            (float)\n" +
                            "                            input\n" +
                            "                        )\n" +
                            "                    )\n" +
                            "            );",
                    "output =\n" +
                            "                    1 /\n" +
                            "                            (1 + (float)\n" +
                            "                                pow(\n" +
                            "                                    (float)\n" +
                            "                                    M_E,\n" +
                            "                                    (float)\n" +
                            "                                    input\n" +
                            "                                )\n" +
                            "                            );\n",
                    false
            );
            parser.apply(
                    "sigmoid",
                    "output = 1 / (1 + (float)pow((float)M_E, -input));\n",
                    "output = input * (1 - input);\n",
                    false
            );
            parser.apply(
                    "sinus",
                    "output = sin(input);\n",
                    "output = cos(input);\n",
                    false
            );
            parser.apply(
                    "relu",
                    "if (input >= 0) {  output = input; } else { output = input * (float)0.01; }\n",
                    "if (input >= 0) { output = (float)1; } else { output = (float)0.01; }\n",
                    false
            );
            parser.apply(
                    "identity",
                    "output = input;\n",
                    "output = input;\n",
                    false
            );
            parser.apply(
                    "tanh",
                    "output = input/pow(1+pow(input, 2.0f), 0.5f);\n",
                    "output = 1-pow(input/pow((1.0f+pow(input,2.0f)),0.5f), 2.0f);\n",
                    false
            );

        } else if (newName.contains("operate")) {
            parser.apply(
                    "multiply",
                    "output = input1 * input2;\n",
                    "if(d==0){output = input2;}else{output = input1;}\n",
                    false
            );
            parser.apply(
                    "add",
                    "output = input1 + input2;\n",
                    "output = 1;\n",
                    false
            );
            parser.apply(
                    "subtract",
                    "output = input1 - input2;\n",
                    "if(d==0){\n" +//drn and src2 switch:
                            "    output = 1;\n" +
                            "} else {\n" +
                            "    output = -1;" +
                            "}",
                    false
            );
            parser.apply(
                    "divide",
                    "output = input1 / input2;\n",
                    "if(d==0){\n" +
                            "    output = 1/input2;\n" +
                            "} else {\n" +
                            "    output = -input2 /(float)pow(input1, 2.0f);\n" +
                            "}",
                    true
            );
            parser.apply(
                    "power",
                    "output = pow(input1, input2);",
                    "if(d==0){\n" +
                            "    output = input2 * pow(input1, input2-1.0f);\n" +
                            "} else {\n" +
                            "    output = pow(input1, input2) * log(input1);\n" +
                            "}",
                    true
            );
        } else if (newName.contains("scalar")) {
            parser.apply(
                    "multiply",
                    "output = input1 * value;\n",
                    "if(d==0){output = value;}else{output = input1;}\n",
                    false
            );
            parser.apply(
                    "add",
                    "output = input1 + value;\n",
                    "output = 1;\n",
                    false
            );
            parser.apply(
                    "subtract",
                    "output = input1 - value;\n",
                    "if(d==0){\n" +//drn and src2 switch:
                            "    output = 1;\n" +
                            "} else {\n" +
                            "    output = -1;" +
                            "}",
                    false
            );
            parser.apply(
                    "divide",
                    "output = input1 / value;\n",
                    "if(d==0){\n" +
                            "    output = 1/value;\n" +
                            "} else {\n" +
                            "    output = -value /(float)pow(input1, 2.0f);\n" +
                            "}",
                    true
            );
            parser.apply(
                    "power",
                    "output = pow(input1, value);",
                    "if(d==0){\n" +
                            "    output = (value * pow(input1, value-(float)1 ));\n" +
                            "} else {\n" +
                            "    output = (pow(input1, value) * log(value));\n" +
                            "}",
                    true
            );
            parser.apply(
                    "identity",
                    "output = value;\n",
                    "output = value;\n",
                    true
            );
        } else {// broadcast / convolve:
            parser.apply(
                    "multiply",
                    "value = src1 * src2;\n",
                    "value += handle * drain;\n",
                    false
            );
            parser.apply(
                    "add",
                    "value = src1 + src2;\n",
                    "value += 1 * drain;\n",
                    false
            );
            parser.apply(
                    "subtract",
                    "value = src1 - src2;\n",
                    "if(d==0){\n" +//drn and src2 switch:
                            "    value += 1 * drain;\n" +
                            "} else {\n" +
                            "    value += -1 * drain;" +
                            "}",
                    false
            );
            parser.apply(
                    "divide",
                    "value = src1 / src2;\n",
                    "if(d==0){\n" +
                            "    value += (1/handle) * drain;\n" +
                            "} else {\n" +
                            "    value += (-(handle /(float)pow(target, (float)2)) ) * drain;\n" +
                            "}",
                    true
            );
            parser.apply(
                    "power",
                    "value += pow(src1, src2);",
                    "if(d==0){\n" +
                            "    value = (handle * pow(target, handle-(float)1 )) * drain;\n" +
                            "} else {\n" +
                            "    value += (pow(target, handle) * log(handle)) * drain;\n" +
                            "}",
                    true
            );
            //code.forEachForward((k, v)->{System.out.println(k+"\n------------\n"+v);});
        }

        return code;
    }

    public cl_platform_id getID() {
        return _pid;
    }

    public List<OpenCLDevice> getDevices() {
        return _devices;
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

    private static class _setup {
        public static List<OpenCLPlatform> PLATFORMS = findAllPlatforms();

        public static List<OpenCLPlatform> findAllPlatforms() {
            // Obtain the number of platforms
            int numPlatforms[] = new int[1];
            clGetPlatformIDs(0, null, numPlatforms);

            // Obtain the platform IDs
            cl_platform_id platforms[] = new cl_platform_id[numPlatforms[0]];
            clGetPlatformIDs(platforms.length, platforms, null);

            List<OpenCLPlatform> list = new ArrayList<>();
            for (cl_platform_id id : platforms) {
                list.add(new OpenCLPlatform(id));
            }
            return list;
        }

        /**
         * Helper function which reads the file with the given name and returns
         * the contents of this file as a String. Will exit the application
         * if the file can not be read.
         *
         * @param
         * @return The contents of the file
         */
        private static String readFile(InputStream stream){//String fileName) {
            try {
                BufferedReader br = new BufferedReader(new InputStreamReader(stream));//new FileInputStream(fileName)));
                StringBuffer sb = new StringBuffer();
                String line = null;
                while (true) {
                    line = br.readLine();
                    if (line == null) break;
                    sb.append(line).append("\n");
                }
                return sb.toString();
            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
                return null;
            }
        }
    }


}
