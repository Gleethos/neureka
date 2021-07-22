package neureka.devices.opencl;

import neureka.Neureka;
import neureka.backend.api.ImplementationFor;
import neureka.backend.api.Operation;
import neureka.backend.standard.algorithms.*;
import neureka.backend.standard.implementations.CLImplementation;
import org.jocl.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.jocl.CL.*;

public class OpenCLPlatform {

    private final static Logger _LOG = LoggerFactory.getLogger( OpenCLPlatform.class );

    private final cl_platform_id _pid;
    private final cl_context _context;
    private boolean _isDoingLegacyIndexing = false;

    private final Map<cl_device_id, OpenCLDevice> _id_device;
    private final Map<String, cl_kernel> _kernels = new HashMap<>();


    private OpenCLPlatform( cl_platform_id pid )
    {
        _id_device = new TreeMap<>(Comparator.comparingInt(NativePointerObject::hashCode));
        _pid = pid;
        // Obtain the number of devices for the current platform
        int[] numDevices = new int[ 1 ];
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 0, null, numDevices);
        cl_device_id[] devicesArray = new cl_device_id[numDevices[ 0 ]];
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, numDevices[ 0 ], devicesArray, null);

        // Enable exceptions and subsequently omit error checks in this sample
        setExceptionsEnabled( true );

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
            OpenCLDevice clDevice = OpenCLDevice.newInstanceOf( this, did );
            _id_device.put(did, clDevice);
        }
        _compile(devicesArray);
    }

    public void recompile() {
        List<OpenCLDevice> devices = getDevices();
        cl_device_id[] devicesArray = new cl_device_id[devices.size()];
        for ( int i = 0; i < devicesArray.length; i++) devicesArray[ i ] = devices.get( i ).getDeviceId();
        _compile(devicesArray);
    }

    private void _compile(cl_device_id[] devicesArray)
    {
        //Reading all kernels!
        List<String> templateSources = new ArrayList<>();

        String[] fileNames = {
                "activation_template.cl",
                "broadcast_template.cl",
                "convolution_template.cl",
                "operator_template.cl",
                "scalarization_template.cl",
                "utility.cl"
        };
        for ( String name : fileNames )
            templateSources.add(Neureka.get().utility().readResource("kernels/"+name));

        ArrayList<String> names = new ArrayList<>();
        ArrayList<String> sources = new ArrayList<>();
        for ( int i = 0; i < fileNames.length; i++ )
        {
            String kernelSource = templateSources.get( i );
            boolean templateFound = false;
            if ( kernelSource.contains( "__kernel" ) )
            {
                String[] parts = kernelSource.split("__kernel")[ 1 ].split("\\(")[ 0 ].split(" ");

                templateFound = parts[parts.length - 1].contains("template");
                if ( !templateFound ) names.add(parts[parts.length - 1]);
                else
                {
                    String preName = parts[ parts.length - 1 ].replace("template", "");
                    // Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
                    // Tsr t0_drain,  Tsr t1_src1,   Tsr t2_src2
                    // drn[di], src1[_i_of_idx_on_tln(prv_src1_cfg, rank)], src2[_i_of_idx_on_tln(prv_src2_cfg, rank)]
                    // default:  src1 o src2 -> drain
                    // inverse:  src1/fdrn <- src2 <- drain
                    //===========================================================================
                    Map<String, String> code = new HashMap<>();
                    ImplementationFor<OpenCLDevice> impl = null;
                    for ( Operation type : Neureka.get().context().instances() ) {
                        if ( preName.contains("activation") && type.supportsAlgorithm(Activation.class) ) {
                            impl = type.getAlgorithm(Activation.class).getImplementationFor( OpenCLDevice.class );
                        } else if ( preName.contains("operator") && type.supportsAlgorithm(Operator.class) ) {
                            impl = type.getAlgorithm(Operator.class).getImplementationFor( OpenCLDevice.class );
                        } else if ( preName.contains("scalarization") && type.supportsAlgorithm(Scalarization.class) ) {
                            impl = type.getAlgorithm(Scalarization.class).getImplementationFor( OpenCLDevice.class );
                        } else if ( preName.contains("broadcast") && type.supportsAlgorithm(Broadcast.class) ) {//broadcast
                            impl = type.getAlgorithm(Broadcast.class).getImplementationFor( OpenCLDevice.class );
                        } else if ( preName.contains("convolution") && type.supportsAlgorithm(Convolution.class) ) {
                            impl = type.getAlgorithm(Convolution.class).getImplementationFor( OpenCLDevice.class );
                        } else if (
                                type.supportsAlgorithm(GenericAlgorithm.class)
                                && preName.contains(type.getAlgorithm(GenericAlgorithm.class).getName())
                        ) { // TODO: cover!
                            impl = type.getAlgorithm(GenericAlgorithm.class).getImplementationFor( OpenCLDevice.class );
                        }
                        if ( impl instanceof CLImplementation ) {
                            CLImplementation clImplementation = (CLImplementation) impl;
                            if ( clImplementation.getSource() != null )
                                code.put( clImplementation.getName(), clImplementation.getSource() );
                        }
                    }
                    code.forEach( ( n, s ) -> {
                                names.add( n );
                                sources.add( s );
                            }
                    );
                }
            }
            if ( !templateFound ) sources.add( kernelSource );
        }

        // Create the program
        cl_program cpProgram = clCreateProgramWithSource(
                _context,
                sources.size(),
                sources.toArray( new String[ 0 ] ),
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
        if ( err != CL_SUCCESS )
            _LOG.error("Failed to compile the OpenCL code of the current context. Error code: '"+err+"'.");

        //TODO: check compilation errors!

        // Create the kernels
        for ( String name : names ) {
            if ( name != null ) _kernels.put( name, clCreateKernel( cpProgram, name, null ) );
        }
    }

    public List<OpenCLDevice> getDevices() {
        List<OpenCLDevice> devices = new ArrayList<>();
        _id_device.forEach( ( k, v ) -> devices.add( v ) );
        return devices;
    }

    public boolean has( cl_device_id did ) {
        return _id_device.containsKey( did );
    }

    public OpenCLDevice get( cl_device_id did ) {
        return _id_device.get( did );
    }

    public void put( cl_device_id did, OpenCLDevice device ) {
       _id_device.put( did, device );
    }

    public Map<String, cl_kernel> getKernels() {
        return _kernels;
    }


    public static List<OpenCLPlatform> PLATFORMS() {
        return Setup.PLATFORMS;
    }

    public cl_platform_id getPid() {
        return this._pid;
    }

    public cl_context getContext() {
        return this._context;
    }

    public boolean isDoingLegacyIndexing() {
        return this._isDoingLegacyIndexing;
    }

    private static class Setup
    {
        public static List<OpenCLPlatform> PLATFORMS = findAllPlatforms();

        public static List<OpenCLPlatform> findAllPlatforms()
        {
            // Obtain the number of platforms
            int[] numPlatforms = new int[ 1 ];
            clGetPlatformIDs(0, null, numPlatforms);

            // Obtain the platform IDs
            cl_platform_id[] platforms = new cl_platform_id[ numPlatforms[ 0 ] ];
            clGetPlatformIDs( platforms.length, platforms, null );

            List<OpenCLPlatform> list = new ArrayList<>();
            for ( cl_platform_id id : platforms ) list.add( new OpenCLPlatform( id ) );
            return list;
        }

    }


}
