package neureka.devices.opencl;

import neureka.Neureka;
import neureka.backend.api.ImplementationFor;
import neureka.backend.api.Algorithm;
import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.Operation;
import neureka.backend.ocl.CLBackend;
import neureka.backend.main.algorithms.*;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.implementations.SimpleCLImplementation;
import org.jocl.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

import static org.jocl.CL.*;

/**
 *  This class models the OpenCL concept of platforms, which refer to device
 *  vendors / or vendor OpenCL runtime drivers.
 *  For example, in a system with 1 Intel CPU, 1 Nvidia GPUs and 2 AMD GPU,
 *  you will have 3 OpenCL platforms exposed by the OpenCL API, one for Intel,
 *  one for Nvidia and one for AMD. With an AMD CPU and AMD GPU,
 *  you will have a single Platform for both.
 *  Same with Intel CPU and Intel GPU/FPGA, also 1 Platform only.
 *
 *  Here an example for an exception to the Platforms=vendors rule:
 *  There is 1 Intel CPU in the system,
 *  but the Intel OpenCL runtime and also POCL OpenCL runtime installed.
 *  Then you have 2 Platforms (Intel and POCL),
 *  each with the same Intel CPU as device.
 *
 *  For every platform exposed by the OpenCL runtime (modelled by a {@link CLBackend} instance),
 *  there will be a {@link OpenCLPlatform} instance.
 */
public class OpenCLPlatform
{
    private final static Logger _LOG = LoggerFactory.getLogger( OpenCLPlatform.class );

    private final cl_platform_id _pid;
    private final cl_context _context;

    private final Map<cl_device_id, OpenCLDevice> _id_device;
    private final Map<String, cl_kernel> _kernels = new HashMap<>();


    public OpenCLPlatform(cl_platform_id pid)
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

        List<cl_device_id> successfullyLoaded = new ArrayList<>();

        List<String> failures = new ArrayList<>();
        // Collect all devices of this platform
        for (cl_device_id did : devicesArray) {
            try {
                OpenCLDevice clDevice = OpenCLDevice.of(this, did);
                _id_device.put(did, clDevice);
                successfullyLoaded.add(did);
            } catch ( Exception e ) {
                String message =
                        "Failed to create '"+OpenCLDevice.class.getSimpleName()+"' instance for " +
                        "OpenCL device id '0x" + Long.toHexString(did.getNativePointer()) + "' under platform id '0x"+Long.toHexString(pid.getNativePointer())+"'!";
                _LOG.error(message, e);
                failures.add(message + " Reason: " + e.getMessage());
            }
        }
        if ( !successfullyLoaded.isEmpty() )
            _compile(successfullyLoaded.toArray(new cl_device_id[0]));
        else
            _LOG.warn(
                "'"+this.getClass().getSimpleName()+"' with id '"+Long.toHexString(pid.getNativePointer())+"' does not have a valid device attached to it!"
            );

        if ( successfullyLoaded.isEmpty() && devicesArray.length > 0 )
            throw new RuntimeException(
                "Failed to create '"+OpenCLDevice.class.getSimpleName()+"' instances for all devices of platform id '0x"+Long.toHexString(pid.getNativePointer())+"'! \n" +
                "Reasons: \n    " + failures.stream().collect(Collectors.joining("\n    "))
            );
    }

    public void recompile() {
        List<OpenCLDevice> devices = getDevices();
        cl_device_id[] devicesArray = new cl_device_id[devices.size()];
        for ( int i = 0; i < devicesArray.length; i++) devicesArray[ i ] = devices.get( i ).getId();
        _compile(devicesArray);
    }

    /**
     *   This is where all the kernels defined by all the {@link CLImplementation}
     *   in the standard backend, will be compiled to OpenCL programs.
     *   These kernels are usually based on pre-made template kernel source files...
     *   They are supposed to be as general purpose as possible, meaning they use
     *   a rather complicated indexing mechanism (see 'utility.cl').
     *
     * @param devicesArray The array of devices for which kernels should be compiled.
     */
    private void _compile( cl_device_id[] devicesArray )
    {
        //Reading all kernels!
        List<String> templateSources = new ArrayList<>();

        String[] fileNames = {
                "activation_template.cl",
                "broadcast_template.cl",
                "convolution_template.cl",
                "elementwise_template.cl",
                "scalarization_template.cl",
                "scalar_broadcast.cl",
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
                    for ( Operation type : Neureka.get().backend().getOperations() ) {
                        if ( preName.contains("activation") && type.supportsAlgorithm(Activation.class) )
                            impl = type.getAlgorithm(Activation.class).getImplementationFor( OpenCLDevice.class );
                        else if ( preName.contains("elementwise") && type.supportsAlgorithm(BiElementWise.class) )
                            impl = type.getAlgorithm(BiElementWise.class).getImplementationFor( OpenCLDevice.class );
                        else if ( preName.contains("scalarization") && type.supportsAlgorithm(Scalarization.class) )
                            impl = type.getAlgorithm(Scalarization.class).getImplementationFor( OpenCLDevice.class );
                        else if ( preName.contains("broadcast") && type.supportsAlgorithm(Broadcast.class) )
                            impl = type.getAlgorithm(Broadcast.class).getImplementationFor( OpenCLDevice.class );
                        else if ( preName.contains("convolution") && type.supportsAlgorithm(NDConvolution.class) )
                            impl = type.getAlgorithm(NDConvolution.class).getImplementationFor( OpenCLDevice.class );
                        else if (
                                type.supportsAlgorithm(DeviceAlgorithm.class)
                                &&
                                preName.contains(type.getAlgorithm(DeviceAlgorithm.class).getName())
                        ) { // TODO: cover!
                            impl = type.getAlgorithm(DeviceAlgorithm.class).getImplementationFor( OpenCLDevice.class );
                        }
                        if ( impl instanceof CLImplementation ) {
                            KernelCode kernelCode = ((CLImplementation) impl).getKernelCode();
                            if ( kernelCode.getCode() != null )
                                code.put( kernelCode.getName(), kernelCode.getCode() );
                        }
                    }
                    code.forEach( ( n, s ) -> { names.add( n ); sources.add( s ); } );
                }
            }
            if ( !templateFound ) sources.add( kernelSource );
        }

        for ( Operation type : Neureka.get().backend().getOperations() ) {
            for ( Algorithm algorithm : type.getAllAlgorithms()) {
                DeviceAlgorithm<?> deviceAlgorithm = ( algorithm instanceof DeviceAlgorithm ? ((DeviceAlgorithm<?>) algorithm) : null );
                ImplementationFor<OpenCLDevice> impl =  ( deviceAlgorithm == null ? null : deviceAlgorithm.getImplementationFor(OpenCLDevice.class) );
                if ( impl instanceof CLImplementation ) {
                    CLImplementation cli = ((CLImplementation) impl);
                    if ( cli instanceof SimpleCLImplementation ) {
                        names.add(cli.getKernelCode().getName());
                        sources.add(cli.getKernelCode().getCode());
                    }
                }
            }
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
        for ( String name : names )
            if ( name != null ) _kernels.put( name, clCreateKernel( cpProgram, name, null ) );
    }

    public List<OpenCLDevice> getDevices() {
        List<OpenCLDevice> devices = new ArrayList<>();
        _id_device.forEach( ( k, v ) -> devices.add( v ) );
        return devices;
    }

    /**
     * @param did The {@link cl_device_id} representing an OpenCL supporting device.
     * @return The truth value determining if this platform hosts the device represented by the provided id.
     */
    public boolean has( cl_device_id did ) { return _id_device.containsKey( did ); }

    public OpenCLDevice get( cl_device_id did ) {
        return _id_device.get( did );
    }

    void put( cl_device_id did, OpenCLDevice device ) {
       _id_device.put( did, device );
    }

    public cl_kernel getKernel( String kernelName ) {
        return _kernels.get( kernelName );
    }

    public boolean hasKernel( String kernelName ) {
        return _kernels.containsKey( kernelName );
    }

    public final long getId() { return _pid.getNativePointer(); }

    public cl_context getContext() {
        return _context;
    }

    public void dispose() {
        clReleaseContext( _context );
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName()+"@"+Integer.toHexString(hashCode())+"[" +
                    "id=0x" + Long.toHexString(_pid.getNativePointer()) + "," +
                    "context=0x"+Long.toHexString(_context.getNativePointer()) + "," +
                    "kernels=[.."+_kernels.size()+"..]" +
                "]";
    }

}
