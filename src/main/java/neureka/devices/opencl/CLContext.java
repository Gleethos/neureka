package neureka.devices.opencl;

import neureka.Component;
import neureka.backend.api.OperationContext;
import org.jocl.cl_platform_id;

import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.clGetPlatformIDs;

/**
 *  This is an OpenCL context component for any given {@link OperationContext}.
 *  {@link OperationContext}s are thread local states
 *  used for managing {@link neureka.backend.api.Operation}, {@link neureka.calculus.Function}
 *  as well as {@link Component<OperationContext>} implementation instances like this one.
 *  This component system exist so that a given context can be extended for more functionality
 *  and also to attach relevant states like for example kernels, memory objects and other concepts
 *  exposed by OpenCL.
 *  This state might not be compatible with the concepts introduced in other contexts
 *  which is why it makes sense to have separate "worlds".
 */
public class CLContext  implements Component<OperationContext>
{
    private final List<OpenCLPlatform> _platforms = new ArrayList<>();

    public List<OpenCLPlatform> getPlatforms() { return _platforms; }

    @Override
    public void update( OperationContext oldOwner, OperationContext newOwner ) {
        this._platforms.clear();
        this._platforms.addAll( _findLoadAndCompileForAllPlatforms() );
    }

    private static List<OpenCLPlatform> _findLoadAndCompileForAllPlatforms()
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
