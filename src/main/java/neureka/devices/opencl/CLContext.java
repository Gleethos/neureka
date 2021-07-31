package neureka.devices.opencl;

import neureka.Component;
import neureka.Tsr;
import neureka.backend.api.OperationContext;
import org.jocl.cl_platform_id;

import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.clGetPlatformIDs;

/**
 *  This is an OpenCL context component for any given {@link OperationContext} which
 *  extends a given operation context instance for additional functionality, which in
 *  this case is the OpenCL backend storing platform and device information.
 *  {@link OperationContext}s are thread local states
 *  used for managing {@link neureka.backend.api.Operation}, {@link neureka.calculus.Function}
 *  as well as {@link Component<OperationContext>} implementation instances like this one.
 *  A given state might not be compatible with the concepts introduced in other contexts
 *  which is why it makes sense to have separate "worlds" with potential different operations...
 *  The component system of the {@link OperationContext} exist so that a given context
 *  can be extended for more functionality
 *  and also to attach relevant states like for example in this case the {@link CLContext}
 *  instance will directly or indirectly reference kernels, memory objects and other concepts
 *  exposed by OpenCL...
 */
public class CLContext  implements Component<OperationContext>
{
    private final List<OpenCLPlatform> _platforms = new ArrayList<>();

    /**
     *  Use this constructor if you want to create a new OpenCL world in which there
     *  are unique {@link OpenCLPlatform} and {@link OpenCLDevice} instances.
     */
    public CLContext() {}

    /**
     * @return A list of context specific {@link OpenCLPlatform} instances possible containing {@link OpenCLDevice}s.
     */
    public List<OpenCLPlatform> getPlatforms() { return _platforms; }

    /**
     *  Updating the CLContext will cause the list of existing {@link OpenCLPlatform} instances to be
     *  cleared and refilled with completely new {@link OpenCLPlatform} instances.
     *  This will in effect also cause the recreation of any {@link {@link OpenCLDevice} instances
     *  as part of these {@link OpenCLPlatform}s.
     *  This will subsequently cause the recompilation of many OpenCL kernels.
     *
     * @param oldOwner The previous {@link OperationContext} instance or null if the component is being added to the owner.
     * @param newOwner The new {@link OperationContext} instance.
     */
    @Override
    public void update( OwnerChangeRequest<OperationContext> changeRequest ) {
        this._platforms.clear();
        this._platforms.addAll( _findLoadAndCompileForAllPlatforms() );
        changeRequest.executeChange();
    }

    /**
     * @return A new list of freshly created {@link OpenCLPlatform} instances containing freshly instantiated {@link OpenCLDevice}s and kernels.
     */
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
