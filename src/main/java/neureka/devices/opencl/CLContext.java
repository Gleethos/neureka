package neureka.devices.opencl;

import neureka.Component;
import neureka.backend.api.OperationContext;
import org.jocl.cl_platform_id;

import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.clGetPlatformIDs;

public class CLContext  implements Component<OperationContext>
{
    private final List<OpenCLPlatform> _platforms = new ArrayList<>();

    public List<OpenCLPlatform> getPlatforms() { return _platforms; }

    @Override
    public void update( OperationContext oldOwner, OperationContext newOwner ) {
        this._platforms.clear();
        this._platforms.addAll( findAllPlatforms() );
    }

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
