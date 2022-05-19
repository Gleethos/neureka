package neureka.devices.opencl;

import neureka.backend.api.BackendContext;
import neureka.backend.api.BackendExtension;
import neureka.backend.api.Extensions;
import neureka.calculus.assembly.ParseUtil;
import neureka.common.composition.Component;
import neureka.devices.Device;
import neureka.devices.opencl.utility.Messages;
import org.jocl.cl_platform_id;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.jocl.CL.clGetPlatformIDs;

/**
 *  This is an OpenCL context component for any given {@link BackendContext} which
 *  extends a given backend context instance for additional functionality, which in
 *  this case is the OpenCL backend storing platform and device information.
 *  {@link BackendContext}s are thread local states
 *  used for managing {@link neureka.backend.api.Operation}, {@link neureka.calculus.Function}
 *  as well as {@link Component} implementation instances like this one.
 *  A given state might not be compatible with the concepts introduced in other contexts
 *  which is why it makes sense to have separate "worlds" with potential different operations...
 *  The component system of the {@link BackendContext} exist so that a given context
 *  can be extended for more functionality
 *  and also to attach relevant states like for example in this case the {@link CLContext}
 *  instance will directly or indirectly reference kernels, memory objects and other concepts
 *  exposed by OpenCL...
 */
public final class CLContext implements BackendExtension
{
    private static Logger _LOG = LoggerFactory.getLogger(CLContext.class);

    private final List<OpenCLPlatform> _platforms = new ArrayList<>();
    private final CLSettings _settings = new CLSettings();

    /**
     *  Use this constructor if you want to create a new OpenCL world in which there
     *  are unique {@link OpenCLPlatform} and {@link OpenCLDevice} instances.
     */
    public CLContext() {}

    /**
     * @return The number of all {@link OpenCLDevice} instances across all {@link OpenCLPlatform}s.
     */
    public int getTotalNumberOfDevices() {
        List<OpenCLPlatform> platforms = getPlatforms();
        if ( getPlatforms().isEmpty() ) return 0;
        return platforms.stream().mapToInt( p -> p.getDevices().size() ).sum();
    }

    /**
     * @return A list of context specific {@link OpenCLPlatform} instances possible containing {@link OpenCLDevice}s.
     */
    public List<OpenCLPlatform> getPlatforms() { return Collections.unmodifiableList( _platforms ); }

    /**
     * @return A container for OpenCL specific settings.
     */
    public CLSettings getSettings() { return _settings; }

    /**
     *  Updating the CLContext will cause the list of existing {@link OpenCLPlatform} instances to be
     *  cleared and refilled with completely new {@link OpenCLPlatform} instances.
     *  This will in effect also cause the recreation of any {@link OpenCLDevice} instances
     *  as part of these {@link OpenCLPlatform}s.
     *  This will subsequently cause the recompilation of many OpenCL kernels.
     */
    @Override
    public boolean update( OwnerChangeRequest<Extensions> changeRequest ) {
        _platforms.clear();
        _platforms.addAll( _findLoadAndCompileForAllPlatforms() );
        changeRequest.executeChange();
        return true;
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName()+"@"+Integer.toHexString(hashCode())+"[" +
                    "platforms=["+
                        _platforms.stream().map(Object::toString).collect(Collectors.joining(","))+
                    "]" +
                "]";
    }

    /**
     * @return A new list of freshly created {@link OpenCLPlatform} instances containing freshly instantiated {@link OpenCLDevice}s and kernels.
     */
    private static List<OpenCLPlatform> _findLoadAndCompileForAllPlatforms()
    {
        // Obtain the number of platforms
        int[] numPlatforms = new int[ 1 ];
        clGetPlatformIDs( 0, null, numPlatforms );

        // Obtain the platform IDs
        cl_platform_id[] platforms = new cl_platform_id[ numPlatforms[ 0 ] ];
        clGetPlatformIDs( platforms.length, platforms, null );

        List<OpenCLPlatform> loadedPlatforms = new ArrayList<>();
        for ( cl_platform_id id : platforms ) {
            OpenCLPlatform newPlatform;
            try {
                newPlatform = new OpenCLPlatform( id );
                loadedPlatforms.add( newPlatform );
            } catch ( Exception e ) {
                String message =
                        "Failed to instantiate '"+OpenCLPlatform.class.getSimpleName()+"' " +
                        "with id '0x"+Long.toHexString(id.getNativePointer())+"'!";
                _LOG.error( message, e );
            }
        }
        if ( loadedPlatforms.isEmpty() || loadedPlatforms.stream().allMatch( p -> p.getDevices().isEmpty() ) ) {
            _LOG.warn( Messages.clContextCouldNotFindAnyDevices() );
        }
        return loadedPlatforms;
    }

    @Override
    public DeviceOption find( String searchKey ) {
        Device<Number> result = null;
        double score = 0;
        for ( OpenCLPlatform p : _platforms ) {
            for ( OpenCLDevice d : p.getDevices() ) {
                double similarity = Stream.of("opencl",d.type().name(),d.name(),d.vendor())
                                            .map( word -> word.trim().toLowerCase() )
                                            .mapToDouble( word -> ParseUtil.similarity( word, searchKey ) )
                                            .max()
                                            .orElse(0);
                if ( similarity > score ) {
                    result = d;
                    score = similarity;
                    if ( score == 1 )
                        return new DeviceOption( result, score );
                }
            }
        }
        return new DeviceOption( result, score );
    }

    /**
     *  This method will free all the resources occupied by this context,
     *  meaning that all platforms and their devices will be disposed.
     *  Their kernels will be removed and their tensors restored.
     */
    @Override
    public void dispose() {
        for ( OpenCLPlatform platform : _platforms ) {
            for ( OpenCLDevice device : platform.getDevices() ) {
                device.dispose();
            }
            platform.dispose();
        }
        _platforms.clear();
    }

}
