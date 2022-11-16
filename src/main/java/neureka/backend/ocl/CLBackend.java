package neureka.backend.ocl;

import neureka.backend.api.BackendContext;
import neureka.backend.api.BackendExtension;
import neureka.backend.api.Extensions;
import neureka.backend.api.ini.BackendLoader;
import neureka.backend.api.ini.ReceiveForDevice;
import neureka.backend.main.algorithms.*;
import neureka.backend.main.implementations.broadcast.*;
import neureka.backend.main.implementations.convolution.CLConvolution;
import neureka.backend.main.implementations.elementwise.*;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.backend.main.implementations.matmul.CLMatMul;
import neureka.backend.main.implementations.scalar.CLScalarFunction;
import neureka.backend.main.operations.functions.*;
import neureka.backend.main.operations.linear.Convolution;
import neureka.backend.main.operations.linear.MatMul;
import neureka.backend.main.operations.linear.XConvLeft;
import neureka.backend.main.operations.linear.XConvRight;
import neureka.backend.main.operations.linear.internal.opencl.CLSum;
import neureka.backend.main.operations.operator.*;
import neureka.backend.main.operations.other.AssignLeft;
import neureka.backend.main.operations.other.Randomization;
import neureka.backend.main.operations.other.Sum;
import neureka.math.parsing.ParseUtil;
import neureka.common.composition.Component;
import neureka.devices.Device;
import neureka.devices.opencl.OpenCLDevice;
import neureka.devices.opencl.OpenCLPlatform;
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
 *  used for managing {@link neureka.backend.api.Operation}, {@link neureka.math.Function}
 *  as well as {@link Component} implementation instances like this one.
 *  A given state might not be compatible with the concepts introduced in other contexts
 *  which is why it makes sense to have separate "worlds" with potential different operations...
 *  The component system of the {@link BackendContext} exist so that a given context
 *  can be extended for more functionality
 *  and also to attach relevant states like for example in this case the {@link CLBackend}
 *  instance will directly or indirectly reference kernels, memory objects and other concepts
 *  exposed by OpenCL...
 */
public final class CLBackend implements BackendExtension
{
    private static final Logger _LOG = LoggerFactory.getLogger(CLBackend.class);

    private final List<OpenCLPlatform> _platforms = new ArrayList<>();
    private final CLSettings _settings = new CLSettings();

    /**
     *  Use this constructor if you want to create a new OpenCL world in which there
     *  are unique {@link OpenCLPlatform} and {@link OpenCLDevice} instances.
     */
    public CLBackend() {}

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
        changeRequest.executeChange(); // This can be an 'add', 'remove' or 'transfer' of this component!
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
        List<String> failures = new ArrayList<>();
        for ( cl_platform_id id : platforms ) {
            OpenCLPlatform newPlatform = null;
            try {
                newPlatform = new OpenCLPlatform( id );
            } catch ( Exception e ) {
                String message =
                        "Failed to instantiate '"+OpenCLPlatform.class.getSimpleName()+"' " +
                        "with id '0x"+Long.toHexString(id.getNativePointer())+"'!";
                _LOG.error( message, e );
                failures.add( message + " Reason: " + e.getMessage() );
            }
            if ( newPlatform != null )
                loadedPlatforms.add( newPlatform );
        }
        if ( loadedPlatforms.isEmpty() || loadedPlatforms.stream().allMatch( p -> p.getDevices().isEmpty() ) )
            _LOG.info( Messages.clContextCouldNotFindAnyDevices() );

        if ( loadedPlatforms.isEmpty() && platforms.length > 0 )
            // There should be at least one platform with at least one device!
            throw new RuntimeException(
                "Failed to instantiate any '"+OpenCLPlatform.class.getSimpleName()+"' instance!\n" +
                "Reasons: \n    " + failures.stream().collect(Collectors.joining("\n    "))
            );

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

    @Override
    public void reset() {
        _settings.reset();
    }

    /**
     *  This method will free all the resources occupied by this context,
     *  meaning that all platforms and their devices will be disposed.
     *  Their kernels will be removed and their tensors restored.
     */
    @Override
    public void dispose() {
        for ( OpenCLPlatform platform : _platforms ) {
            for ( OpenCLDevice device : platform.getDevices() ) device.dispose();
            platform.dispose();
        }
        _platforms.clear();
    }

    @Override
    public BackendLoader getLoader() {
        return receiver -> _load( receiver.forDevice(OpenCLDevice.class) );
    }

    private void _load( ReceiveForDevice<OpenCLDevice> receive )
    {
        receive.forOperation( Power.class )
                .set( BiScalarBroadcast.class, context -> new CLScalarBroadcastPower( context.getOperationIdentidier() ) )
                .set( Broadcast.class,     context -> new CLBroadcastPower( context.getOperationIdentidier() )       )
                .set( BiElementwise.class, context -> new CLBiElementwisePower( context.getOperationIdentidier() )   );

        receive.forOperation( Addition.class )
                .set( BiScalarBroadcast.class, context -> new CLScalarBroadcastAddition(context.getOperationIdentidier()) )
                .set( Broadcast.class,     context -> new CLBroadcastAddition( context.getOperationIdentidier() )       )
                .set( BiElementwise.class, context -> new CLBiElementwiseAddition( context.getOperationIdentidier() ));

        receive.forOperation( Subtraction.class )
                .set( BiScalarBroadcast.class, context -> new CLScalarBroadcastSubtraction( context.getOperationIdentidier() ) )
                .set( Broadcast.class,     context -> new CLBroadcastSubtraction( context.getOperationIdentidier() )       )
                .set( BiElementwise.class, context -> new CLBiElementwiseSubtraction( context.getOperationIdentidier() ) );

        receive.forOperation( Multiplication.class )
                .set( BiScalarBroadcast.class, context -> new CLScalarBroadcastMultiplication( context.getOperationIdentidier() ) )
                .set( Broadcast.class,     context -> new CLBroadcastMultiplication( context.getOperationIdentidier() )       )
                .set( BiElementwise.class, context -> new CLBiElementwiseMultiplication( context.getOperationIdentidier() ) );

        receive.forOperation( Division.class )
                .set( BiScalarBroadcast.class, context -> new CLScalarBroadcastDivision( context.getOperationIdentidier() ) )
                .set( Broadcast.class,     context -> new CLBroadcastDivision( context.getOperationIdentidier() )       )
                .set( BiElementwise.class, context -> new CLBiElementwiseDivision( context.getOperationIdentidier() ) );

        receive.forOperation( Modulo.class )
                .set( BiScalarBroadcast.class, context -> new CLScalarBroadcastModulo( context.getOperationIdentidier() ) )
                .set( Broadcast.class,     context -> new CLBroadcastModulo( context.getOperationIdentidier() )       )
                .set( BiElementwise.class, context -> new CLBiElementwiseModulo( context.getOperationIdentidier() ) );

        receive.forOperation( AssignLeft.class )
                .set( BiScalarBroadcast.class, context -> new CLScalarBroadcastIdentity( context.getOperationIdentidier() ) )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.IDENTITY ) );

        receive.forOperation( Convolution.class )
                .set( NDConvolution.class, context -> new CLConvolution( context.getOperationIdentidier() ) );
        receive.forOperation( XConvLeft.class )
                .set( NDConvolution.class, context -> new CLConvolution( context.getOperationIdentidier() ) );
        receive.forOperation( XConvRight.class )
                .set( NDConvolution.class, context -> new CLConvolution( context.getOperationIdentidier() ) );

        receive.forOperation( MatMul.class )
                .set( MatMulAlgorithm.class, context -> new CLMatMul() );

        receive.forOperation( Sum.class )
                .set( SumAlgorithm.class, context -> new CLSum() );

        receive.forOperation( Randomization.class )
                .set( ElementwiseAlgorithm.class, context -> new CLRandomization() );

        receive.forOperation( Absolute.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.ABSOLUTE) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.ABSOLUTE) );
        receive.forOperation( Cosinus.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.COSINUS) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.COSINUS) );
        receive.forOperation( GaSU.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.GASU) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.GASU) );
        receive.forOperation( GaTU.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.GATU) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.GATU) );
        receive.forOperation( Gaussian.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.GAUSSIAN) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.GAUSSIAN) );
        receive.forOperation( GaussianFast.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.GAUSSIAN_FAST) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.GAUSSIAN_FAST) );
        receive.forOperation( GeLU.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.GELU) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.GELU) );
        receive.forOperation( Identity.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.IDENTITY) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.IDENTITY) );
        receive.forOperation( Logarithm.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.LOGARITHM) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.LOGARITHM) );
        receive.forOperation( Quadratic.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.QUADRATIC) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.QUADRATIC) );
        receive.forOperation( ReLU.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.RELU) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.RELU) );
        receive.forOperation( SeLU.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.SELU) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.SELU) );
        receive.forOperation( Sigmoid.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.SIGMOID) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.SIGMOID) );
        receive.forOperation( SiLU.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.SILU) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.SILU) );
        receive.forOperation( Sinus.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.SINUS) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.SINUS) );
        receive.forOperation( Softplus.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.SOFTPLUS) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.SOFTPLUS) );
        receive.forOperation( Softsign.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.SOFTSIGN) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.SOFTSIGN) );
        receive.forOperation( Tanh.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.TANH) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.TANH) );
        receive.forOperation( TanhFast.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.TANH_FAST) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.TANH_FAST) );

        receive.forOperation( Exp.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.EXP) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.EXP) );
        receive.forOperation( Cbrt.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.CBRT) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.CBRT) );
        receive.forOperation( Log10.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.LOG10) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.LOG10) );
        receive.forOperation( Sqrt.class )
                .set( ElementwiseAlgorithm.class, context -> new CLElementwiseFunction( ScalarFun.SQRT) )
                .set( ScalarAlgorithm.class, context -> new CLScalarFunction(ScalarFun.SQRT) );
    }

}
