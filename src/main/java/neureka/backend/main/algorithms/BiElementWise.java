package neureka.backend.main.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.fun.ADAgentSupplier;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.backend.main.algorithms.internal.WithForward;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.internal.RecursiveExecutor;
import neureka.devices.Device;
import neureka.dtype.NumericType;
import neureka.ndim.NDimensional;

public final class BiElementWise extends AbstractFunDeviceAlgorithm<BiElementWise>
{
    public BiElementWise(RecursiveExecutor finalExecutor ) {
        super("elementwise");
        setIsSuitableFor(
            call -> call
                    .validate()
                    .allNotNullHaveSame(NDimensional::size)
                    .allNotNullHaveSame(NDimensional::shape)
                    .allNotNull( t -> t.getDataType().typeClassImplements( NumericType.class ) )
                    .basicSuitability()
        );
        setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD );
        setDeviceExecution( (call, callback) -> finalExecutor.execute(call, callback), (ADAgentSupplier) null );
        setCallPreparation(
            call -> {
                Device<Object> device = (Device<Object>) call.getDevice();
                if ( call.input( 0 ) == null ) // Creating a new tensor:
                {
                    int[] outShape = call.input( 1 ).getNDConf().shape();

                    Class<Object> type = (Class<Object>) call.input(  1 ).getItemType();
                    Tsr<Object> output = Tsr.of( type ).withShape( outShape ).all( 0.0 ).getUnsafe().setIsIntermediate( true );
                    output.setIsVirtual( false );
                    try {
                        device.store( output );
                    } catch( Exception e ) {
                        e.printStackTrace();
                    }
                    call = call.withInputAt( 0, output );
                }
                return call;
            }
        );
    }

    public static WithForward<String> implementationForGPU(String postfix ) {
        return
            forward ->
                backward ->
                    CLImplementation
                        .compiler()
                        .arity( -1 )
                        .kernelSource( Neureka.get().utility().readResource("kernels/elementwise_template.cl") )
                        .activationSource( forward )
                        .differentiationSource( backward )
                        .kernelPostfix( postfix )
                        .execution(
                            call -> {
                                int offset = (call.input( Number.class, 0 ) != null) ? 0 : 1;
                                int gwz = (call.input( Number.class, 0 ) != null) ? call.input( Number.class, 0 ).size() : call.input( Number.class, 1 ).size();
                                call.getDevice()
                                    .getKernel(call)
                                    .passAllOf( call.input( Number.class, offset ) )
                                    .passAllOf( call.input( Number.class, offset + 1 ) )
                                    .passAllOf( call.input( Number.class, offset + 2 ) )
                                    .pass( call.input( Number.class, 0 ).rank() )
                                    .pass( call.getDerivativeIndex() )
                                    .call( gwz );

                                return call.input( 0 );
                            }
                        )
                        .build();
    }

}