package neureka.backend.main.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.fun.ADAgentSupplier;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.algorithms.internal.WithForward;
import neureka.backend.main.implementations.CLImplementation;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import neureka.ndim.iterator.NDIterator;

/**
 *  This is lambda based {@link neureka.backend.api.Algorithm} implementation
 *  providing some basic functionality for implementing custom
 *  activation functions.
 */
public final class Activation extends AbstractFunDeviceAlgorithm<Activation>
{
    public Activation() {
        super("activation");
        setIsSuitableFor(
           call -> call.validate()
                       .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                       .basicSuitability()
        );
        setAutogradModeFor(
            call ->
                call
                    .validate()
                    .all( ( first, second ) -> first.shape().equals(second.shape()) )
                    .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                    .orElse(AutoDiffMode.BACKWARD_ONLY)
        );
        setDeviceExecution( (call, callback) -> AbstractDeviceAlgorithm.executeDeviceAlgorithm(call, callback), (ADAgentSupplier) null );
        setCallPreparation(
            call -> {
                Device device = call.getDeviceFor(Number.class);
                if ( call.input(  0 ) == null ) // Creating a new tensor:
                {
                    int[] shape = call.input(  1 ).getNDConf().shape();
                    Class<Object> type = (Class<Object>) call.input(  1 ).getItemType();
                    Tsr<Object> output = Tsr.of(type).withShape(shape).all( 0.0 ).getUnsafe().setIsIntermediate( true );
                    output.setIsVirtual( false );
                    device.store( output );
                    call = call.withInputAt( 0, output );
                }
                return call;
            }
        );
    }

    public String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/activation_template.cl");
    }

    public static WithForward<String> implementationForGPU( String postfix ) {
        return
            forward ->
                backward ->
                    CLImplementation
                        .compiler()
                        .arity( 2 )
                        .kernelSource( Neureka.get().utility().readResource("kernels/activation_template.cl") )
                        .activationSource( forward )
                        .differentiationSource( backward )
                        .kernelPostfix( postfix )
                        .execution(
                            call -> {
                                int offset = (call.input( Number.class, 0 ) != null) ? 0 : 1;
                                int gwz = (call.input( Number.class, 0 ) != null) ? call.input( Number.class, 0 ).size() : call.input( Number.class, 1 ).size();
                                // Drain tensor needs to be 'actual'! :
                                if (!call.input( Number.class, offset + 1).isVirtual()) call.input( Number.class, offset).setIsVirtual( false );
                                call.getDevice()
                                    .getKernel(call)
                                    .passAllOf( call.input( Number.class, offset ) )
                                    .passAllOf( call.input( Number.class, offset + 1 ) )
                                    .pass( call.input( Number.class, 0 ).rank() )
                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
                                    .call( gwz );

                                return call.input( 0 );
                            }
                        )
                        .build();
    }

}
