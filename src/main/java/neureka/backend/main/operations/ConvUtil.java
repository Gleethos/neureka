package neureka.backend.main.operations;

import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.main.algorithms.Convolution;
import neureka.backend.main.internal.CallExecutor;
import neureka.backend.main.operations.other.Reshape;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import org.jetbrains.annotations.Contract;

public class ConvUtil
{
    public static Convolution createDeconvolutionFor( String op ) {
        return new Convolution()
                .setAutogradModeFor( call -> {
                    if ( call.getOperation().supports( Convolution.class ) ) return AutoDiffMode.BACKWARD_ONLY;
                    Tsr<?> last = null;
                    for ( Tsr<?> t : call.inputs() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return AutoDiffMode.BACKWARD_ONLY;
                        last = t; // Note: shapes are cached!
                    }
                    return AutoDiffMode.FORWARD_AND_BACKWARD;
                })
                .setDeviceExecution(
                    (context, executor) ->
                    {
                        ExecutionCall<?> call = context.initialCall();
                        Function caller = context.caller();
                        Tsr<?>[] tensors = AbstractDeviceAlgorithm.flatten( caller, call ).inputs();
                        Reshape.makeFit(tensors, caller.isDoingAD()); // This might not fit here... (fitting should probably be a setup thing...)
                        for ( Tsr<?> t : tensors ) t.setIsVirtual( false );
                        tensors[ 0 ] = AbstractDeviceAlgorithm.prepareAndExecuteRecursively(
                                                ExecutionCall.of( tensors )
                                                        .andArgs( Arg.DerivIdx.of(0) )
                                                        .running( call.getOperation() )
                                                        .on( call.getDevice() ),
                                                (a, b) -> ConvUtil.executeRecursively(op, a, b)
                                        );

                        return tensors[ 0 ];
                    },
                    ( Function f, ExecutionCall<? extends Device<?>> adCall ) -> {
                        throw new UnsupportedOperationException("Not yet implemented!");
                    }
                )
                .setCallPreparation(
                     call -> {
                         Device<Number> device = call.getDeviceFor(Number.class);
                         if ( call.input( 0 ) == null ) // Creating a new tensor:
                         {
                             int[] shp = shapeOfCon(call.input( 1 ).getNDConf().shape(), call.input( 2 ).getNDConf().shape());
                             Tsr<Double> output = Tsr.of( shp, 0.0 ).getUnsafe().setIsIntermediate( true );
                             output.setIsVirtual( false );
                             device.store( output );
                             return call.withInputAt( 0, output );
                         }
                         return call;
                     }
                )
                .buildFunAlgorithm();
    }

    @Contract(pure = true)
    public static int[] shapeOfCon(int[] shape1, int[] shape2 ) {
        int[] shape = new int[ ( shape1.length + shape2.length ) / 2 ];
        for ( int i = 0; i < shape1.length && i < shape2.length; i++ )
            shape[ i ] = Math.abs( shape1[ i ] - shape2[ i ] ) + 1;
        return shape;
    }


    @Contract( pure = true )
    public static Tsr<?> executeRecursively(
            String op,
            ExecutionCall<? extends Device<?>> call,
            CallExecutor recursiveExecutor // This will indirectly be a recursive call!
    ) {
        call = call.withInputs(call.inputs().clone()); // Let's make sure we don't have any side effects!
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation operation = call.getOperation();

        Tsr<?> result = null;
        if ( call.arity() > 3 ) {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{ call.input( 0 ), call.input( 1 ), call.input( 2 ) };
                result = recursiveExecutor.execute(
                        ExecutionCall.of( reduction )
                                .andArgs( Arg.DerivIdx.of(d) )
                                .running( operation )
                                .on(device)
                );
                call = call.withInputAt( 0, result );

                reduction = Operation.Utility.offsetted(call.inputs(), 1);
                result = recursiveExecutor.execute(
                        ExecutionCall.of( reduction )
                                .andArgs(Arg.DerivIdx.of(d))
                                .running(operation)
                                .on(device)
                );
                call = call.withInputAt( 0, result );
            }
            if ( result == null ) return AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, null );
            return result;
        } else {
            if ( op.equals("x") ) {
                if ( d >= 0 ) {
                    if ( d == 0 )
                        call = call.withInputAt( 0, call.input( 2 ) );
                    else
                        call = call.withInputAt( 0, call.input( 1 ) );
                    return
                        call.input( 0 );
                } else {
                    call.rearrangeInputs( 0, 1, 2 );
                }
            } else if ( op.equals("x"+ ((char) 187)) ) {
                call.rearrangeInputs( 2, 1, 0 );
            }
            return AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, null );
        }
    }


}
