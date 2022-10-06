package neureka.backend.main.operations;

import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.main.algorithms.NDConvolution;
import neureka.backend.main.internal.CallExecutor;
import neureka.backend.main.operations.other.Reshape;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;

public class ConvUtil
{
    public static NDConvolution createDeconvolutionFor(String op ) {
        return new NDConvolution()
                .setAutogradModeFor( call -> {
                    if ( call.getOperation().supports( NDConvolution.class ) ) return AutoDiffMode.BACKWARD_ONLY;
                    Tsr<?> last = null;
                    for ( Tsr<?> t : call.inputs() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return AutoDiffMode.BACKWARD_ONLY;
                        last = t; // Note: shapes are cached!
                    }
                    return AutoDiffMode.FORWARD_AND_BACKWARD;
                })
                .setDeviceExecution(
                    (call, executor) ->
                    {
                        int offset = ( call.input(0) == null ? 1 : 0 );
                        Tsr<?>[] tensors = new Tsr[]{call.input(offset+0), call.input(offset+1), call.input(offset+2)};
                        Reshape.makeFit(tensors, false); // This might not fit here... (fitting should probably be a setup thing...)
                        for ( Tsr<?> t : tensors ) t.getMut().setIsVirtual( false );
                        return AbstractDeviceAlgorithm.prepareAndExecuteRecursively(
                                                ExecutionCall.of( tensors )
                                                        .andArgs( Arg.DerivIdx.of(0) )
                                                        .running( call.getOperation() )
                                                        .on( call.getDevice() ),
                                                (a, b) -> ConvUtil.executeRecursively(op, a, b)
                                            );
                    },
                    ( Function f, ExecutionCall<? extends Device<?>> adCall ) -> {
                        throw new UnsupportedOperationException("Not yet implemented!");
                    }
                )
                .setCallPreparation(
                     call -> {
                         if ( call.input( 0 ) == null )
                             return call.withRemovedInputAt( 0 );
                         return call;
                     }
                )
                .buildFunAlgorithm();
    }

    
    public static int[] shapeOfCon(int[] shape1, int[] shape2 ) {
        int[] shape = new int[ ( shape1.length + shape2.length ) / 2 ];
        for ( int i = 0; i < shape1.length && i < shape2.length; i++ )
            shape[ i ] = Math.abs( shape1[ i ] - shape2[ i ] ) + 1;
        return shape;
    }


    
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
