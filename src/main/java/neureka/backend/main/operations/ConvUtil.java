package neureka.backend.main.operations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Operation;
import neureka.backend.main.algorithms.Convolution;
import neureka.backend.main.internal.CallExecutor;
import neureka.backend.main.operations.other.Reshape;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
import neureka.backend.main.internal.AlgoUtil;
import neureka.devices.Device;
import org.jetbrains.annotations.Contract;

public class ConvUtil {

    /**
     *  There will always only be a single convolution instance
     *  shared among all 3 convolution operations.
     */
    private static Convolution conv = null;

    public static Convolution createDeconvolutionFor( String operator ) {
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
                        if ( call.getOperation().getOperator().equals("x") ) {
                            Tsr<?>[] tensors = new Tsr[]{null, call.input( 0 ), call.input( 1 )};
                            tensors[ 0 ] =
                                (call.getValOf( Arg.DerivIdx.class ) < 0)
                                        ? Tsr.of(
                                                call.input(0).getItemClass(),
                                                _shapeOfCon( tensors[ 1 ].getNDConf().shape(), tensors[ 2 ].getNDConf().shape() ),
                                                0
                                        )
                                        .getUnsafe()
                                        .setIsIntermediate( true )
                                        : null;

                            for ( Tsr<?> t : tensors ) if ( t != null ) t.setIsVirtual( false );

                            ExecutionCall<?> prepared = AlgoUtil._prepareForExecution( call.withInputs(tensors) );
                            return AlgoUtil.executeOnCommonDevice(prepared,()->ConvUtil._executeRecursively( prepared, null/*recursion is not expected to happen here*/ ));
                        } else {
                            Tsr<?>[] tensors = AlgoUtil.flatten( caller, call ).inputs();
                            Reshape.makeFit(tensors, caller.isDoingAD()); // This might not fit here... (fitting should probably be a setup thing...)
                            for ( Tsr<?> t : tensors ) t.setIsVirtual( false );
                            tensors[ 0 ] = AlgoUtil.prepareAndExecuteRecursively(
                                    ExecutionCall.of( tensors )
                                            .andArgs( Arg.DerivIdx.of(0) )
                                            .running( call.getOperation() )
                                            .on( call.getDevice() ),
                                    ConvUtil::_executeRecursively
                            );

                            return tensors[ 0 ];
                        }
                    },
                    ( Function f, ExecutionCall<? extends Device<?>> adCall ) ->
                    {
                        int d = adCall.getDerivativeIndex();
                        Function deConv = new FunctionParser( Neureka.get().backend() ).parse(
                                "I[ 0 ]" + operator + ">>I[ 1 ]" + operator + ">>I[ 2 ]",
                                false
                        );
                        Tsr<?> derivative = f.derive( (Tsr[]) adCall.inputs(), d );
                        assert d >= 0 && d <= 1;
                        assert derivative != null;
                        assert deConv != null;
                        assert adCall.arity() >= 2 && adCall.arity() <= 3;
                        // Now we need to remember the shape of the input which is targeted for back prop.
                        int[] shape = adCall.input( adCall.arity() > 2 ? d + 1 : d ).getNDConf().shape();
                        // This is because it will be the shape of the output to the de-convolution!
                        return ADAgent.of( derivative )
                                .withAD( target ->
                                        deConv.execute(
                                                target.error(),
                                                derivative,
                                                Tsr.of(shape, 0).getUnsafe().setIsIntermediate( false )
                                        )
                                );
                    }
                )
                .setCallPreparation(
                     call -> {
                         Device<Number> device = call.getDeviceFor(Number.class);
                         if ( call.input( 0 ) == null ) // Creating a new tensor:
                         {
                             int[] shp = _shapeOfCon(call.input( 1 ).getNDConf().shape(), call.input( 2 ).getNDConf().shape());
                             Tsr<Double> output = Tsr.of( shp, 0.0 ).getUnsafe().setIsIntermediate( true );
                             output.setIsVirtual( false );
                             device.store( output );
                             call.setInput( 0, output );
                         }
                         return call;
                     }
                )
                .buildFunAlgorithm();
    }

    public static Convolution getConv() {
        if ( conv == null )
            conv = createDeconvolutionFor("x");
        return ConvUtil.conv;
    }

    @Contract(pure = true)
    private static int[] _shapeOfCon( int[] shape1, int[] shape2 ) {
        int[] shape = new int[ ( shape1.length + shape2.length ) / 2 ];
        for ( int i = 0; i < shape1.length && i < shape2.length; i++ )
            shape[ i ] = Math.abs( shape1[ i ] - shape2[ i ] ) + 1;
        return shape;
    }


    @Contract( pure = true )
    private static Tsr<?> _executeRecursively(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor recursiveExecutor // This will indirectly be a recursive call!
    ) {
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
                call.setInput( 0, result );

                reduction = Operation.Utility.offsetted(call.inputs(), 1);
                result = recursiveExecutor.execute(
                        ExecutionCall.of( reduction )
                                .andArgs(Arg.DerivIdx.of(d))
                                .running(operation)
                                .on(device)
                );
                call.setInput( 0, result );
            }
            if ( result == null ) return AlgoUtil.executeDeviceAlgorithm( call, null );
            return result;
        } else {
            if ( call.getOperation().getOperator().equals("x") ) {
                if ( d >= 0 ) {
                    if ( d == 0 ) call.setInput( 0, call.input( 2 ) );
                    else call.setInput( 0, call.input( 1 ) );
                    return call.input( 0 );
                } else {
                    call.rearrangeInputs( 0, 1, 2 );
                }
            } else if ( call.getOperation().getOperator().equals("x"+ ((char) 187)) ) {
                call.rearrangeInputs( 2, 1, 0 );
            }
            return AlgoUtil.executeDeviceAlgorithm( call, null );
        }
    }


}
