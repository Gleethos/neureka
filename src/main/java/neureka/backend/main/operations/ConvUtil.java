package neureka.backend.main.operations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.main.algorithms.Convolution;
import neureka.backend.main.operations.other.Reshape;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
import neureka.calculus.internal.CalcUtil;
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
                .setExecution(
                    ( caller, call ) ->
                        Result.of(CalcUtil.executeFor( caller, call, (innerCaller, innerCall) ->
                        {
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

                                ExecutionCall<?> prepared = CalcUtil._prepareForExecution( call.withInputs(tensors) );
                                return CalcUtil.executeOnCommonDevice(prepared,()->JunctionUtil.forConvolution( prepared, null/*recursion is not expected to happen here*/ ));
                            } else {
                                Tsr<?>[] tensors = CalcUtil.flatten( caller, call ).inputs();
                                Reshape.makeFit(tensors, caller.isDoingAD()); // This might not fit here... (fitting should probably be a setup thing...)
                                for ( Tsr<?> t : tensors ) t.setIsVirtual( false );
                                tensors[ 0 ] = CalcUtil.prepareAndExecuteRecursively(
                                        ExecutionCall.of( tensors )
                                                .andArgs( Arg.DerivIdx.of(0) )
                                                .running( call.getOperation() )
                                                .on( call.getDevice() ),
                                        JunctionUtil::forConvolution
                                );

                                return tensors[ 0 ];
                            }
                        }))
                        .withAutoDiff(( Function f, ExecutionCall<? extends Device<?>> adCall ) ->
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
                        })
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
}
