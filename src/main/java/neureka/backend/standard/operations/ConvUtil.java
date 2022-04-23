package neureka.backend.standard.operations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.fun.AutoDiff;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.operations.other.Reshape;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionBuilder;
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
                    if ( call.getOperation().supports( Convolution.class ) ) return AutoDiff.BACKWARD_ONLY;
                    if ( call.getOperation().getOperator().equals(",") ) return AutoDiff.BACKWARD_ONLY; //Reshape
                    Tsr<?> last = null;
                    for ( Tsr<?> t : call.inputs() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return AutoDiff.BACKWARD_ONLY;
                        last = t; // Note: shapes are cached!
                    }
                    return AutoDiff.FORWARD_AND_BACKWARD;
                })
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    {
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                        if ( forward )
                            throw new IllegalArgumentException("Convolution does not support forward-AD!");

                        int d = call.getDerivativeIndex();

                        Function deConv = new FunctionBuilder( Neureka.get().backend() ).build(
                                "I[ 0 ]" + operator + ">>I[ 1 ]" + operator + ">>I[ 2 ]",
                                false
                        );
                        Tsr<?> derivative = f.derive( (Tsr[]) call.inputs(), d );
                        assert d >= 0 && d <= 1;
                        assert derivative != null;
                        assert deConv != null;
                        assert call.arity() >= 2 && call.arity() <= 3;
                        // Now we need to remember the shape of the input which is targeted for back prop.
                        int[] shape = call.input( call.arity() > 2 ? d + 1 : d ).getNDConf().shape();
                        // This is because it will be the shape of the output to the de-convolution!
                        return ADAgent.of( derivative )
                                .setForward( null )
                                .setBackward(
                                    (node, error) ->
                                        deConv.execute(
                                            error,
                                            derivative,
                                            Tsr.of(shape, 0).getUnsafe().setIsIntermediate( false )
                                        )
                                );
                    }
                )
                .setExecutionDispatcher(
                    ( caller, call ) -> {
                        if ( !caller.isFlat() ) return CalcUtil.defaultRecursiveExecution( caller, call );
                        if ( call.getOperation().getOperator().equals("x") ) {
                            Tsr<?>[] tensors = new Tsr[]{null, call.input( 0 ), call.input( 1 )};
                            tensors[ 0 ] =
                                (call.getValOf( Arg.DerivIdx.class ) < 0)
                                    ? Tsr.of(
                                        call.input(0).getValueClass(),
                                            _shapeOfCon( tensors[ 1 ].getNDConf().shape(), tensors[ 2 ].getNDConf().shape() ),
                                            0
                                        )
                                        .getUnsafe()
                                        .setIsIntermediate( true )
                                    : null;

                            for ( Tsr<?> t : tensors ) if ( t != null ) t.setIsVirtual( false );
                            tensors[ 0 ] = CalcUtil.recursiveExecution( call.withInputs(tensors), JunctionUtil::forConvolution );
                            if ( tensors[ 0 ] == null )
                                throw new IllegalStateException("Failed to execute convolution!");
                            return tensors[ 0 ];
                        } else {
                            if ( call.getValOf( Arg.DerivIdx.class ) < 0 ) {
                                Tsr<?>[] tensors = CalcUtil.srcActivation(call.inputs(), call.getValOf( Arg.VarIdx.class ), -1, 0, caller.getSubFunctions().toArray(new Function[0]));
                                Reshape.makeFit(tensors, caller.isDoingAD()); // This might not fit here... (fitting should probably be a setup thing...)
                                for ( Tsr<?> t : tensors ) t.setIsVirtual( false );
                                tensors[ 0 ] = CalcUtil.recursiveExecution(
                                                            ExecutionCall.of( tensors )
                                                                            .andArgs( Arg.DerivIdx.of(0) )
                                                                            .running( call.getOperation() )
                                                                            .on( call.getDevice() ),
                                                            JunctionUtil::forConvolution
                                                        );
                                if ( call.getOperation() == Neureka.get().backend().getOperation("x>>") )
                                    return tensors[ 2 ];
                                else
                                    return tensors[ 0 ];
                            }
                        }
                        return CalcUtil.defaultRecursiveExecution( caller, call );
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
                             try {
                                 device.store( output );
                             } catch ( Exception e ) {
                                 e.printStackTrace();
                             }
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
