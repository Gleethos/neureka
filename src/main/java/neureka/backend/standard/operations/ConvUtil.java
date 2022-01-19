package neureka.backend.standard.operations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.operations.other.Reshape;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.Device;

public class ConvUtil {

    private static Convolution conv = null;

    public static Convolution createDeconvolutionFor( String operator ) {
        return new Convolution()
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor(
                        call -> {
                            if ( call.getOperation().supports( Convolution.class ) ) return false;
                            if ( call.getOperation().getOperator().equals(",") ) return false; //Reshape
                            Tsr<?> last = null;
                            for ( Tsr<?> t : call.getTensors() ) {
                                if ( last != null && !last.shape().equals(t.shape()) ) return false;
                                last = t; // Note: shapes are cached!
                            }
                            return true;
                        }
                )
                .setSupplyADAgentFor(
                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                        {
                            Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                            if ( forward )
                                throw new IllegalArgumentException("Convolution does not support forward-AD!");

                            Function mul = Neureka.get().backend().getFunction().mul();
                            Tsr[] inputs = call.getTensors();
                            int d = call.getDerivativeIndex();

                            Function deConv = new FunctionBuilder( Neureka.get().backend() ).build(
                                    "I[ 0 ]" + operator + ">>I[ 1 ]" + operator + ">>I[ 2 ]",
                                    false
                            );
                            Tsr<?> derivative = f.derive( inputs, d );
                            assert d >= 0 && d <= 1;
                            assert mul != null;
                            assert derivative != null;
                            assert deConv != null;
                            assert inputs.length >= 2 && inputs.length <= 3;
                            // Now we need to remember the shape of the input which is targeted for back prop.
                            int[] shape = inputs[ inputs.length > 2 ? d + 1 : d ].getNDConf().shape();
                            // This is because it will be the shape of the output to the de-convolution!
                            return ADAgent.of( derivative )
                                    .setForward(
                                        (node, forwardDerivative ) -> mul.execute( forwardDerivative, derivative )
                                    )
                                    .setBackward(
                                        (node, error) ->
                                                deConv.execute(
                                                        error,
                                                        derivative,
                                                        Tsr.of(shape, 0).getMutate().setIsIntermediate( false )
                                                )
                                    );
                        }
                )
                .setExecutionDispatcher(
                        ( caller, call ) -> {
                            if ( !caller.isFlat() ) return CalcUtil.defaultRecursiveExecution( caller, call );
                            if ( call.getOperation().getOperator().equals("x") ) {

                                Tsr<?>[] inputs = call.getTensors();
                                Tsr<?>[] tsrs = new Tsr[]{null, inputs[ 0 ], inputs[ 1 ]};
                                tsrs[ 0 ] = (call.getValOf( Arg.DerivIdx.class ) < 0)
                                        ? Tsr.ofShape(Tsr.Utility.Indexing.shpOfCon(tsrs[ 1 ].getNDConf().shape(), tsrs[ 2 ].getNDConf().shape())).getMutate().setIsIntermediate( true )
                                        : null;

                                for (Tsr<?> t : tsrs) if ( t != null ) t.setIsVirtual( false );
                                CalcUtil.recursiveExecution( call.withTensors(tsrs), JunctionUtil::forConvolution );
                                return tsrs[ 0 ];
                            } else {
                                if ( call.getValOf( Arg.DerivIdx.class ) < 0 ) {
                                    Tsr<?>[] tsrs = CalcUtil.srcActivation(call.getTensors(), call.getJ(), -1, 0, caller.getSubFunctions().toArray(new Function[0]));
                                    Reshape.makeFit(tsrs, caller.isDoingAD()); // This might not fit here... (fitting should probably be a setup thing...)
                                    for ( Tsr<?> t : tsrs ) t.setIsVirtual( false );
                                    CalcUtil.recursiveExecution(
                                                    ExecutionCall.of(tsrs)
                                                                    .andArgs(Arg.DerivIdx.of(0))
                                                                    .running(call.getOperation())
                                                                    .on(call.getDevice()),
                                                    JunctionUtil::forConvolution
                                                );
                                    if ( call.getOperation() == Neureka.get().backend().getOperation("x>>") )
                                        return tsrs[ 2 ];
                                    else
                                        return tsrs[ 0 ];
                                }
                            }
                            return CalcUtil.defaultRecursiveExecution( caller, call );
                        }
                )
                .setCallPreparation(
                        call -> {
                            Tsr<?>[] tensors = call.getTensors();
                            Device<Number> device = call.getDeviceFor(Number.class);
                            if ( tensors[ 0 ] == null ) // Creating a new tensor:
                            {
                                int[] shp = Tsr.Utility.Indexing.shpOfCon(tensors[ 1 ].getNDConf().shape(), tensors[ 2 ].getNDConf().shape());
                                Tsr<Double> output = Tsr.of( shp, 0.0 ).getMutate().setIsIntermediate( true );
                                output.setIsVirtual( false );
                                try {
                                    device.store( output );
                                } catch ( Exception e ) {
                                    e.printStackTrace();
                                }
                                tensors[ 0 ] = output;
                            }
                            return call;
                        }
                )
                .buildFunAlgorithm();
    }

    public static Convolution getConv() {
        if ( conv == null ) conv = createDeconvolutionFor("x");
        return ConvUtil.conv;
    }
}
