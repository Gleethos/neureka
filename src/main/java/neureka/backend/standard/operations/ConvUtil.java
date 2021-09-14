package neureka.backend.standard.operations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.algorithms.Convolution;
import neureka.calculus.CalcUtil;
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
                        (Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                        {
                            Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                            if ( forward ) throw new IllegalArgumentException("Convolution of does not support forward-AD!");

                            Function mul = Neureka.get().context().getFunction().mul();
                            Tsr[] inputs = call.getTensors();
                            int d = call.getDerivativeIndex();

                            Function invX = new FunctionBuilder( Neureka.get().context() ).build(
                                    "I[ 0 ]" + operator + ">>I[ 1 ]" + operator + ">>I[ 2 ]",
                                    false
                            );
                            Tsr<?> deriv = f.derive( inputs, d ); // TODO: Fix 'deriveExecute' here returns null! WHY?!?
                            assert mul != null;
                            assert deriv != null;
                            assert invX != null;
                            return new DefaultADAgent( deriv )
                                    .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, deriv ) )
                                    .setBackward( (node, error) -> invX.execute( error, deriv, Tsr.of(node.getPayload().shape(), 0) ) ); // WARNING! This produced null pointer!
                        }
                )
                .setOrchestration(
                        ( caller, call ) -> {
                            if ( !caller.isFlat() ) return CalcUtil.executeFor( caller, call, (executionCall, executor) -> null );
                            if ( call.getOperation().getOperator().equals("x") ) {

                                Tsr<?>[] inputs = call.getTensors();
                                Tsr<?>[] tsrs = new Tsr[]{null, inputs[ 0 ], inputs[ 1 ]};
                                tsrs[ 0 ] = (call.getDerivativeIndex() < 0)
                                        ? Tsr.ofShape(Tsr.Utility.Indexing.shpOfCon(tsrs[ 1 ].getNDConf().shape(), tsrs[ 2 ].getNDConf().shape()))
                                        : null;

                                for (Tsr<?> t : tsrs) if ( t != null ) t.setIsVirtual( false );
                                CalcUtil.recursiveExecution( call.withTensors(tsrs), JunctionUtil::forConvolution );
                                return tsrs[ 0 ];
                            } else {
                                if (call.getDerivativeIndex() < 0) {
                                    Tsr<?>[] tsrs = CalcUtil.srcActivation(call.getTensors(), call.getJ(), -1, 0, caller.getSubFunctions().toArray(new Function[0]));
                                    Tsr.makeFit(tsrs, caller.isDoingAD()); // This might not fit here... (fitting should probably be a setup thing...)
                                    for ( Tsr<?> t : tsrs ) t.setIsVirtual( false );
                                    CalcUtil.recursiveExecution(
                                                    ExecutionCall.of(tsrs)
                                                                    .andArgs(Arg.DerivIdx.of(0))
                                                                    .running(call.getOperation())
                                                                    .on(call.getDevice()),
                                                    JunctionUtil::forConvolution
                                                );
                                    if ( call.getOperation() == Neureka.get().context().getOperation("x>>") )
                                        return tsrs[ 2 ];
                                    else
                                        return tsrs[ 0 ];
                                }
                            }
                            return CalcUtil.executeFor( caller, call, ( executionCall, executor ) -> null );
                        }
                )
                .setInstantiateNewTensorsForExecutionIn(
                        call -> {
                            Tsr<?>[] tsrs = call.getTensors();
                            Device device = call.getDevice();
                            if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                            {
                                int[] shp = Tsr.Utility.Indexing.shpOfCon(tsrs[ 1 ].getNDConf().shape(), tsrs[ 2 ].getNDConf().shape());
                                Tsr<?> output = Tsr.of( shp, 0.0 );
                                output.setIsVirtual( false );
                                try {
                                    device.store(output);
                                } catch ( Exception e ) {
                                    e.printStackTrace();
                                }
                                tsrs[ 0 ] = output;
                            }
                            return call;
                        }
                )
                .build();
    }

    public static Convolution getConv() {
        if ( conv == null ) conv = createDeconvolutionFor("x");
        return ConvUtil.conv;
    }
}
