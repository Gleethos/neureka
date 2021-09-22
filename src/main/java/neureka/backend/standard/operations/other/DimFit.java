package neureka.backend.standard.operations.other;

import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.CalcUtil;
import neureka.calculus.args.Arg;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.GenericAlgorithm;
import neureka.calculus.Function;
import neureka.devices.Device;
import org.jetbrains.annotations.Contract;

public class DimFit extends AbstractOperation
{

    public DimFit()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "dimfit"    )
                        .setOperator(         "dimfit"    )
                        .setArity(            -1          )
                        .setIsOperator(       false       )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( true        )
                        .setIsInline(         false       )
        );

        GenericAlgorithm implementation = new GenericAlgorithm("reshape")
                .setIsSuitableFor( call -> 1.0f )
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> false )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    {
                        //int index = call.getDerivativeIndex();
                        //int prefix = ((int[]) call.getAt("ends"))[ 0 ];
                        //int postfix = ((int[]) call.getAt("ends"))[ 1 ];
                        if ( forward ) {
                            throw new IllegalArgumentException("Dim-Fit operation does not support forward-AD!");
                        }
                        return new DefaultADAgent()
                                        .withContext(call.getMetaArgs().getAll(Arg.class))
                                        .setForward(null)
                                        .setBackward(
                                                null//(t, error) -> pad(error, new int[]{prefix, postfix}, true)
                                        );
                    }
                )
                .setExecutionDispatcher(
                        ( caller, call ) ->
                        {
                            Tsr<?>[] inputs = CalcUtil.srcActivation(call.getTensors(), call.getJ(), -1, 0, caller.getSubFunctions().toArray(new Function[0]));
                            assert call.getDerivativeIndex() < 0;

                            int largest = -1;
                            int[] shape = null;
                            for ( Tsr<?> t : inputs ) if ( t.rank() > largest ) {
                                largest = t.rank();
                                shape = t.getNDConf().shape();
                            }
                            int prefix = 0;
                            for ( int s : shape ) if ( s == 1 ) prefix++; else break;
                            int postfix = 0;
                            for ( int i = shape.length-1; i>=0; i-- ) if ( shape[ i ] == 1 ) postfix++; else break;

                            int[][] change = new int[inputs.length][];


                            for ( int i=0; i<inputs.length; i++)
                            {
                                if (inputs[ i ].rank()!=largest)
                                {
                                    int[] oldShape = inputs[ i ].getNDConf().shape();
                                    int[] newReshape = new int[largest];
                                    int padding = largest-oldShape.length;

                                    int handle = ( postfix <= prefix )? padding : largest-padding;
                                    for ( int ii=0; ii<handle; ii++) newReshape[ ii ]       = ( postfix <= prefix )? -1 : ii;
                                    for ( int ii=handle; ii<largest; ii++) newReshape[ ii ] = ( postfix <= prefix )? ii-padding : -1;

                                    change[ i ] = newReshape;
                                    //Function f = Function.create(
                                    //        AbstractNDArray.Utility.Stringify.strConf(newReshape) +":(I[ 0 ])"
                                    //);
                                    //inputs[ i ] = f.call( inputs[ i ] );

                                    //inputs[ i ].setNDConf(inputs[ i ].getNDConf().view())
                                }
                            }




                            return null;


                            //Tsr<?> t = inputs[ 0 ];
                            //if ( call.getDerivativeIndex() == 0 ) {
                            //    int prefix = ((int[]) call.getAt("ends"))[ 0 ];
                            //    int postfix = ((int[]) call.getAt("ends"))[ 0 ];
                            //    return pad(t, new int[]{prefix, postfix}, true);
                            //} else {
                            //    int[] ends = new int[ 2 ];
                            //    call.putAt("ends", ends);
                            //    return trim(t, ends, true);
                            //}
                        }
                )
                .setCallPreparation( call -> call )
                .build();

        setAlgorithm(
                GenericAlgorithm.class,
                implementation
        );
    }

    @Contract( pure = true )
    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if (expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')') {
            return "dimfit" + expression;
        }
        return "dimfit" + "(" + expression + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return src[ 0 ].call( inputs, j );
    }
}
