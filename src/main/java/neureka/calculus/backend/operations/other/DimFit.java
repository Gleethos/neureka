package neureka.calculus.backend.operations.other;

import neureka.Tsr;
import neureka.devices.Device;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.Function;
import neureka.calculus.backend.operations.AbstractOperationType;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.implementations.functional.GenericImplementation;

import java.util.List;

public class DimFit extends AbstractOperationType
{

    public DimFit()
    {

        super(
                "dimfit",
                "dimfit",
                -1,
                false,
                false,
                true,
                false
        );

        setStringifier(
                children -> {
                    String expression = String.join( ", ", children );
                    if (expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')') {
                        return "dimfit" + expression;
                    }
                    return "dimfit" + "(" + expression + ")";
                }
        );

        GenericImplementation implementation = new GenericImplementation("reshape")
                .setSuitabilityChecker( call -> 1.0f )
                .setBackwardADAnalyzer( call -> true )
                .setForwardADAnalyzer( call -> false )
                .setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    {
                        //int index = call.getDerivativeIndex();
                        //int prefix = ((int[]) call.getAt("ends"))[ 0 ];
                        //int postfix = ((int[]) call.getAt("ends"))[ 1 ];
                        if(forward) {
                            throw new IllegalArgumentException("Dim-Fit operation does not support forward-AD!");
                        }
                        return new DefaultADAgent()
                                .withContext(call.getContext())
                                .withForward(null)
                                .withBackward(
                                        null//(t, error) -> pad(error, new int[]{prefix, postfix}, true)
                                );
                    }
                )
                .setCallHock(
                        ( caller, call ) ->
                        {
                            Tsr<?>[] inputs = caller.srcActivation(call.getTensors(), call.getJ(), -1, 0);
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


                            for (int i=0; i<inputs.length; i++)
                            {
                                if (inputs[ i ].rank()!=largest)
                                {
                                    int[] oldShape = inputs[ i ].getNDConf().shape();
                                    int[] newReshape = new int[largest];
                                    int padding = largest-oldShape.length;

                                    int handle = ( postfix <= prefix )? padding : largest-padding;
                                    for (int ii=0; ii<handle; ii++) newReshape[ii]       = ( postfix <= prefix )? -1 : ii;
                                    for (int ii=handle; ii<largest; ii++) newReshape[ii] = ( postfix <= prefix )? ii-padding : -1;

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
                .setRJAgent( ( call, goDeeperWith ) -> null )
                .setDrainInstantiation( call -> call );

        setImplementation(
                GenericImplementation.class,
                implementation
        );

    }


    @Override
    public double calculate( double[] inputs, int j, int d, List<Function> src ) {
        return src.get( 0 ).call( inputs, j );
    }
}
