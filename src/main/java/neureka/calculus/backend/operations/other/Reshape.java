package neureka.calculus.backend.operations.other;

import neureka.Tsr;
import neureka.device.Device;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.Function;
import neureka.calculus.backend.operations.AbstractOperationType;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.implementations.functional.GenericImplementation;
import neureka.calculus.frontend.assembly.FunctionBuilder;
import neureka.ndim.AbstractNDArray;
import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;

import java.util.ArrayList;
import java.util.List;

public class Reshape extends AbstractOperationType
{

    public Reshape()
    {

        super(
                "reshape", ",", -1,
                true,
                false,
                true,
                false
        );

        setStringifier(
            children ->
            {
                java.util.function.Function<String, Boolean> isConstantNumeric =
                s ->
                {
                    try {
                        Double.parseDouble(s);
                        return true;
                    } catch (Exception e) { return false; }
                };
                StringBuilder reconstructed = new StringBuilder();
                reconstructed.insert(0, "[");
                for ( int i = 0; i < children.size(); ++i ) {
                    if ( i == children.size() - 1 ) {
                        reconstructed.append("]:(").append(
                                ( isConstantNumeric.apply(children.get(i)) )
                                        ? children.get(i).split("\\.")[0]
                                        : children.get(i)
                        ).append(")");
                    } else {
                        reconstructed.append(
                                ( isConstantNumeric.apply(children.get(i)) )
                                        ? children.get(i).split("\\.")[0]
                                        : children.get(i)
                        );
                    }
                    if ( i < children.size() - 2 ) {
                        reconstructed.append(",");
                    }
                }
                return "(" + reconstructed + ")";
            }
        );

        GenericImplementation implementation = new GenericImplementation("reshape")
                .setSuitabilityChecker( call -> 1.0f )
                .setBackwardADAnalyzer( call -> true )
                .setForwardADAnalyzer(call -> false )
                .setADAgentSupplier(
                    ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    {
                        //Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                        if(forward){
                            throw new IllegalArgumentException("Reshape operation does not support forward-AD!");
                        }
                        return new DefaultADAgent(null)
                                .withForward( ( t, derivative ) -> FunctionBuilder.build( f.toString(), false ).derive( new Tsr[]{ derivative },0 ) )
                                .withBackward( ( t, error ) -> FunctionBuilder.build( f.toString(), false ).derive( new Tsr[]{ error },0 ) );
                    }
                ).setCallHock(
                    ( caller, call ) ->
                    {
                        Tsr[] inputs = caller.srcActivation( call.getTensors(), call.getJ(), -1, 0 );
                        int[] newForm = new int[ inputs.length - 1 ];
                        for ( int i = 0; i < inputs.length - 1; i++ ) {
                            newForm[ i ] = (int) Tsr.IO.getFrom( inputs[ i ], 0 );
                        }
                        if ( call.getDerivativeIndex() >= 0 ) {//reverse reshape:
                            int reverseLength = 0;
                            for ( int e : newForm ) {
                                if ( e >= 0 ) reverseLength++;
                            }
                            int[] reversed = new int[reverseLength];
                            int reshape_i = 0;
                            int reverse_i = 0;
                            while ( reverse_i < reverseLength ) {
                                if ( newForm[ reshape_i ] >= 0 ) {
                                    reversed[ newForm[reshape_i] ] = reshape_i;
                                    reverse_i++;
                                }
                                reshape_i++;
                            }
                            newForm = reversed;
                        }
                        Tsr t = inputs[inputs.length - 1];
                        return reshaped(t, newForm, true);
                    }
                )
                .setRJAgent( ( call, goDeeperWith ) -> null )
                .setDrainInstantiation( call -> call);

        setImplementation(
                GenericImplementation.class,
                implementation
        );

    }


    public static Tsr reshaped(Tsr tensor, int[] newForm, boolean newTsr)
    {
        tensor = (newTsr) ? (Tsr)tensor.getAt(new ArrayList<>()) : tensor;
        NDConfiguration newNDC = tensor.getNDConf().newReshaped(newForm);
        AbstractNDArray.Utility.Indexing.shpCheck(newNDC.shape(), tensor);
        tensor.setNDConf( newNDC );
        return tensor;
    }


    @Override
    public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
    }
}
