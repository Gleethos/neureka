package neureka.calculus.environment.operations.other;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.implementations.Convolution;
import neureka.calculus.environment.implementations.GenericImplementation;
import neureka.calculus.factory.assembly.FunctionBuilder;
import neureka.ndim.AbstractNDArray;
import neureka.ndim.config.AbstractNDC;

import java.util.ArrayList;

public class Reshape extends OperationType
{

    public Reshape()
    {

        super(
                "", ",", -1,
                true,
                false,
                false,
                false,
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

        GenericImplementation implementation = new GenericImplementation(
        ).setHandleChecker(
                call -> true
        ).setADAnalyzer(
                call -> false
        ).setADAgentCreator(
            ( Function f, Tsr derivv, ExecutionCall<Device> call, boolean forward ) ->
            {
                if(forward){
                    throw new IllegalArgumentException("Reshape operation does not support forward-AD!");
                }
                return new ADAgent(
                        null
                ).withForward(
                        (t, derivative) -> FunctionBuilder.build(f.toString(), false).derive(new Tsr[]{derivative},0)
                ).withBackward(
                        (t, error) -> FunctionBuilder.build(f.toString(), false).derive(new Tsr[]{error},0)
                );
            }
        ).setCallHock(
                ( caller, call ) ->
                {
                    Tsr[] inputs = caller.srcActivation(call.getTensors(), call.getJ(), -1, 0);
                    int[] newForm = new int[inputs.length - 1];
                    for ( int i = 0; i < inputs.length - 1; i++ ) {
                        newForm[i] = (int) Tsr.IO.getFrom(inputs[i], 0);//_src.get(i).call(inputs)
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
        ).setRJAgent(
                ( call, goDeeperWith ) ->
                {
                    Tsr[] inputs = call.getTensors();
                    Device device = call.getDevice();
                    int d = call.getDerivativeIndex();
                    OperationType type = call.getType();

                    //inputs = _src_acti(inputs, j, -1, 0);
                    int[] newForm = new int[inputs.length - 1];
                    for ( int i = 0; i < inputs.length - 1; i++ ) {
                        newForm[i] = (int) Tsr.IO.getFrom(inputs[i], 0);//_src.get(i).call(inputs)
                    }
                    if ( d >= 0 ) {//reverse reshape:
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
                    return reshaped( t, newForm, true );
                }
        ).setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[0] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[1].getNDConf().shape();
                        Tsr output = new Tsr( shp, 0.0 );
                        output.setIsVirtual( false );
                        device.add(output);
                        tsrs[0] = output;
                    }
                    return call;
                }
        );

        setImplementation(
                GenericImplementation.class,
                implementation
        );

    }


    public static Tsr reshaped(Tsr tensor, int[] newForm, boolean newTsr)
    {
        tensor = (newTsr) ? (Tsr)tensor.getAt(new ArrayList<>()) : tensor;
        int[] newShape = AbstractNDArray.Utility.Indexing.shpCheck(AbstractNDArray.Utility.Indexing.rearrange(tensor.getNDConf().shape(), newForm), tensor);
        int[] newTranslation = AbstractNDArray.Utility.Indexing.rearrange(tensor.getNDConf().translation(), newShape, newForm);
        int[] newIdxmap = AbstractNDArray.Utility.Indexing.newTlnOf(newShape);
        int[] newSpread = new int[newForm.length];
        for (int i = 0; i < newForm.length; i++) {
            if (newForm[i] < 0) newSpread[i] = 1;
            else if (newForm[i] >= 0) newSpread[i] = tensor.getNDConf().spread(newForm[i]);
        }
        int[] newOffset = new int[newForm.length];
        for (int i = 0; i < newForm.length; i++) {
            if (newForm[i] < 0) newOffset[i] = 0;
            else if (newForm[i] >= 0) newOffset[i] = tensor.getNDConf().offset(newForm[i]);
        }
        tensor.setNDConf( AbstractNDC.construct(newShape, newTranslation, newIdxmap, newSpread, newOffset) );
        return tensor;
    }




}
