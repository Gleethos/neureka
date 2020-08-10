package neureka.calculus.environment.operations.other;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.OperationType;
import neureka.calculus.factory.assembly.FunctionBuilder;

public class Reshape extends OperationType {

    public Reshape(){

        super(
                "", ",", -1, true, false, false, false, false
        );

    }

    @Override
    public ADAgent getADAgentOf(Function f, Tsr[] inputs, int i, boolean forward){
        if(forward){
            throw new IllegalArgumentException("Reshape operation does not support forward-AD!");
        }
        return new ADAgent(
                ()->null,
                (t, derivative) -> FunctionBuilder.build(f.toString(), false).derive(new Tsr[]{derivative},0),
                (t, error) -> FunctionBuilder.build(f.toString(), false).derive(new Tsr[]{error},0)
        );
    }



}
