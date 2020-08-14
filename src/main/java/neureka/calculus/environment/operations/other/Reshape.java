package neureka.calculus.environment.operations.other;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.OperationType;
import neureka.calculus.factory.assembly.FunctionBuilder;
import neureka.calculus.factory.components.FunctionConstant;

public class Reshape extends OperationType
{

    public Reshape(){

        super(
                "", ",", -1,
                true,
                false,
                false,
                false,
                false
        );

        setStringifier(
            children -> {
                java.util.function.Function<String, Boolean> isConstantNumeric = (s)->{
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
                                        : children.get(i));
                    }
                    if ( i < children.size() - 2 ) {
                        reconstructed.append(",");
                    }
                }
                return "(" + reconstructed + ")";
            }
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
