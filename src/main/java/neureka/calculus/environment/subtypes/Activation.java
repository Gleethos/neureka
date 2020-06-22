package neureka.calculus.environment.subtypes;

import neureka.calculus.environment.Type;

public class Activation extends AbstractTypeComponent<Type.OperatorCreator>
{
    public Activation(String strActivation, String strDeriviation, Type.OperatorCreator creator){
        super(strActivation, strDeriviation, creator);
    }
}
