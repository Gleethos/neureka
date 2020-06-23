package neureka.calculus.environment.executors;

import neureka.calculus.environment.Type;

public class Activation extends AbstractTypeExecutor<Type.OperatorCreator>
{
    public Activation(String strActivation, String strDeriviation, Type.OperatorCreator creator){
        super(strActivation, strDeriviation, creator);
    }

}
