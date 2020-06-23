package neureka.calculus.environment.executors;

import neureka.calculus.environment.Type;

public class Operation  extends AbstractTypeExecutor<Type.OperatorCreator>
{
    public Operation(String strOperation, String strDeriviation, Type.OperatorCreator creator){
        super(strOperation, strDeriviation, creator);
    }
}