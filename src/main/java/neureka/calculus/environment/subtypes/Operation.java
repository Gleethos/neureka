package neureka.calculus.environment.subtypes;

import neureka.calculus.environment.Type;

public class Operation  extends AbstractTypeComponent<Type.OperatorCreator>
{
    public Operation(String strOperation, String strDeriviation, Type.OperatorCreator creator){
        super(strOperation, strDeriviation, creator);
    }
}