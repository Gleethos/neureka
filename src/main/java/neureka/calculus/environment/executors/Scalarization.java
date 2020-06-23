package neureka.calculus.environment.executors;

import neureka.calculus.environment.Type;

public class Scalarization extends AbstractTypeExecutor<Type.ScalarOperatorCreator>
{
    public Scalarization(String strScalarized, String strDeriviation, Type.ScalarOperatorCreator creator){
        super(strScalarized, strDeriviation, creator);
    }
}