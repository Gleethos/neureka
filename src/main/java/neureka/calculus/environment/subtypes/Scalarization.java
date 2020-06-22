package neureka.calculus.environment.subtypes;

import neureka.calculus.environment.Type;

public class Scalarization extends AbstractTypeComponent<Type.ScalarOperatorCreator>
{
    public Scalarization(String strScalarized, String strDeriviation, Type.ScalarOperatorCreator creator){
        super(strScalarized, strDeriviation, creator);
    }
}