package neureka.calculus.environment.subtypes;

import neureka.calculus.environment.Type;

public class Convolution extends AbstractTypeComponent<Type.OperatorCreator>
{
    public Convolution(String strConvolution, String strDeriviation, Type.OperatorCreator creator){
        super(strConvolution, strDeriviation, creator);
    }
}
