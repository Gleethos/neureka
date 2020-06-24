package neureka.calculus.environment.executors;

import neureka.calculus.environment.Type;

public class Convolution extends AbstractTypeExecutor<Type.OperatorCreator>
{
    public Convolution(String strConvolution, String strDeriviation, Type.OperatorCreator creator){
        super(strConvolution, strDeriviation, creator);
    }

    @Override
    public boolean canHandle(ExecutionCall call) {
        return true;
    }
}
