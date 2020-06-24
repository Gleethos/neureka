package neureka.calculus.environment.executors;

import neureka.calculus.environment.Type;

public class Scalarization extends AbstractTypeExecutor<Scalarization, Type.ScalarOperatorCreator>
{
    public Scalarization(String strScalarized, String strDeriviation, Type.ScalarOperatorCreator creator){
        super(strScalarized, strDeriviation, creator);
    }

    @Override
    public boolean canHandle(ExecutionCall call) {
        return true;
    }
}