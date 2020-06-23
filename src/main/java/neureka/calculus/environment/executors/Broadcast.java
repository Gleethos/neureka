package neureka.calculus.environment.executors;

import neureka.calculus.environment.Type;

public class Broadcast extends AbstractTypeExecutor<Type.OperatorCreator>
{
    public Broadcast(String strBroadcast, String strDeriviation, Type.OperatorCreator creator){
        super(strBroadcast, strDeriviation, creator);
    }
}
