package neureka.calculus.environment.subtypes;

import neureka.calculus.environment.Type;

public class Broadcast extends AbstractTypeComponent<Type.OperatorCreator>
{
    public Broadcast(String strBroadcast, String strDeriviation, Type.OperatorCreator creator){
        super(strBroadcast, strDeriviation, creator);
    }
}
