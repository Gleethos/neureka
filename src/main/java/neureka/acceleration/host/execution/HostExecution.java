package neureka.acceleration.host.execution;

import neureka.calculus.environment.Type;
import neureka.calculus.environment.executors.Execution;

public class HostExecution implements Execution<Type.OperatorCreator>
{
    private Type.OperatorCreator _creator;
    private int _arity;

    public HostExecution(Type.OperatorCreator creator, int arity){
        _creator = creator;
        _arity = arity;
    }

    @Override
    public Type.OperatorCreator getLambda() {
        return _creator;
    }

    @Override
    public int arity() {
        return _arity;
    }
}
