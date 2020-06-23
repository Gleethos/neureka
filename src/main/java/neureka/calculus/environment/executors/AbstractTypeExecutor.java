package neureka.calculus.environment.executors;


import neureka.calculus.environment.Type;

import java.util.HashMap;
import java.util.Map;

public abstract class AbstractTypeExecutor<CreatorType> implements TypeExecutor
{
    protected String _operation;
    protected String _deriviation;
    protected CreatorType _creator;

    private Map<Class<Execution>, Execution> _executions;

    public AbstractTypeExecutor(String operation, String deriviation, CreatorType creator)
    {
        _operation = operation;
        _deriviation = deriviation;
        _creator = creator;
        _executions = new HashMap<>();
    }

    public String getAsString(){
        return _operation;
    }
    public String getDeriviationAsString(){
        return _deriviation;
    }
    public CreatorType getCreator(){
        return _creator;
    }


    @Override
    public boolean canHandle(ExecutionCall call) {
        return false;
    }

    @Override
    public void handle(ExecutionCall call) {

    }
}


