package neureka.core.function.environment;

import neureka.core.function.IFunction;

import java.util.HashMap;

public class FunctionCache
{
    public synchronized HashMap<String, IFunction> FUNCTIONS(){
        return this.FUNCTIONS;
    }

    private final HashMap<String, IFunction> FUNCTIONS = new HashMap<>();

}
