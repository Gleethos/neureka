package neureka.core.function.environment;

import neureka.core.function.IFunction;

import java.util.HashMap;

public class FunctionCache
{
    private final HashMap<String, IFunction> FUNCTIONS = new HashMap<>();

    public synchronized HashMap<String, IFunction> FUNCTIONS(){
        return this.FUNCTIONS;
    }
    
}
