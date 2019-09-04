package neureka.core.function.environment;

import neureka.core.function.IFunction;

import java.util.HashMap;

public class FunctionCache
{
    public final double BIAS = 0;
    public final double INCLINATION = 1;
    public final double RELU_INCLINATION = 0.01;
    public final String[] REGISTER = new String[]{
            "relu", "sig", "tanh", "quad", "lig", "lin", "gaus", "abs", "sin", "cos",
            "sum", "prod",
            "^", "/", "*", "%", "-", "+", "x", ""+((char)171), ""+((char)187), ","
            // (char)187 //>>
            // (char)171 //<<
    };
    public synchronized HashMap<String, IFunction> FUNCTIONS(){
        return this.FUNCTIONS;
    }

    private final HashMap<String, IFunction> FUNCTIONS = new HashMap<>();

}
