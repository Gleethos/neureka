package neureka.core.function.factory.environment;

import neureka.core.function.TFunction;

import java.util.HashMap;

public class FCache {

    public final double BIAS = 0;
    public final double INCLINATION = 1;
    public final double RELU_INCLINATION = 0.01;
    public final String[] REGISTER = new String[]{
            "relu", "sig", "tanh", "quad", "lig", "lin", "gaus", "abs", "sin", "cos",
            "sum", "prod",
            "^", "/", "*", "%", "-", "+", "x", ","
    };

    public synchronized HashMap<String, TFunction> FUNCTIONS(){
        return this.FUNCTIONS;
    }

    private final HashMap<String, TFunction> FUNCTIONS = new HashMap<>();
			/*
			case 0:  return "relu";
			case 1:  return "sig";
			case 2:  return "tanh";
			case 3:  return "quad";
			case 4:  return "lig";
			case 5:  return "lin";
			case 6:  return "gaus";
			case 7:  return "abs";
			case 8:  return "sin";
			case 9:  return "cos";

			case 10: return "sum";
			case 11: return "prod";

			case 12: return "^";
			case 13: return "/";
			case 14: return "*";
			case 15: return "%";
			case 16: return "-";
			case 17: return "+";

			case 18: return "conv";
			* */


}
