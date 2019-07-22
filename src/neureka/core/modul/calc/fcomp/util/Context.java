package neureka.core.modul.calc.fcomp.util;

import neureka.core.T;
import neureka.core.modul.calc.fcomp.Function;

import java.util.HashMap;

public class Context {
    public static final double shift = 0;
    public static final double inclination = 1;
    public static final double secondaryInclination = 0.01;
    public static final String[] register;
    public static final HashMap<Long, T> stack;
    public static final HashMap<String, Function> shared;
    static {
        register = new String[]{
                "relu", "sig", "tanh", "quad", "lig", "lin", "gaus", "abs", "sin", "cos",
                "sum", "prod",
                "^", "/", "*", "%", "-", "+", "x"
        };
        shared = new HashMap<>();
        stack = new HashMap<>();
    }
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
