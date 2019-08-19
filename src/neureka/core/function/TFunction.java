
package neureka.core.function;

import neureka.core.T;
import neureka.core.function.autograd.TGradientNode;
import neureka.core.function.factory.TFunctionBuilder;

import java.util.HashMap;
import java.util.TreeMap;
import java.util.function.Supplier;

public interface TFunction
{
	class Variables{

        public static final double BIAS = 0;
		public static final double INCLINATION = 1;
		public static final double RELU_INCLINATION = 0.01;
		public static final String[] REGISTER;

		public static final HashMap<String, TFunction> FUNCTIONS;
		static {
			REGISTER = new String[]{
					"relu", "sig", "tanh", "quad", "lig", "lin", "gaus", "abs", "sin", "cos",
					"sum", "prod",
					"^", "/", "*", "%", "-", "+", "x", ","
			};
			FUNCTIONS = new HashMap<>();
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
	}

	class Cache
	{
		private static final TreeMap<TLock, TreeMap<TFunction, T>> CACHE;
		static {
			CACHE = new TreeMap<>((a, b)->(a.key()-b.key()));
		}

		public static void free(T[] input){
			for(T t : input){
				CACHE.remove(t.find(TLock.class));
				t.remove(TLock.class);
			}
		}
		public static T handle(T[] input, TFunction function, Supplier<T> activation){
			TLock lock = (TLock) input[0].find(TLock.class);
			T result = (!function.isFlat())? get(lock, function):null;
			if(result==null){
				result = activation.get();
				if(!function.isFlat()&&lock!=null) {
					put(result, lock, function);
				}
			}
			return result;
		}
		private static T get(TLock lock, TFunction function){//function and source
			if(Cache.CACHE.containsKey(lock)){
				if(CACHE.get(lock).containsKey(function)){
					return CACHE.get(lock).get(function);
				}
			}
			return null;
		}
		private static void put(T t, TLock lock, TFunction function){
			TreeMap<TFunction, T> variables = null;
			if(!CACHE.containsKey(lock)){
				variables = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());
				CACHE.put(lock, variables);
			}else{
				variables = CACHE.get(lock);
			}
			variables.put(function, t);
		}
	}

	static void execute(T drain, T[] tensors, String operation){
		TFunction function = TFunctionBuilder.newBuild(operation, true);
		execute(drain, tensors, function);
	}

	static void execute(T drain, T[] tensors, TFunction function){
		TLock lock = new TLock(function);
		for(T t : tensors){
			t.add(lock);
		}
		drain.internalize(function.activate(tensors));
		if(drain.has(TGradientNode.class)){
			((TGradientNode)drain.find(TGradientNode.class)).trimTree(null);//TODO implement!! test
		}
		TFunction.Cache.free(tensors);
	}

	public TFunction newBuild(String expression);
	public String toString();
	public boolean isFlat();
	public int id();
	public String type();
	//-------------------------------------------
	public double activate(double[] input, int j);// Iteration over input via j !
	public double activate(double[] input);
	//-----------------------------------------------------
	public double derive(double[] input, int index, int j);
	public double derive(double[] input, int index);
	//--------------------------------------------------------------
	//-------------------------------------------
	public T activate(T[] input, int j);// Iteration over input via j !
	public T activate(T[] input);
	//-----------------------------------------------------
	public T derive(T[] input, int index, int j);
	public T derive(T[] input, int index);
	//--------------------------------------------------------------
}

 