
package neureka.core.function;

import neureka.core.T;
import neureka.core.autograd.TGradientNode;
import neureka.core.function.factory.worker.TFunctionBuilder;
import neureka.core.function.factory.environment.ResultCache;
import neureka.core.function.factory.environment.FunctionCache;

public interface TFunction
{
	FunctionCache F_CACHE = new FunctionCache();
	ResultCache R_CACHE = new ResultCache();
	//------------------------------------------------------------------------------------------------------------------

    static void execute(T drain, T[] tensors, String operation){
		TFunction function = TFunctionBuilder.newBuild(operation, true);
		execute(drain, tensors, function);
	}

	static void execute(T drain, T[] tensors, TFunction function){
		TLock lock = new TLock(function, tensors);
		for(T t : tensors){
			t.add(lock);
		}
		drain.internalize(function.activate(tensors));
		if(drain.has(TGradientNode.class)){
			((TGradientNode)drain.find(TGradientNode.class)).trimTree(null);//TODO implement!! test
		}
		TFunction.R_CACHE.free(tensors);
	}
	//------------------------------------------------------------------------------------------------------------------
	TFunction newBuild(String expression);
	boolean isFlat();
	int id();
	String type();
	//------------------------------------------------------------------------------------------------------------------
	double activate(double[] input, int j);// Iteration over input via j !
	double activate(double[] input);
	double derive(double[] input, int index, int j);
	double derive(double[] input, int index);
	//------------------------------------------------------------------------------------------------------------------
	T activate(T[] input, int j);// Iteration over input via j !
	T activate(T[] input);
	T derive(T[] input, int index, int j);
	T derive(T[] input, int index);
	//------------------------------------------------------------------------------------------------------------------
	String toString();
}

 