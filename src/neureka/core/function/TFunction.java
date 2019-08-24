
package neureka.core.function;

import neureka.core.T;
import neureka.core.autograd.TGradientNode;
import neureka.core.function.factory.worker.TFunctionBuilder;
import neureka.core.function.factory.environment.ResultCache;
import neureka.core.function.factory.environment.FunctionCache;

import java.util.Random;

public interface TFunction
{
	FunctionCache F_CACHE = new FunctionCache();
	ResultCache R_CACHE = new ResultCache();
	//------------------------------------------------------------------------------------------------------------------

    static T execute(T drain, T[] tensors, String operation){
		TFunction function = TFunctionBuilder.newBuild(operation, true);
		return execute(drain, tensors, function);
	}

	static T execute(T drain, T[] tensors, TFunction function){
		//TLock lock = new TLock(function, tensors);
		TLock newGid = new TLock(function, tensors);//Random().nextLong();
		for(T t : tensors){
			t.add(new TGradientNode(t, null, null,  newGid));
		}
		drain.inject(function.activate(tensors));
		if(drain.has(TGradientNode.class)){
			((TGradientNode)drain.find(TGradientNode.class)).trimTree(null);//TODO implement!! test
		}
		TFunction.R_CACHE.free(tensors);
		return drain;
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

 