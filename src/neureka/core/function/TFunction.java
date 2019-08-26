
package neureka.core.function;

import neureka.core.T;
import neureka.core.function.autograd.TGraphLock;
import neureka.core.function.autograd.TGraphNode;
import neureka.core.function.factory.worker.FBuilder;
import neureka.core.function.factory.environment.TCache;
import neureka.core.function.factory.environment.FCache;

public interface TFunction
{
	FCache F_CACHE = new FCache();
	TCache R_CACHE = new TCache();
	//------------------------------------------------------------------------------------------------------------------

    static T execute(T drain, T[] tensors, String operation){
		TFunction function = FBuilder.newBuild(operation, true);
		return execute(drain, tensors, function);
	}

	static T execute(T drain, T[] tensors, TFunction function){
		//TGraphLock lock = new TGraphLock(function, tensors);
		TGraphLock newGid = new TGraphLock(function, tensors);//Random().nextLong();
		for(T t : tensors){
			t.add(new TGraphNode(t, null, null,  newGid));
		}
		drain.inject(function.activate(tensors));
		if(drain.has(TGraphNode.class)){
			((TGraphNode)drain.find(TGraphNode.class)).trimTree(null);//TODO implement!! test
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

 