
package neureka.core.function;

import neureka.core.T;
import neureka.core.function.factory.autograd.GraphLock;
import neureka.core.function.factory.autograd.GraphNode;
import neureka.core.function.factory.assembly.FunctionGraphBuilder;
import neureka.core.function.environment.TensorCache;
import neureka.core.function.environment.FunctionCache;

import java.util.HashMap;
import java.util.function.Consumer;
import java.util.function.Supplier;

public interface IFunction
{
    FunctionCache F_CACHE = new FunctionCache();
    TensorCache T_CACHE = new TensorCache();
    String[] REGISTER = new String[]
    {
            "relu", "sig", "tanh", "quad", "lig", "lin", "gaus", "abs", "sin", "cos",
            "sum", "prod",
            "^", "/", "*", "%", "-", "+", "x", ""+((char)171), ""+((char)187), ","
            // (char)171 -> <<    // (char)187 -> >>
    };
    Supplier<HashMap<String, Integer>> setup = ()->{
        HashMap<String, Integer> lookup = new HashMap<>();
        for(int i=0; i<REGISTER.length; i++){
            lookup.put(REGISTER[i], i);
        }
        return lookup;
    };
    HashMap<String, Integer> LOOKUP = setup.get();

    //------------------------------------------------------------------------------------------------------------------

    static T execute(T drain, T[] tensors, String operation, boolean doAD) {
        return execute(drain, tensors, FunctionGraphBuilder.newBuild(operation, doAD));
    }

    static T execute(T drain, T[] tensors, IFunction function) {
        GraphLock newGid = new GraphLock(function, tensors);
        for (T t : tensors) {
            t.add(new GraphNode(t, null, null, newGid));
        }
        drain.inject(function.activate(tensors));
        if (drain.has(GraphNode.class)) {
            ((GraphNode) drain.find(GraphNode.class)).trimTree(null);
        }
        IFunction.T_CACHE.free(tensors);
        for(T t : tensors){
            t.setGradientIsTargeted(false);
        }
        return drain;
    }

    //------------------------------------------------------------------------------------------------------------------
    IFunction newBuild(String expression);

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

 