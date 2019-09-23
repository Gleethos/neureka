
package neureka.core.function;

import neureka.core.T;
import neureka.core.function.environment.Types;
import neureka.core.function.factory.autograd.GraphLock;
import neureka.core.function.factory.autograd.GraphNode;
import neureka.core.function.factory.assembly.FunctionBuilder;
import neureka.core.function.environment.Cache;

public interface IFunction {
    Cache CACHE = new Cache();
    Types TYPES = new Types();

    //------------------------------------------------------------------------------------------------------------------
    class setup {
        public static T commit(T drain, T[] tensors, String operation, boolean doAD) {
            return commit(drain, tensors, FunctionBuilder.build(operation, doAD));
        }

        public static T commit(T drain, T[] tensors, IFunction function) {
            GraphLock newGid = new GraphLock(function, tensors);
            for (T t : tensors) {
                t.add(new GraphNode(t, null, null, newGid));
            }
            drain.inject(function.activate(tensors));
            if (drain.has(GraphNode.class)) {
                ((GraphNode) drain.find(GraphNode.class)).trimTree(null);
            }
            IFunction.CACHE.free(tensors);
            for (T t : tensors) {
                t.setGradientIsTargeted(false);
            }
            return drain;
        }
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

 