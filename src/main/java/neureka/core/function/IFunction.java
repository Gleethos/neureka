
package neureka.core.function;

import neureka.core.T;
import neureka.core.function.environment.Types;
import neureka.core.function.factory.autograd.GraphLock;
import neureka.core.function.factory.autograd.GraphNode;
import neureka.core.function.factory.assembly.FunctionBuilder;
import neureka.core.function.environment.Cache;

public interface IFunction
{
    Cache CACHE = new Cache();
    Types TYPES = new Types();
    //------------------------------------------------------------------------------------------------------------------

    class setup
    {
        public static T commit(T[] tensors, String operation, boolean doAD) {
            return commit(tensors, FunctionBuilder.build(operation, doAD));
        }

        public static T commit(T[] inputs, IFunction function) {
            GraphLock newGid = new GraphLock(function, inputs);
            for (T t : inputs) {
                t.add(new GraphNode(t, null, null, newGid));
            }
            T result = (function.activate(inputs));
            if (result.has(GraphNode.class)) {
                ((GraphNode) result.find(GraphNode.class)).trimTree(null);
            }
            IFunction.CACHE.free(inputs);
            for (T t : inputs) {
                t.setGradientIsTargeted(false);
            }
            return result;
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    IFunction newBuild(String expression);

    boolean isFlat();

    int id();

    String type();

    //------------------------------------------------------------------------------------------------------------------
    double activate(double[] inputs, int j);// Iteration over input via j !

    double activate(double[] inputs);

    double derive(double[] inputs, int index, int j);

    double derive(double[] inputs, int index);

    //------------------------------------------------------------------------------------------------------------------
    T activate(T[] inputs, int j);// Iteration over input via j !

    T activate(T[] inputs);

    T derive(T[] inputs, int index, int j);

    T derive(T[] inputs, int index);

    //------------------------------------------------------------------------------------------------------------------
    String toString();
}

 