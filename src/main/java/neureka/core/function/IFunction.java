
package neureka.core.function;

import neureka.core.Tsr;
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
        public static Tsr commit(Tsr[] tensors, String operation, boolean doAD) {
            return commit(tensors, FunctionBuilder.build(operation, doAD));
        }

        public static Tsr commit(Tsr[] inputs, IFunction function) {
            GraphLock newGid = new GraphLock(function, inputs);
            for (Tsr t : inputs) {
                t.add(new GraphNode(t, null, null, newGid));
            }
            Tsr result = (function.activate(inputs));
            if (result.has(GraphNode.class)) {
                ((GraphNode) result.find(GraphNode.class)).trimTree(null);
            }
            IFunction.CACHE.free(newGid);
            for (Tsr t : inputs) {
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
    Tsr activate(Tsr[] inputs, int j);// Iteration over input via j !

    Tsr activate(Tsr[] inputs);

    Tsr derive(Tsr[] inputs, int index, int j);

    Tsr derive(Tsr[] inputs, int index);

    //------------------------------------------------------------------------------------------------------------------
    String toString();
}

 