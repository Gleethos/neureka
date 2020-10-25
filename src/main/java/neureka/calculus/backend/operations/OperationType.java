package neureka.calculus.backend.operations;

import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.calculus.Function;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.implementations.AbstractFunctionalOperationTypeImplementation;
import neureka.calculus.backend.implementations.OperationTypeImplementation;

import java.util.List;
import java.util.function.Consumer;

public interface OperationType
{
    static List<AbstractOperationType> instances() {
        return OperationContext.instance().getRegister();
    }

    static AbstractOperationType instance(int index) {
        return OperationContext.instance().getRegister().get(index);
    }

    static AbstractOperationType[] ALL() {
        return OperationContext.instance().getRegister().toArray(new AbstractOperationType[ 0 ]);
    }

    static int COUNT() {
        return OperationContext.instance().getID();
    }


    static AbstractOperationType instance(String identifier) {
        return OperationContext.instance().getLookup().getOrDefault(identifier, null);
    }

    interface TertiaryNDXConsumer {
        double execute(int[] t0Idx, int[] t1Idx, int[] t2Idx);
    }

    interface SecondaryNDXConsumer {
        double execute(int[] t0Idx, int[] t1Idx);
    }

    interface PrimaryNDXConsumer {
        double execute(int[] t0Idx);
    }

    //---

    interface DefaultOperatorCreator<T> {
        T create(Tsr<?>[] inputs, int d);
    }

    interface ScalarOperatorCreator<T> {
        T create(Tsr<?>[] inputs, double scalar, int d);
    }

    //==================================================================================================================

    OperationTypeImplementation implementationOf(ExecutionCall call);

    //==================================================================================================================

    String getFunction();

    //==================================================================================================================

    <T extends AbstractFunctionalOperationTypeImplementation> T getImplementation(Class<T> type);

    <T extends AbstractFunctionalOperationTypeImplementation> boolean supportsImplementation(Class<T> type);

    <T extends AbstractFunctionalOperationTypeImplementation> OperationType setImplementation(Class<T> type, T instance);

    OperationType forEachImplementation(Consumer<OperationTypeImplementation> action);

    //==================================================================================================================

    interface Stringifier {
        String asString(List<String> children);
    }

    //---

    OperationType setStringifier(Stringifier stringifier);

    Stringifier getStringifier();

    //==================================================================================================================

    int getId();

    String getOperator();

    /**
     * Arity is the number of arguments or operands
     * that this function or operation takes.
     */
    int getArity();

    boolean isOperator();

    boolean isIndexer();

    boolean isDifferentiable();

    boolean isInline();

    boolean supports(Class implementation);

    /**
     * This method mainly ought to serve as a reference- and fallback- implementation for tensor backends and also
     * as the backend for handling the calculation of scalar inputs passed to a given abstract syntax tree of
     * Function instances...
     * ( (almost) every Function instance contains an OperationType reference to which it passes scalar executions... )
     *
     * This is also the reason why the last parameter of this method is a list of Function objects :
     * The list stores the child nodes of the Function node that is currently being processed.
     * Therefore when implementing this method one should first call the child nodes in
     * order to get the "real inputs" of this current node.
     *
     * One might ask : Why does that not happen automatically?
     * The answer is to that question lies in the other parameters of this method.
     * Specifically the parameter "d" !
     * This argument determines if the derivative ought to be calculated and
     * also which value is being targeted within the input array.
     * Depending on this variable and also the nature of the operation,
     * the execution calls to the child nodes of this node change considerably!
     *
     *
     * @param inputs An array of scalar input variables.
     * @param j The index variable for indexed execution on the input array. (-1 if no indexing should occur)
     * @param d The index of the variable of which a derivative ought to be calculated.
     * @param src The child nodes of the Function node to which this very OperationType belongs.
     * @return The result of the calculation.
     */
    double calculate( double[] inputs, int j, int d, List<Function> src );


    /**
     *  This static utility class contains simple methods used for creating slices of plain old
     *  arrays of tensor objects...
     *  These slices may be used for many reasons, however mainly when iterating over
     *  inputs to a Function recursively in order to execute them pairwise for example...
     */
    class Utility
    {
        public static Tsr<?>[] subset(Tsr<?>[] tsrs, int padding, int index, int offset) {
            if (offset < 0) {
                index += offset;
                offset *= -1;
            }
            Tsr<?>[] newTsrs = new Tsr[offset + padding];
            System.arraycopy(tsrs, index, newTsrs, padding, offset);
            return newTsrs;
        }

        public static Tsr<?>[] without(Tsr<?>[] tsrs, int index) {
            Tsr<?>[] newTsrs = new Tsr[tsrs.length - 1];
            for (int i = 0; i < newTsrs.length; i++) newTsrs[ i ] = tsrs[i + ((i < index) ? 0 : 1)];
            return newTsrs;
        }

        public static Tsr<?>[] offsetted(Tsr<?>[] tsrs, int offset) {
            Tsr<?>[] newTsrs = new Tsr[tsrs.length - offset];
            newTsrs[ 0 ] = Tsr.Create.newTsrLike(tsrs[ 1 ]);
            if (!tsrs[ 1 ].has(GraphNode.class) && tsrs[ 1 ] != tsrs[ 0 ]) {//Deleting intermediate results!
                tsrs[ 1 ].delete();
                tsrs[ 1 ] = null;
            }
            if (!tsrs[ 2 ].has(GraphNode.class) && tsrs[ 2 ] != tsrs[ 0 ]) {//Deleting intermediate results!
                tsrs[ 2 ].delete();
                tsrs[ 2 ] = null;
            }
            System.arraycopy(tsrs, 1 + offset, newTsrs, 1, tsrs.length - 1 - offset);
            newTsrs[ 1 ] = tsrs[ 0 ];
            return newTsrs;
        }

    }

}
