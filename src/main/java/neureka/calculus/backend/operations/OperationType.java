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
    public static List<AbstractOperationType> instances() {
        return OperationContext.instance().getRegister();
    }

    public static AbstractOperationType instance(int index) {
        return OperationContext.instance().getRegister().get(index);
    }

    public static AbstractOperationType[] ALL() {
        return OperationContext.instance().getRegister().toArray(new AbstractOperationType[0]);
    }

    public static int COUNT() {
        return OperationContext.instance().getID();
    }


    public static AbstractOperationType instance(String identifier) {
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
        T create(Tsr[] inputs, int d);
    }

    interface ScalarOperatorCreator<T> {
        T create(Tsr[] inputs, double scalar, int d);
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

    boolean isCommutative();

    boolean supports(Class implementation);


    public abstract double calculate(double[] inputs, int j, int d, List<Function> src);


    public static class Utility {
        public static Tsr[] _subset(Tsr[] tsrs, int padding, int index, int offset) {
            if (offset < 0) {
                index += offset;
                offset *= -1;
            }
            Tsr[] newTsrs = new Tsr[offset + padding];
            System.arraycopy(tsrs, index, newTsrs, padding, offset);
            return newTsrs;
        }

        public static Tsr[] _without(Tsr[] tsrs, int index) {
            Tsr[] newTsrs = new Tsr[tsrs.length - 1];
            for (int i = 0; i < newTsrs.length; i++) newTsrs[i] = tsrs[i + ((i < index) ? 0 : 1)];
            return newTsrs;
        }

        public static Tsr[] _offsetted(Tsr[] tsrs, int offset) {
            Tsr[] newTsrs = new Tsr[tsrs.length - offset];
            newTsrs[0] = Tsr.Create.newTsrLike(tsrs[1]);
            if (!tsrs[1].has(GraphNode.class) && tsrs[1] != tsrs[0]) {//Deleting intermediate results!
                tsrs[1].delete();
                tsrs[1] = null;
            }
            if (!tsrs[2].has(GraphNode.class) && tsrs[2] != tsrs[0]) {//Deleting intermediate results!
                tsrs[2].delete();
                tsrs[2] = null;
            }
            System.arraycopy(tsrs, 1 + offset, newTsrs, 1, tsrs.length - 1 - offset);
            newTsrs[1] = tsrs[0];
            return newTsrs;
        }

    }

}
