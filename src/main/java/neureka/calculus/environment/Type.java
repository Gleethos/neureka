package neureka.calculus.environment;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.executors.AbstractOperationTypeImplementation;

public interface Type
{
    interface DefaultOperator {
        double execute(int[] t0Idx, int[] t1Idx, int[] t2Idx);
    }

    interface OperatorCreator {
        DefaultOperator create(Tsr[] inputs, int d);
    }
    interface ScalarOperatorCreator {
        DefaultOperator create(Tsr[] inputs, double scalar, int d);
    }

    //==================================================================================================================

    OperationTypeImplementation executorOf(OperationTypeImplementation.ExecutionCall call);

    //==================================================================================================================

    String getName();

    //==================================================================================================================

    <T extends AbstractOperationTypeImplementation> T getImplementation(Class<T> type );
    <T extends AbstractOperationTypeImplementation> boolean supportsImplementation(Class<T> type );
    <T extends AbstractOperationTypeImplementation> Type setImplementation(Class<T> type, T instance );



    //==================================================================================================================

    int id();
    
    String identifier();

    /**
     * Arity is the number of arguments or operands
     * that this function or operation takes.
     */
    int arity();

    boolean isOperation();

    boolean isIndexer();
    
    boolean isConvection();
    
    boolean isCommutative();

    boolean allowsForward(Tsr[] inputs);

    ADAgent getADAgentOf(Function f, Tsr[] inputs, int i, boolean forward);






}
