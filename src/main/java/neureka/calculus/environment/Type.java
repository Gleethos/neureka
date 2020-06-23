package neureka.calculus.environment;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.executors.AbstractTypeExecutor;
import neureka.calculus.environment.executors.Execution;
import neureka.calculus.environment.executors.TypeExecutor;

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

    Execution generateExecutionFrom(TypeExecutor.ExecutionCall call);

    //==================================================================================================================

    String getName();

    //==================================================================================================================

    <T extends AbstractTypeExecutor> T get(Class<T> type );
    <T extends AbstractTypeExecutor> boolean supports(Class<T> type );
    <T extends AbstractTypeExecutor> Type set(Class<T> type, T instance );



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
