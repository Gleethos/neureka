package neureka.calculus.environment;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.subtypes.AbstractTypeComponent;

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

    interface OperationPreprocessor
    {

    }

    //==================================================================================================================

    String getName();

    //==================================================================================================================

    <T> T get( Class<T> type );
    <T> boolean supports( Class<T> type );
    <T> Type set( Class<T> type, T instance );



    //==================================================================================================================

    int id();
    
    String identifier();

    int numberOfParameters();

    boolean isOperation();

    boolean isIndexer();
    
    boolean isConvection();
    
    boolean isCommutative();

    boolean allowsForward(Tsr[] inputs);

    ADAgent getADAgentOf(Function f, Tsr[] inputs, int i, boolean forward);






}
