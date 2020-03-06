package neureka.calculus.environment;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;

public interface Type {
    
    //==================================================================================================================

    String getName();

    //-----------------

    OperationType.OperationCreator getActivationCreator();

    String getActivationOperationAsString();

    String getActivationDeriviationAsString();

    //-----------------

    OperationType.ScalarOperationCreator getScalarOperationCreator();

    String getScalarOperationAsString();

    String getScalarDeriviationAsString();
        
    //-----------------

    OperationType.OperationCreator getBroadcastOperationCreator();
    
    String getBroadcastOperationAsString();

    String getBroadcastDeriviationAsString();

    //==================================================================================================================

    int id();
    
    String identifier();
    
    boolean isOperation();
    
    boolean isFunction();
    
    boolean isIndexer();
    
    boolean isConvection();
    
    boolean isCommutative();
    
    boolean supportsScalar();

    boolean allowsForward(Tsr[] inputs);

    ADAgent getADAgentOf(Function f, Tsr[] inputs, int i, boolean forward);






}
