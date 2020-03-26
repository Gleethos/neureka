package neureka.calculus.environment;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;

public interface Type
{
    interface Operator {
        double execute(int[] t0Idx, int[] t1Idx, int[] t2Idx);
    }

    interface OperationCreator{
        Operator create(Tsr[] inputs, int d);
    }
    interface ScalarOperationCreator {
        Operator create(Tsr[] inputs, double scalar, int d);
    }

    abstract class TypeComponent<CreatorType>
    {
        protected String _operation;
        protected String _deriviation;
        protected CreatorType _creator;

        TypeComponent(String operation, String deriviation, CreatorType creator){
            _operation = operation;
            _deriviation = deriviation;
            _creator = creator;
        }
        public String getAsString(){
            return _operation;
        }
        public String getDeriviationAsString(){
            return _deriviation;
        }
        public CreatorType getCreator(){
            return _creator;
        }
    }

    //==================================================================================================================

    class Activation extends TypeComponent<OperationType.OperationCreator>
    {
        public Activation(String strActivation, String strDeriviation, OperationType.OperationCreator creator){
            super(strActivation, strDeriviation, creator);
        }
    }

    class Convolution extends TypeComponent<OperationType.OperationCreator>
    {
        public Convolution(String strConvolution, String strDeriviation, OperationType.OperationCreator creator){
            super(strConvolution, strDeriviation, creator);
        }
    }

    class Broadcast extends TypeComponent<OperationType.OperationCreator>
    {
        public Broadcast(String strBroadcast, String strDeriviation, OperationType.OperationCreator creator){
            super(strBroadcast, strDeriviation, creator);
        }
    }

    class Scalarization extends TypeComponent<OperationType.ScalarOperationCreator>
    {
        public Scalarization(String strScalarized, String strDeriviation, OperationType.ScalarOperationCreator creator){
            super(strScalarized, strDeriviation, creator);
        }
    }

    class Operation extends TypeComponent<OperationType.OperationCreator>
    {
        public Operation(String strOperation, String strDeriviation, OperationType.OperationCreator creator){
            super(strOperation, strDeriviation, creator);
        }
    }

    //==================================================================================================================

    Activation getActivation();

    boolean supportsActivation();

    //-----------------

    Scalarization getScalarization();

    boolean supportsScalar();

    //-----------------

    Convolution getConvolution();

    boolean supportsConvolution();

    //-----------------

    Broadcast getBroadcast();

    boolean supportsBroadcast();

    //-----------------

    Operation getOperation();

    boolean supportsOperation();

    //==================================================================================================================

    String getName();

    //-----------------

    int id();
    
    String identifier();
    
    boolean isOperation();

    boolean isIndexer();
    
    boolean isConvection();
    
    boolean isCommutative();

    boolean allowsForward(Tsr[] inputs);

    ADAgent getADAgentOf(Function f, Tsr[] inputs, int i, boolean forward);






}
