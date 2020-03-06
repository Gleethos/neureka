package neureka.calculus.environment;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;

import java.net.Proxy;

public interface Type
{
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

    //==================================================================================================================

    String getName();

    //-----------------

    //Activation getActivation();

    OperationType.OperationCreator getActivationCreator();

    String getActivationOperationAsString();

    String getActivationDeriviationAsString();

    //-----------------

    //Scalarization getScalarization();

    OperationType.ScalarOperationCreator getScalarOperationCreator();

    String getScalarOperationAsString();

    String getScalarDeriviationAsString();
        
    //-----------------

    //Convolution getConvolution();

    OperationType.OperationCreator getConvolveOperationCreator();
    
    String getConvolveOperationAsString();

    String getConvolveDeriviationAsString();

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
