package neureka.calculus.environment;

import neureka.Tsr;
import neureka.calculus.environment.implementations.AbstractOperationTypeImplementation;
import neureka.calculus.environment.operations.OperationContext;

import java.util.List;
import java.util.function.Consumer;

public interface OperationType
{
    public static List<AbstractOperationType> instances(){
        return OperationContext.instance().getRegister();
    }

    public static AbstractOperationType instance(int index ) {
        return OperationContext.instance().getRegister().get(index);
    }

    public static AbstractOperationType[] ALL(){
        return OperationContext.instance().getRegister().toArray(new AbstractOperationType[0]);
    }

    public static int COUNT(){
        return OperationContext.instance().getID();
    }


    public static AbstractOperationType instance(String identifier){
        return OperationContext.instance().getLookup().getOrDefault( identifier, null );
    }

    interface TertiaryNDXConsumer {
        double execute( int[] t0Idx, int[] t1Idx, int[] t2Idx );
    }

    interface SecondaryNDXConsumer {
        double execute( int[] t0Idx, int[] t1Idx );
    }

    interface PrimaryNDXConsumer {
        double execute( int[] t0Idx );
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

    <T extends AbstractOperationTypeImplementation> T getImplementation(Class<T> type );
    <T extends AbstractOperationTypeImplementation> boolean supportsImplementation(Class<T> type );
    <T extends AbstractOperationTypeImplementation> OperationType setImplementation(Class<T> type, T instance );

    OperationType forEachImplementation(Consumer<OperationTypeImplementation> action );

    //==================================================================================================================

    interface Stringifier{
        String asString( List<String> children );
    }

    //---

    OperationType setStringifier(Stringifier stringifier );

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

}
