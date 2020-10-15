
package neureka.calculus.backend.operations;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.device.Device;
import neureka.device.host.execution.HostExecutor;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.Function;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.implementations.AbstractBaseOperationTypeImplementation;
import neureka.calculus.backend.implementations.AbstractFunctionalOperationTypeImplementation;
import neureka.calculus.backend.implementations.OperationTypeImplementation;
import neureka.calculus.frontend.AbstractFunction;
import neureka.calculus.frontend.assembly.FunctionBuilder;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Consumer;

public abstract class AbstractOperationType implements OperationType
{

    private Stringifier _stringifier;

    protected int _id;
    protected String _function;
    protected String _operator;
    /**
     * Arity is the number of arguments or operands
     * that this function or operation takes.
     */
    protected int _arity;
    protected boolean _isOperator;
    protected boolean _isIndexer;
    protected boolean _isDifferentiable;
    protected boolean _isInline;

    private final Map<Class, OperationTypeImplementation> _implementations = new LinkedHashMap<>();
    private final OperationTypeImplementation _defaultImplementation;

    public AbstractOperationType(
            String function,
            String operator,
            int arity,
            boolean isOperator,
            boolean isIndexer,
            boolean isDifferentiable,
            boolean isInline
    ) {
        _function = function;
        _arity = arity;
        _id = OperationContext.instance().getID();
        OperationContext.instance().incrementID();
        _operator = operator;
        _isOperator = isOperator;
        _isIndexer = isIndexer;
        _isDifferentiable = isDifferentiable;
        _isInline = isInline;

        OperationContext.instance().getRegister().add(this);
        OperationContext.instance().getLookup().put(operator, this);
        OperationContext.instance().getLookup().put(operator.toLowerCase(), this);
        if (
                operator
                        .replace((""+((char)171)), "")
                        .replace((""+((char)187)), "")
                        .matches("[a-z]")
        ) {
            if (operator.contains((""+((char)171)))) {
                OperationContext.instance().getLookup().put(operator.replace((""+((char)171)), "<<"), this);
            }
            if (operator.contains((""+((char)187)))) {
                OperationContext.instance().getLookup().put(operator.replace((""+((char)187)),">>"), this);
            }
        }


        _defaultImplementation = new AbstractBaseOperationTypeImplementation<OperationTypeImplementation>("default")
        {
            @Override
            public float isImplementationSuitableFor(ExecutionCall call) {
                int[] shape = null;
                for ( Tsr t : call.getTensors() ) {
                    if ( shape == null ) if ( t != null ) shape = t.getNDConf().shape();
                    else if ( t != null && !Arrays.equals( shape, t.getNDConf().shape() ) ) return 0.0f;
                }
                return 1.0f;
            }

            @Override
            public Device findDeviceFor(ExecutionCall call) {
                return null;
            }

            @Override
            public boolean canImplementationPerformForwardADFor(ExecutionCall call) {
                return true;
            }

            @Override
            public boolean canImplementationPerformBackwardADFor(ExecutionCall call) {
                return true;
            }

            @Override
            public ADAgent supplyADAgentFor(Function f, ExecutionCall<Device> call, boolean forward)
            {
                Tsr<?> ctxDerivative = (Tsr<?>) call.getAt("derivative");
                Function mul = Function.Detached.MUL;
                if ( ctxDerivative != null ) {
                    return new DefaultADAgent( ctxDerivative )
                            .withForward( ( node, forwardDerivative ) -> mul.call( new Tsr[]{forwardDerivative, ctxDerivative} ) )
                            .withBackward( ( node, backwardError ) -> mul.call( new Tsr[]{backwardError, ctxDerivative} ) );
                }
                Tsr<?> localDerivative = f.derive(call.getTensors(), call.getDerivativeIndex());
                return new DefaultADAgent( localDerivative )
                        .withForward( (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, localDerivative}) )
                        .withBackward( (node, backwardError) -> mul.call(new Tsr[]{backwardError, localDerivative}) );
            }

            @Override
            public Tsr handleInsteadOfDevice(AbstractFunction caller, ExecutionCall call) {
                return null;
            }

            @Override
            public Tsr handleRecursivelyAccordingToArity(ExecutionCall call, java.util.function.Function<ExecutionCall, Tsr> goDeeperWith)
            {
                return null;
            }

            @Override
            public ExecutionCall instantiateNewTensorsForExecutionIn(ExecutionCall call) {
                Tsr[] tensors = call.getTensors();
                Device device = call.getDevice();
                if ( tensors[0] == null ) // Creating a new tensor:
                {
                    int[] shp = tensors[1].getNDConf().shape();
                    Tsr output = new Tsr( shp, 0.0 );
                    output.setIsVirtual(false);
                    device.add(output);
                    tensors[0] = output;
                }
                return call;
            }
        }.setExecutor(
                HostExecutor.class,
                new HostExecutor(
                        call -> {
                            Function f = FunctionBuilder.build(this, call.getTensors().length-1, false);
                            double[] inputs = new double[call.getTensors().length-1];
                            call
                                .getDevice()
                                .getExecutor()
                                .threaded (
                                        call.getTensor(0).size(),
                                        ( start, end ) -> {
                                            for ( int i = start; i < end; i++ ) {
                                                for ( int ii = 0; ii < inputs.length; ii++ ) {
                                                    inputs[ii] = call.getTensor(1+ii).value64(i);
                                                }
                                                call.getTensor(0).value64()[i] = f.call(inputs);
                                            }
                                        }
                                );
                        },
                        _arity
                )
        );

    }

    //==================================================================================================================

    public OperationTypeImplementation defaultImplementation() {
        return _defaultImplementation;
    }

    //==================================================================================================================

    @Override
    public <T extends AbstractFunctionalOperationTypeImplementation> T getImplementation(Class<T> type){
        return (T) _implementations.get(type);
    }
    @Override
    public <T extends AbstractFunctionalOperationTypeImplementation> boolean supportsImplementation(Class<T> type){
        return _implementations.containsKey(type);
    }
    @Override
    public <T extends AbstractFunctionalOperationTypeImplementation> OperationType setImplementation(Class<T> type, T instance) {
        _implementations.put(type, instance);
        return this;
    }

    @Override
    public OperationType forEachImplementation(Consumer<OperationTypeImplementation> action ){
        _implementations.values().forEach(action);
        return this;
    }

    //==================================================================================================================

    @Override
    public OperationType setStringifier(Stringifier stringifier) {
        _stringifier = stringifier;
        return this;
    }

    @Override
    public Stringifier getStringifier() {
        return _stringifier;
    }

    //==================================================================================================================

    @Override
    public OperationTypeImplementation implementationOf( ExecutionCall call ) {
        float bestScore = 0f;
        OperationTypeImplementation bestImpl = null;
        for( OperationTypeImplementation impl : _implementations.values() ) {
            float currentScore = impl.isImplementationSuitableFor( call );
            if ( currentScore > bestScore ) {
                if ( currentScore == 1.0 ) return impl;
                else {
                    bestScore = currentScore;
                    bestImpl = impl;
                }
            }
        }
        return bestImpl;
    }

    //==================================================================================================================

    @Override
    public String getFunction(){
        return _function;
    }

    @Override
    public String getOperator(){
        return _operator;
    }

    @Override
    public int getId(){
        return _id;
    }

    @Override
    public int getArity(){
        return _arity;
    }

    @Override
    public boolean isOperator() {
        return _isOperator;
    }

    @Override
    public boolean isIndexer(){
        return _isIndexer;
    }

    @Override
    public boolean isDifferentiable(){
        return _isDifferentiable;
    }

    @Override
    public boolean isInline(){
        return _isInline;
    }

    @Override
    public boolean supports(Class implementation) {
        return _implementations.containsKey(implementation);
    }

}
