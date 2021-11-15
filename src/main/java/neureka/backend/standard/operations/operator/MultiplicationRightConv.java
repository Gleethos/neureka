package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.CalcUtil;
import neureka.calculus.args.Arg;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.calculus.Function;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public class MultiplicationRightConv extends AbstractOperation {


    public MultiplicationRightConv() {
        super(
                new OperationBuilder()
                        .setFunction(         "mul_conv_right" )
                        .setOperator(         "*" + ((char) 187)  )
                        .setArity(            3          )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( false      )
                        .setIsInline(         false      )
        );

        Broadcast xBroadcast = new Broadcast((executionCall, executor) -> null)
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor(
                        call -> {
                            Tsr<?> last = null;
                            for ( Tsr<?> t : call.getTensors() ) {
                                if ( last != null && !last.shape().equals(t.shape()) ) return false;
                                last = t; // Note: shapes are cached!
                            }
                            return true;
                        }
                )
                .setSupplyADAgentFor(
                        (Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                        {
                            Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                            Function mul = Neureka.get().context().getFunction().mul();
                            if ( ctxDerivative != null ) {
                                return ADAgent.of( ctxDerivative )
                                                .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) )
                                                .setBackward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) );
                            }
                            Tsr<?>[] inputs = call.getTensors();
                            int d = call.getDerivativeIndex();
                            if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                            else
                            {
                                Tsr<?> derivative = f.executeDerive( inputs, d );
                                return ADAgent.of( derivative )
                                                .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, derivative ) )
                                                .setBackward( (node, backwardError ) -> mul.execute( backwardError, derivative ) );
                            }
                        }
                )
                .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
                .setCallPreparation(
                        call -> {
                            Tsr<?>[] tsrs = call.getTensors();
                            int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                            return
                                    ExecutionCall.of(tsrs[offset], tsrs[1+offset]).andArgs(Arg.DerivIdx.of(-1)).running(Neureka.get().context().getOperation("idy")).on(call.getDevice());
                        }
                )
                .buildFunAlgorithm();

        setAlgorithm(
                Broadcast.class,
                xBroadcast.setImplementationFor(
                        CPU.class,
                        CPUImplementation
                            .withArity(3)
                            .andImplementation(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTsrOfType( Number.class, 0 ).size(),
                                                        (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                                ? ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ), call.getTsrOfType( Number.class, 2 ),
                                                                        call.getValOf( Arg.DerivIdx.class ), start, end,
                                                                        Multiplication.xBCCreatorX.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                                : ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ), call.getTsrOfType( Number.class, 2 ),
                                                                        call.getValOf( Arg.DerivIdx.class ), start, end,
                                                                        Multiplication.xBCCreator.create(call.getTensors(), call.getValOf( Arg.DerivIdx.class ))
                                                                )
                                                )
                            )
                        )
                        .setImplementationFor(
                                OpenCLDevice.class,
                                CLImplementation.compiler()
                                        .arity( 3 )
                                        .kernelSource( xBroadcast.getKernelSource() )
                                        .activationSource( "value = src1 * src2;\n" )
                                        .differentiationSource( "value += handle * drain;\n" )
                                        .kernelPostfix( this.getFunction() )
                                        .execution(
                                                call -> {
                                                    int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                                    int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                                    call.getDevice().getKernel(call)
                                                            .passAllOf( call.getTsrOfType( Number.class, offset ) )
                                                            .passAllOf( call.getTsrOfType( Number.class, offset + 1 ) )
                                                            .passAllOf( call.getTsrOfType( Number.class, offset + 2 ) )
                                                            .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                                            .pass( call.getValOf( Arg.DerivIdx.class ) )
                                                            .call( gwz );
                                                }
                                        )
                                        .build()
                        )
        );

    }

    @Override
    public String stringify(String[] children) {
        return null;
    }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return 0;
    }

}
