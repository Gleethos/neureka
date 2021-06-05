package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.api.operations.OperationContext;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.Function;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.devices.opencl.OpenCLDevice;

public class MultiplicationLeftConv extends AbstractOperation {

    public MultiplicationLeftConv() {
        super(
                new OperationBuilder()
                        .setFunction(         "mul_conv_left"    )
                        .setOperator(         ((char) 171) + "*"    )
                        .setArity(            3          )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( false        )
                        .setIsInline(         false       )
        );

        Broadcast xBroadcast = new Broadcast()
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
                ).setSupplyADAgentFor(
                        (Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                        {
                            Tsr ctxDerivative = (Tsr) call.getAt("derivative");
                            Function mul = Function.DETACHED().MUL();
                            if ( ctxDerivative != null ) {
                                return new DefaultADAgent( ctxDerivative )
                                        .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) )
                                        .setBackward( null );
                            }
                            Tsr[] inputs = call.getTensors();
                            int d = call.getDerivativeIndex();
                            if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                            else
                            {
                                Tsr deriv = f.derive( inputs, d );
                                return new DefaultADAgent( deriv )
                                        .setForward( ( node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, deriv } ) )
                                        .setBackward( ( node, backwardError ) -> mul.call( new Tsr[]{ backwardError, deriv } ) );
                            }
                        }
                )
                .setHandleInsteadOfDevice( (caller, call ) -> null )
                .setHandleRecursivelyAccordingToArity( (call, goDeeperWith ) -> null )
                .setInstantiateNewTensorsForExecutionIn(
                        call -> {
                            Tsr[] tsrs = call.getTensors();
                            int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                            return ExecutionCall.builder()
                                    .device( call.getDevice() )
                                    .tensors( new Tsr[]{tsrs[offset], tsrs[1+offset]} )
                                    .derivativeIndex( -1 )
                                    .operation( OperationContext.get().instance("idy") )
                                    .build();
                        }
                )
                .build();


        setAlgorithm(
                Broadcast.class,
                xBroadcast.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call ->
                                        call.getDevice().getExecutor()
                                                .threaded (
                                                        call.getTsrOfType( Number.class, 0 ).size(),
                                                        (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                                ? ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ), call.getTsrOfType( Number.class, 2 ),
                                                                        call.getDerivativeIndex(), start, end,
                                                                        Multiplication.xBCCreatorX.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                                : ( start, end ) ->
                                                                Broadcast.broadcast (
                                                                        call.getTsrOfType( Number.class, 0 ), call.getTsrOfType( Number.class, 1 ), call.getTsrOfType( Number.class, 2 ),
                                                                        call.getDerivativeIndex(), start, end,
                                                                        Multiplication.xBCCreator.create(call.getTensors(), call.getDerivativeIndex())
                                                                )
                                                ),
                                3
                        )
                ).setImplementationFor(
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
                                                    .pass( call.getTsrOfType( Number.class, offset ) )
                                                    .pass( call.getTsrOfType( Number.class, offset + 1 ) )
                                                    .pass( call.getTsrOfType( Number.class, offset + 2 ) )
                                                    .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                                    .pass( call.getDerivativeIndex() )
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
