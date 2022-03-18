package neureka.backend.standard.operations.operator;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Broadcast;
import neureka.backend.standard.algorithms.Operator;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.backend.standard.operations.JunctionUtil;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Addition extends AbstractOperation {

    private final Broadcast _broadcast =
            (Broadcast)
                new Broadcast((executionCall, executor) -> null)
                .setCanPerformBackwardADFor( call -> true )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    {
                        Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                        assert ctxDerivative == null;
                        int d = call.getDerivativeIndex();
                        if ( forward ) throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                        else
                        {
                            Tsr<?> derivative = JunctionUtil.newTsrLike(call.input( d==0?1:0 ), 0);
                            Tsr<?> toBeDerived = JunctionUtil.newTsrLike(call.input( d ), 0);
                            Device device = call.getDeviceFor(Number.class);
                            return ADAgent.of( derivative )
                                            .setBackward(
                                                ( node, backwardError ) ->
                                                    this.getAlgorithm( Broadcast.class )
                                                         .getImplementationFor( device )
                                                         .runAndGetFirstTensor(
                                                                 ExecutionCall.of(
                                                                         toBeDerived.setIsVirtual(false),
                                                                         derivative,
                                                                         backwardError
                                                                 )
                                                                 .andArgs( Arg.DerivIdx.of(d) )
                                                                 .running( this )
                                                                 .on( device )
                                                         )
                                            );
                        }
                    }
                )
                .buildFunAlgorithm();

    public Addition()
    {
        super (
                new OperationBuilder()
                        .setIdentifier(         "add"      )
                        .setOperator(         "+"        )
                        .setArity(            -1         )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( true       )
                        .setIsInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        Operator operator = new Operator(JunctionUtil::forAdditions)
                                    .setSupplyADAgentFor( getDefaultAlgorithm() )
                                    .buildFunAlgorithm();

        setAlgorithm(
            operator
                .setImplementationFor(
                    CPU.class,
                    Operator.implementationForCPU()
                        .with(Fun.F64F64ToF64.triple(
                            ( a, b ) -> a + b,
                            ( a, b ) -> 1d, // Deriving at input 0
                            ( a, b ) -> 1d  // deriving input 1
                        ))
                        .with(Fun.F32F32ToF32.triple(
                            ( a, b ) -> a + b,
                            ( a, b ) -> 1f, // Deriving at input 0
                            ( a, b ) -> 1f  // deriving input 1
                        ))
                        .with(Fun.I32I32ToI32.triple(
                            ( a, b ) -> a + b,
                            ( a, b ) -> 1, // Deriving at input 0
                            ( a, b ) -> 1  // deriving input 1
                        ))
                        .get()
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    Operator.implementationForGPU( this.getIdentifier() )
                                    .with( "output = input1 + input2;\n" )
                                    .and( "output = 1;\n" )
            )
        );

        //________________
        // BROADCASTING :

        setAlgorithm(
                _broadcast
                .setImplementationFor(
                    CPU.class,
                    Broadcast.implementationForCPU()
                            .with(Fun.F64F64ToF64.triple(
                                ( a, b ) -> a + b,
                                ( a, b ) -> a + b, // Deriving at input 0
                                ( a, b ) -> a + b  // deriving input 1
                            ))
                            .with(Fun.F32F32ToF32.triple(
                                ( a, b ) -> a + b,
                                ( a, b ) -> a + b, // Deriving at input 0
                                ( a, b ) -> a + b  // deriving input 1
                            ))
                            .get()
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    Broadcast.implementationForGPU( this.getIdentifier() )
                            .with( "value += src1 + src2;\n" )
                            .and( "value += src1 + src2;\n" )
                )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        Scalarization scalarization = new Scalarization()
                                            .setSupplyADAgentFor( getDefaultAlgorithm() )
                                            .setExecutionDispatcher( (caller, call) -> CalcUtil.executeFor( caller, call, JunctionUtil::forAdditions ) )
                                            .buildFunAlgorithm();

        setAlgorithm(
            scalarization.setImplementationFor(
                CPU.class,
                CPUImplementation
                    .withArity(3)
                    .andImplementation(
                        call -> {
                            assert call.arity() == 3;
                            if ( call.getDerivativeIndex() == 0 )
                                call.setInput( 0, Tsr.of( call.input( 1 ).shape(), 1d ).getUnsafe().setIsIntermediate( true ) );
                            else if ( call.getDerivativeIndex() == 1 )
                                call.setInput( 0, Tsr.of( call.input( 2 ).shape(), 1d ).getUnsafe().setIsIntermediate( true ) );
                            else {
                                Scalarization.implementationForCPU()
                                    .with(Fun.F64F64ToF64.triple(
                                        ( a, b ) -> a + b,
                                        ( a, b ) ->  1, // Deriving at input 0
                                        ( a, b ) ->  1 // deriving input 1
                                    ))
                                   .with(Fun.F32F32ToF32.triple(
                                        ( a, b ) -> a + b,
                                        ( a, b ) ->  1, // Deriving at input 0
                                        ( a, b ) ->  1 // deriving input 1
                                   ))
                                   .with(Fun.I32I32ToI32.triple(
                                        ( a, b ) -> a + b,
                                        ( a, b ) ->  1, // Deriving at input 0
                                        ( a, b ) ->  1 // deriving input 1
                                   ))
                                   .get()
                                   .run( call );
                                }
                            }
                        )
            )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation
                    .compiler()
                    .arity( 3 )
                    .kernelSource( scalarization.getKernelSource() )
                    .activationSource( "output = input1 + value;\n" )
                    .differentiationSource( "output = 1;\n" )
                    .kernelPostfix( this.getIdentifier() )
                    .execution(
                        call -> {
                            assert call.arity() == 3;
                            if ( call.getDerivativeIndex() == 0 )
                                call.setInput( 0, Tsr.of( call.input(1).shape(), 1d ).getUnsafe().setIsIntermediate( true ) );
                            else if ( call.getDerivativeIndex() == 1 )
                                call.setInput( 0, Tsr.of( call.input( 2 ).shape(), 1d ).getUnsafe().setIsIntermediate( true ) );
                            else {
                                int gwz = call.getTsrOfType(Number.class, 0).size();
                                float value = call.getTsrOfType(Number.class, 2).getValueAt(0).floatValue();
                                call.getDevice()
                                    .getKernel(call)
                                    .passAllOf(call.getTsrOfType(Number.class, 0))
                                    .passAllOf(call.getTsrOfType(Number.class, 1))
                                    .pass(value)
                                    .pass(call.getTsrOfType(Number.class, 0).rank())
                                    .pass(call.getValOf(Arg.DerivIdx.class))
                                    .call(gwz);
                            }
                        }
                    )
                    .build()
            )
        );
    }

    @Contract(pure = true)
    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" + ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        String s =  Arrays.stream( children )
                .filter( child -> child.dependsOn(derivationIndex) )
                .map( child -> child.getDerivative(derivationIndex) )
                .map( Object::toString )
                .collect( Collectors.joining( " + " ) );
        return s;
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result += current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( Function function : src )
                derivative += function.derive(inputs, d, j);

            return derivative;
        }
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result += current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( Function function : src ) {
                derivative += function.derive( inputs, d );
            }
            return derivative;
        }
    }




}
